const std = @import("std");
const F = @import("../fields/mersenne31.zig").Mersenne31;
const stack = @import("stack.zig");
const trace_mod = @import("trace.zig");
const constraints = @import("constraints.zig");
const Basefold = @import("../pcs/basefold.zig").Basefold(F);
const Transcript = @import("../transcript.zig").Transcript;

pub const ProverError = error{
    ConstraintViolation,
    TraceTooShort,
    Overflow,
} || std.mem.Allocator.Error || stack.Error;

pub const VMProof = struct {
    /// Output value (top of stack after execution) - public
    output: F,

    /// Number of variables in constraint polynomial (log2 of padded trace size)
    num_vars: usize,

    /// Basefold proof that constraint polynomial sums to 0
    basefold_proof: Basefold.Proof,

    pub fn deinit(self: *const VMProof, allocator: std.mem.Allocator) void {
        self.basefold_proof.deinit(allocator);
    }
};

pub const ProverConfig = struct {
    /// Number of FRI queries for Basefold
    num_queries: usize = 16,
};

/// Execute program and generate proof of correct execution
pub fn prove(
    program: stack.Program,
    allocator: std.mem.Allocator,
    config: ProverConfig,
) ProverError!VMProof {
    // 1. Execute and generate trace
    var exec_trace = try stack.executeAndTrace(program, allocator);
    defer exec_trace.deinit();

    if (exec_trace.rows.len == 0) {
        return error.TraceTooShort;
    }

    // 2. Convert trace to polynomial columns (padded to power of 2)
    const columns = try exec_trace.toPolynomials(allocator);
    defer trace_mod.Trace.freePolynomials(columns, allocator);

    // Create TracePolynomials view (doesn't own memory)
    const polys = constraints.TracePolynomials.fromColumns(columns, allocator);

    // 3. Build constraint polynomial
    const constraint_evals = try constraints.ConstraintEval.buildConstraintPoly(
        &polys,
        allocator,
    );
    defer allocator.free(constraint_evals);

    // 4. Verify constraints are all zero (sanity check)
    if (!constraints.ConstraintEval.verifyAllZero(constraint_evals)) {
        return error.ConstraintViolation;
    }

    // 5. Derive random point for zerocheck
    // We use a transcript seeded with domain separator
    // In production, this should include commitments to the trace
    const num_vars = std.math.log2_int(usize, constraint_evals.len);
    const r_point = try deriveRandomPoint(num_vars, allocator);
    defer allocator.free(r_point);

    // 6. Generate Basefold proof
    // We prove that Σ constraint(x) * eq(x, r) = 0
    // Since constraint(x) = 0 for all x, the sum is 0
    const basefold_config = Basefold.Config{ .num_queries = config.num_queries };
    const basefold_proof = try Basefold.prove(
        allocator,
        constraint_evals,
        r_point,
        basefold_config,
    );

    // 7. Extract output from final state
    const final_row = exec_trace.rows[exec_trace.rows.len - 1];
    const output = final_row.next_stack_top;

    return VMProof{
        .output = output,
        .num_vars = num_vars,
        .basefold_proof = basefold_proof,
    };
}

/// Derive a random evaluation point for zerocheck.
/// Uses Fiat-Shamir with a domain separator.
fn deriveRandomPoint(num_vars: usize, allocator: std.mem.Allocator) ![]F {
    var transcript = Transcript(F).init("vm-zerocheck");

    // Add domain separation for num_vars
    transcript.absorb(F.fromU32(@intCast(num_vars)));

    const r_point = try allocator.alloc(F, num_vars);
    for (r_point) |*r| {
        r.* = transcript.squeeze();
    }

    return r_point;
}

// ============ Tests ============ //

const testing = std.testing;

test "prover: simple add program" {
    const allocator = testing.allocator;

    const program = [_]stack.Instruction{
        .{ .PUSH = 10 },
        .{ .PUSH = 20 },
        .{ .ADD = {} },
    };

    var proof = try prove(&program, allocator, .{});
    defer proof.deinit(allocator);

    // Output should be 30
    try testing.expect(proof.output.eql(F.fromU32(30)));
}

test "prover: multiply program" {
    const allocator = testing.allocator;

    const program = [_]stack.Instruction{
        .{ .PUSH = 7 },
        .{ .PUSH = 6 },
        .{ .MUL = {} },
    };

    var proof = try prove(&program, allocator, .{});
    defer proof.deinit(allocator);

    // Output should be 42
    try testing.expect(proof.output.eql(F.fromU32(42)));
}

test "prover: fibonacci program" {
    const allocator = testing.allocator;

    // fib(7) = 13
    const program = [_]stack.Instruction{
        .{ .PUSH = 1 },
        .{ .PUSH = 1 },
        .{ .SWAP = {} },
        .{ .OVER = {} },
        .{ .ADD = {} },
        .{ .SWAP = {} },
        .{ .OVER = {} },
        .{ .ADD = {} },
        .{ .SWAP = {} },
        .{ .OVER = {} },
        .{ .ADD = {} },
        .{ .SWAP = {} },
        .{ .OVER = {} },
        .{ .ADD = {} },
        .{ .SWAP = {} },
        .{ .OVER = {} },
        .{ .ADD = {} },
    };

    var proof = try prove(&program, allocator, .{});
    defer proof.deinit(allocator);

    // fib(7) = 13
    try testing.expect(proof.output.eql(F.fromU32(13)));
}

test "prover: deriveRandomPoint is deterministic" {
    const allocator = testing.allocator;

    const r1 = try deriveRandomPoint(3, allocator);
    defer allocator.free(r1);

    const r2 = try deriveRandomPoint(3, allocator);
    defer allocator.free(r2);

    // Same num_vars should give same point
    for (r1, r2) |a, b| {
        try testing.expect(a.eql(b));
    }
}

test "prover: different num_vars gives different points" {
    const allocator = testing.allocator;

    const r3 = try deriveRandomPoint(3, allocator);
    defer allocator.free(r3);

    const r4 = try deriveRandomPoint(4, allocator);
    defer allocator.free(r4);

    // Different num_vars should give different first element
    try testing.expect(!r3[0].eql(r4[0]));
}
