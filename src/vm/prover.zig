const std = @import("std");
const F = @import("../fields/mersenne31.zig").Mersenne31;
const stack = @import("stack.zig");
const trace_mod = @import("trace.zig");
const constraints = @import("constraints.zig");
const Basefold = @import("../pcs/basefold.zig").Basefold(F);
const Transcript = @import("../core/transcript.zig").Transcript;

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
    const r_point = try allocator.alloc(F, num_vars);
    defer allocator.free(r_point);
    Transcript(F).derivePoint("vm-zerocheck", r_point);

    // 6. Generate Basefold proof (zerocheck)
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

test "prover: derivePoint is deterministic" {
    var r1: [3]F = undefined;
    Transcript(F).derivePoint("test", &r1);

    var r2: [3]F = undefined;
    Transcript(F).derivePoint("test", &r2);

    // Same domain and num_vars should give same point
    for (r1, r2) |a, b| {
        try testing.expect(a.eql(b));
    }
}

test "prover: different num_vars gives different points" {
    var r3: [3]F = undefined;
    Transcript(F).derivePoint("test", &r3);

    var r4: [4]F = undefined;
    Transcript(F).derivePoint("test", &r4);

    // Different num_vars should give different first element
    try testing.expect(!r3[0].eql(r4[0]));
}
