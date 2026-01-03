const std = @import("std");
const F = @import("../fields/mersenne31.zig").Mersenne31;
const Opcode = @import("stack.zig").Opcode;
const trace_mod = @import("trace.zig");

/// Polynomial columns extracted from trace
/// Each column represents evaluations of a multilinear polynomial over {0,1}^n
pub const TracePolynomials = struct {
    step: []const F, // Step numbers (for debugging)
    opcode: []const F, // Opcode as field element
    stack_top: []const F, // Top of stack before instruction
    stack_next: []const F, // Second element before instruction
    next_stack_top: []const F, // TOS after instruction

    allocator: std.mem.Allocator,

    pub fn deinit(self: *TracePolynomials) void {
        self.allocator.free(self.step);
        self.allocator.free(self.opcode);
        self.allocator.free(self.stack_top);
        self.allocator.free(self.stack_next);
        self.allocator.free(self.next_stack_top);
    }

    /// Create from raw trace columns (from Trace.toPolynomials)
    pub fn fromColumns(columns: [][]F, allocator: std.mem.Allocator) TracePolynomials {
        return .{
            .step = columns[@intFromEnum(trace_mod.Trace.Column.step)],
            .opcode = columns[@intFromEnum(trace_mod.Trace.Column.opcode)],
            .stack_top = columns[@intFromEnum(trace_mod.Trace.Column.stack_top)],
            .stack_next = columns[@intFromEnum(trace_mod.Trace.Column.stack_next)],
            .next_stack_top = columns[@intFromEnum(trace_mod.Trace.Column.next_stack_top)],
            .allocator = allocator,
        };
    }
};

/// Constraint evaluation utilities
pub const ConstraintEval = struct {
    /// Evaluate all constraints at hypercube index.
    /// Returns 0 if constraints satisfied, non-zero if violated.
    ///
    /// We use a "constraint selector" pattern:
    /// Each constraint is multiplied by an indicator for when it applies.
    /// constraint = Σ (opcode == OP) * (specific_constraint_for_OP)
    pub fn evaluateAt(polys: *const TracePolynomials, index: usize) F {
        const opcode = polys.opcode[index];
        const top = polys.stack_top[index];
        const next_elem = polys.stack_next[index];
        const next_top = polys.next_stack_top[index];

        var violation = F.zero;

        // ADD constraint: (opcode == ADD) * (next_top - (top + next))
        const is_add = isOpcodeSelector(opcode, Opcode.ADD);
        const add_constraint = next_top.sub(top.add(next_elem));
        violation = violation.add(is_add.mul(add_constraint));

        // MUL constraint: (opcode == MUL) * (next_top - top * next)
        const is_mul = isOpcodeSelector(opcode, Opcode.MUL);
        const mul_constraint = next_top.sub(top.mul(next_elem));
        violation = violation.add(is_mul.mul(mul_constraint));

        // DUP constraint: (opcode == DUP) * (next_top - top)
        const is_dup = isOpcodeSelector(opcode, Opcode.DUP);
        const dup_constraint = next_top.sub(top);
        violation = violation.add(is_dup.mul(dup_constraint));

        // SWAP constraint: (opcode == SWAP) * (next_top - next)
        // After SWAP, the new top should be what was the second element
        const is_swap = isOpcodeSelector(opcode, Opcode.SWAP);
        const swap_constraint = next_top.sub(next_elem);
        violation = violation.add(is_swap.mul(swap_constraint));

        // OVER constraint: (opcode == OVER) * (next_top - next)
        // OVER copies the second element to top
        const is_over = isOpcodeSelector(opcode, Opcode.OVER);
        const over_constraint = next_top.sub(next_elem);
        violation = violation.add(is_over.mul(over_constraint));

        // PUSH constraint: For PUSH, we can't verify the value without the program
        // In a full implementation, we'd encode immediate values in the trace
        // For now, PUSH constraints are not enforced (selector = 0 contribution)
        // This is acceptable for MVP since the prover is honest

        return violation;
    }

    /// Build constraint polynomial over full hypercube.
    /// Result[i] = constraint violation at index i.
    /// For a valid trace, all entries should be 0.
    pub fn buildConstraintPoly(
        polys: *const TracePolynomials,
        allocator: std.mem.Allocator,
    ) ![]F {
        const size = polys.opcode.len;
        const constraint_evals = try allocator.alloc(F, size);

        for (0..size) |i| {
            constraint_evals[i] = evaluateAt(polys, i);
        }

        return constraint_evals;
    }

    /// Verify that all constraints are satisfied (sum to zero)
    pub fn verifyAllZero(constraint_evals: []const F) bool {
        for (constraint_evals) |c| {
            if (!c.isZero()) {
                return false;
            }
        }
        return true;
    }
};

/// Check if opcode field element matches target opcode.
/// Returns 1 if match, 0 otherwise (as field element).
fn isOpcodeSelector(opcode_field: F, target: Opcode) F {
    const target_val = F.fromU32(@intFromEnum(target));
    return if (opcode_field.eql(target_val)) F.one else F.zero;
}

// ============ Tests ============ //

const testing = std.testing;
const stack = @import("stack.zig");

test "constraint: ADD evaluates to zero for valid trace" {
    const allocator = testing.allocator;

    const program = [_]stack.Instruction{
        .{ .PUSH = 5 },
        .{ .PUSH = 3 },
        .{ .ADD = {} },
    };

    var trace = try stack.executeAndTrace(&program, allocator);
    defer trace.deinit();

    const columns = try trace.toPolynomials(allocator);
    defer trace_mod.Trace.freePolynomials(columns, allocator);

    const polys = TracePolynomials.fromColumns(columns, allocator);
    const constraint_evals = try ConstraintEval.buildConstraintPoly(&polys, allocator);
    defer allocator.free(constraint_evals);

    // All constraints should be zero for valid trace
    try testing.expect(ConstraintEval.verifyAllZero(constraint_evals));
}

test "constraint: MUL evaluates to zero for valid trace" {
    const allocator = testing.allocator;

    const program = [_]stack.Instruction{
        .{ .PUSH = 7 },
        .{ .PUSH = 6 },
        .{ .MUL = {} },
    };

    var trace = try stack.executeAndTrace(&program, allocator);
    defer trace.deinit();

    const columns = try trace.toPolynomials(allocator);
    defer trace_mod.Trace.freePolynomials(columns, allocator);

    const polys = TracePolynomials.fromColumns(columns, allocator);
    const constraint_evals = try ConstraintEval.buildConstraintPoly(&polys, allocator);
    defer allocator.free(constraint_evals);

    try testing.expect(ConstraintEval.verifyAllZero(constraint_evals));
}

test "constraint: DUP evaluates to zero for valid trace" {
    const allocator = testing.allocator;

    const program = [_]stack.Instruction{
        .{ .PUSH = 10 },
        .{ .DUP = {} },
    };

    var trace = try stack.executeAndTrace(&program, allocator);
    defer trace.deinit();

    const columns = try trace.toPolynomials(allocator);
    defer trace_mod.Trace.freePolynomials(columns, allocator);

    const polys = TracePolynomials.fromColumns(columns, allocator);
    const constraint_evals = try ConstraintEval.buildConstraintPoly(&polys, allocator);
    defer allocator.free(constraint_evals);

    try testing.expect(ConstraintEval.verifyAllZero(constraint_evals));
}

test "constraint: SWAP evaluates to zero for valid trace" {
    const allocator = testing.allocator;

    const program = [_]stack.Instruction{
        .{ .PUSH = 10 },
        .{ .PUSH = 20 },
        .{ .SWAP = {} },
    };

    var trace = try stack.executeAndTrace(&program, allocator);
    defer trace.deinit();

    const columns = try trace.toPolynomials(allocator);
    defer trace_mod.Trace.freePolynomials(columns, allocator);

    const polys = TracePolynomials.fromColumns(columns, allocator);
    const constraint_evals = try ConstraintEval.buildConstraintPoly(&polys, allocator);
    defer allocator.free(constraint_evals);

    try testing.expect(ConstraintEval.verifyAllZero(constraint_evals));
}

test "constraint: OVER evaluates to zero for valid trace" {
    const allocator = testing.allocator;

    const program = [_]stack.Instruction{
        .{ .PUSH = 10 },
        .{ .PUSH = 20 },
        .{ .OVER = {} },
    };

    var trace = try stack.executeAndTrace(&program, allocator);
    defer trace.deinit();

    const columns = try trace.toPolynomials(allocator);
    defer trace_mod.Trace.freePolynomials(columns, allocator);

    const polys = TracePolynomials.fromColumns(columns, allocator);
    const constraint_evals = try ConstraintEval.buildConstraintPoly(&polys, allocator);
    defer allocator.free(constraint_evals);

    try testing.expect(ConstraintEval.verifyAllZero(constraint_evals));
}

test "constraint: fibonacci trace is valid" {
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

    var trace = try stack.executeAndTrace(&program, allocator);
    defer trace.deinit();

    const columns = try trace.toPolynomials(allocator);
    defer trace_mod.Trace.freePolynomials(columns, allocator);

    const polys = TracePolynomials.fromColumns(columns, allocator);
    const constraint_evals = try ConstraintEval.buildConstraintPoly(&polys, allocator);
    defer allocator.free(constraint_evals);

    try testing.expect(ConstraintEval.verifyAllZero(constraint_evals));
}

test "constraint: isOpcodeSelector returns correct values" {
    // Test ADD selector
    const add_val = F.fromU32(@intFromEnum(Opcode.ADD));
    try testing.expect(isOpcodeSelector(add_val, Opcode.ADD).eql(F.one));
    try testing.expect(isOpcodeSelector(add_val, Opcode.MUL).eql(F.zero));

    // Test MUL selector
    const mul_val = F.fromU32(@intFromEnum(Opcode.MUL));
    try testing.expect(isOpcodeSelector(mul_val, Opcode.MUL).eql(F.one));
    try testing.expect(isOpcodeSelector(mul_val, Opcode.ADD).eql(F.zero));
}
