const std = @import("std");
const Allocator = std.mem.Allocator;
const constraint = @import("constraint.zig");
const Cell = constraint.Cell;

/// Builder parameterized by field type.
/// Provides ergonomic API for constructing circuits with common gate patterns.
pub fn CircuitBuilder(comptime F: type) type {
    const CS = constraint.ConstraintSystem(F);
    const Trace = CS.Trace;
    const Term = CS.Term;
    const Constraint = CS.Constraint;
    const ConstraintSet = CS.ConstraintSet;

    return struct {
        const Self = @This();

        allocator: Allocator,
        arena: std.heap.ArenaAllocator,
        trace: Trace,
        constraints: ConstraintSet,

        pub fn init(allocator: Allocator, num_rows: usize) !Self {
            return .{
                .allocator = allocator,
                .arena = std.heap.ArenaAllocator.init(allocator),
                .trace = try Trace.init(allocator, num_rows),
                .constraints = ConstraintSet.init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            self.arena.deinit();
            self.constraints.deinit();
            self.trace.deinit();
        }

        // === Column creation ===

        pub fn addWitness(self: *Self) !usize {
            return self.trace.addColumn();
        }

        pub fn addPublic(self: *Self) !usize {
            const col = try self.trace.addColumn();
            try self.trace.markPublic(col);
            return col;
        }

        // === Common gate patterns ===

        /// a + b = c
        pub fn addGate(self: *Self, a: usize, b: usize, c: usize) !void {
            try self.validateColumns(&.{ a, b, c });
            const arena_alloc = self.arena.allocator();
            const terms = try arena_alloc.alloc(Term, 3);
            terms[0] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{.{ .col = a }}) } };
            terms[1] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{.{ .col = b }}) } };
            terms[2] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{.{ .col = c }}) } };
            try self.constraints.add(terms);
        }

        /// a - b = c
        pub fn subGate(self: *Self, a: usize, b: usize, c: usize) !void {
            try self.validateColumns(&.{ a, b, c });
            const arena_alloc = self.arena.allocator();
            const terms = try arena_alloc.alloc(Term, 3);
            terms[0] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{.{ .col = a }}) } };
            terms[1] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{.{ .col = b }}) } };
            terms[2] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{.{ .col = c }}) } };
            try self.constraints.add(terms);
        }

        /// a * b = c
        pub fn mulGate(self: *Self, a: usize, b: usize, c: usize) !void {
            try self.validateColumns(&.{ a, b, c });
            const arena_alloc = self.arena.allocator();
            const terms = try arena_alloc.alloc(Term, 2);
            terms[0] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{ .{ .col = a }, .{ .col = b } }) } };
            terms[1] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{.{ .col = c }}) } };
            try self.constraints.add(terms);
        }

        /// a = constant
        pub fn constGate(self: *Self, a: usize, constant: F) !void {
            try self.validateColumns(&.{a});
            const arena_alloc = self.arena.allocator();
            const terms = try arena_alloc.alloc(Term, 2);
            terms[0] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{.{ .col = a }}) } };
            terms[1] = .{ .constant = constant.neg() };
            try self.constraints.add(terms);
        }

        /// a = 0
        pub fn assertZero(self: *Self, a: usize) !void {
            try self.validateColumns(&.{a});
            const arena_alloc = self.arena.allocator();
            const terms = try arena_alloc.alloc(Term, 1);
            terms[0] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{.{ .col = a }}) } };
            try self.constraints.add(terms);
        }

        /// selector * (a * b - c) = 0  (conditional multiplication)
        pub fn conditionalMul(self: *Self, sel: usize, a: usize, b: usize, c: usize) !void {
            try self.validateColumns(&.{ sel, a, b, c });
            const arena_alloc = self.arena.allocator();
            const terms = try arena_alloc.alloc(Term, 2);
            terms[0] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{ .{ .col = sel }, .{ .col = a }, .{ .col = b } }) } };
            terms[1] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{ .{ .col = sel }, .{ .col = c } }) } };
            try self.constraints.add(terms);
        }

        /// col[row+1] = col[row] + delta[row]  (state transition)
        pub fn transition(self: *Self, col: usize, delta: usize) !void {
            try self.validateColumns(&.{ col, delta });
            const arena_alloc = self.arena.allocator();
            const terms = try arena_alloc.alloc(Term, 3);
            terms[0] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{.{ .col = col, .rot = 1 }}) } };
            terms[1] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{.{ .col = col, .rot = 0 }}) } };
            terms[2] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{.{ .col = delta, .rot = 0 }}) } };
            try self.constraints.add(terms);
        }

        /// selector * (col[row+1] - col[row] - delta[row]) = 0  (conditional transition)
        pub fn conditionalTransition(self: *Self, sel: usize, col: usize, delta: usize) !void {
            try self.validateColumns(&.{ sel, col, delta });
            const arena_alloc = self.arena.allocator();
            const terms = try arena_alloc.alloc(Term, 3);
            terms[0] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{ .{ .col = sel }, .{ .col = col, .rot = 1 } }) } };
            terms[1] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{ .{ .col = sel }, .{ .col = col, .rot = 0 } }) } };
            terms[2] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{ .{ .col = sel }, .{ .col = delta, .rot = 0 } }) } };
            try self.constraints.add(terms);
        }

        // === Custom constraints ===

        /// Add arbitrary constraint (caller responsible for column validation)
        pub fn addConstraint(self: *Self, terms: []const Term) !void {
            try self.constraints.add(terms);
        }

        // === Cell access ===

        pub fn set(self: *Self, col: usize, row: usize, val: F) !void {
            try self.trace.set(col, row, val);
        }

        // === Build ===

        pub fn build(self: *Self) struct { trace: *Trace, constraints: []const Constraint } {
            return .{
                .trace = &self.trace,
                .constraints = self.constraints.constraints.items,
            };
        }

        // === Internal helpers ===

        fn validateColumns(self: *Self, cols: []const usize) !void {
            const num_cols = self.trace.numColumns();
            for (cols) |col| {
                if (col >= num_cols) return error.InvalidColumn;
            }
        }

        fn cellSlice(self: *Self, cells: []const Cell) ![]const Cell {
            return self.arena.allocator().dupe(Cell, cells);
        }
    };
}

// ============ Tests ============ //

const testing = std.testing;
const M31 = @import("../fields/mersenne31.zig").Mersenne31;
const CSTester = constraint.ConstraintSystem(M31);
const Builder = CircuitBuilder(M31);

test "CircuitBuilder: init and deinit" {
    var builder = try Builder.init(testing.allocator, 4);
    defer builder.deinit();

    try testing.expectEqual(@as(usize, 0), builder.trace.numColumns());
}

test "CircuitBuilder: addWitness creates column" {
    var builder = try Builder.init(testing.allocator, 4);
    defer builder.deinit();

    const col0 = try builder.addWitness();
    try testing.expectEqual(@as(usize, 0), col0);
    try testing.expectEqual(@as(usize, 1), builder.trace.numColumns());

    const col1 = try builder.addWitness();
    try testing.expectEqual(@as(usize, 1), col1);
}

test "CircuitBuilder: addPublic creates and marks column" {
    var builder = try Builder.init(testing.allocator, 4);
    defer builder.deinit();

    const pub_col = try builder.addPublic();
    try testing.expectEqual(@as(usize, 0), pub_col);
    try testing.expectEqual(@as(usize, 1), builder.trace.public_columns.items.len);
}

test "CircuitBuilder: mulGate - satisfied" {
    var builder = try Builder.init(testing.allocator, 4);
    defer builder.deinit();

    const a = try builder.addWitness();
    const b = try builder.addWitness();
    const c = try builder.addWitness();

    try builder.mulGate(a, b, c);

    // 3 * 7 = 21
    try builder.set(a, 0, M31.fromU64(3));
    try builder.set(b, 0, M31.fromU64(7));
    try builder.set(c, 0, M31.fromU64(21));

    // Pad remaining rows
    for (1..4) |row| {
        try builder.set(a, row, M31.zero);
        try builder.set(b, row, M31.zero);
        try builder.set(c, row, M31.zero);
    }

    const result = builder.build();
    const evals = try CSTester.evaluate(result.constraints, result.trace, testing.allocator);
    defer testing.allocator.free(evals);

    try testing.expect(CSTester.isSatisfied(evals) == null);
}

test "CircuitBuilder: addGate - satisfied" {
    var builder = try Builder.init(testing.allocator, 4);
    defer builder.deinit();

    const a = try builder.addWitness();
    const b = try builder.addWitness();
    const c = try builder.addWitness();

    try builder.addGate(a, b, c);

    // 5 + 3 = 8
    try builder.set(a, 0, M31.fromU64(5));
    try builder.set(b, 0, M31.fromU64(3));
    try builder.set(c, 0, M31.fromU64(8));

    for (1..4) |row| {
        try builder.set(a, row, M31.zero);
        try builder.set(b, row, M31.zero);
        try builder.set(c, row, M31.zero);
    }

    const result = builder.build();
    const evals = try CSTester.evaluate(result.constraints, result.trace, testing.allocator);
    defer testing.allocator.free(evals);

    try testing.expect(CSTester.isSatisfied(evals) == null);
}

test "CircuitBuilder: subGate - satisfied" {
    var builder = try Builder.init(testing.allocator, 4);
    defer builder.deinit();

    const a = try builder.addWitness();
    const b = try builder.addWitness();
    const c = try builder.addWitness();

    try builder.subGate(a, b, c);

    // 10 - 3 = 7
    try builder.set(a, 0, M31.fromU64(10));
    try builder.set(b, 0, M31.fromU64(3));
    try builder.set(c, 0, M31.fromU64(7));

    for (1..4) |row| {
        try builder.set(a, row, M31.zero);
        try builder.set(b, row, M31.zero);
        try builder.set(c, row, M31.zero);
    }

    const result = builder.build();
    const evals = try CSTester.evaluate(result.constraints, result.trace, testing.allocator);
    defer testing.allocator.free(evals);

    try testing.expect(CSTester.isSatisfied(evals) == null);
}

test "CircuitBuilder: constGate - satisfied" {
    var builder = try Builder.init(testing.allocator, 4);
    defer builder.deinit();

    const a = try builder.addWitness();

    try builder.constGate(a, M31.fromU64(42));

    // All rows must equal 42
    for (0..4) |row| {
        try builder.set(a, row, M31.fromU64(42));
    }

    const result = builder.build();
    const evals = try CSTester.evaluate(result.constraints, result.trace, testing.allocator);
    defer testing.allocator.free(evals);

    try testing.expect(CSTester.isSatisfied(evals) == null);
}

test "CircuitBuilder: assertZero - satisfied" {
    var builder = try Builder.init(testing.allocator, 4);
    defer builder.deinit();

    const a = try builder.addWitness();

    try builder.assertZero(a);

    // All rows must be zero
    for (0..4) |row| {
        try builder.set(a, row, M31.zero);
    }

    const result = builder.build();
    const evals = try CSTester.evaluate(result.constraints, result.trace, testing.allocator);
    defer testing.allocator.free(evals);

    try testing.expect(CSTester.isSatisfied(evals) == null);
}

test "CircuitBuilder: assertZero - unsatisfied" {
    var builder = try Builder.init(testing.allocator, 4);
    defer builder.deinit();

    const a = try builder.addWitness();

    try builder.assertZero(a);

    // Row 0 is not zero
    try builder.set(a, 0, M31.fromU64(5));
    for (1..4) |row| {
        try builder.set(a, row, M31.zero);
    }

    const result = builder.build();
    const evals = try CSTester.evaluate(result.constraints, result.trace, testing.allocator);
    defer testing.allocator.free(evals);

    try testing.expectEqual(@as(?usize, 0), CSTester.isSatisfied(evals));
}

test "CircuitBuilder: chained arithmetic (mul then add)" {
    var builder = try Builder.init(testing.allocator, 4);
    defer builder.deinit();

    const a = try builder.addWitness();
    const b = try builder.addWitness();
    const t = try builder.addWitness(); // intermediate: a * b
    const c = try builder.addWitness();
    const out = try builder.addPublic();

    // t = a * b
    try builder.mulGate(a, b, t);
    // out = t + c
    try builder.addGate(t, c, out);

    // Witness: (3 * 7) + 5 = 26
    try builder.set(a, 0, M31.fromU64(3));
    try builder.set(b, 0, M31.fromU64(7));
    try builder.set(t, 0, M31.fromU64(21));
    try builder.set(c, 0, M31.fromU64(5));
    try builder.set(out, 0, M31.fromU64(26));

    // Pad remaining rows
    for (1..4) |row| {
        try builder.set(a, row, M31.zero);
        try builder.set(b, row, M31.zero);
        try builder.set(t, row, M31.zero);
        try builder.set(c, row, M31.zero);
        try builder.set(out, row, M31.zero);
    }

    const result = builder.build();
    const evals = try CSTester.evaluate(result.constraints, result.trace, testing.allocator);
    defer testing.allocator.free(evals);

    try testing.expect(CSTester.isSatisfied(evals) == null);
}

test "CircuitBuilder: transition gate" {
    var builder = try Builder.init(testing.allocator, 4);
    defer builder.deinit();

    const state = try builder.addWitness();
    const delta = try builder.addWitness();

    // state[row+1] = state[row] + delta[row]
    try builder.transition(state, delta);

    // state: 0 -> 5 -> 8 -> 15
    // delta: 5, 3, 7, X (last delta unused)
    try builder.set(state, 0, M31.fromU64(0));
    try builder.set(state, 1, M31.fromU64(5));
    try builder.set(state, 2, M31.fromU64(8));
    try builder.set(state, 3, M31.fromU64(15));

    try builder.set(delta, 0, M31.fromU64(5));
    try builder.set(delta, 1, M31.fromU64(3));
    try builder.set(delta, 2, M31.fromU64(7));
    try builder.set(delta, 3, M31.zero); // unused

    const result = builder.build();
    const evals = try CSTester.evaluate(result.constraints, result.trace, testing.allocator);
    defer testing.allocator.free(evals);

    try testing.expect(CSTester.isSatisfied(evals) == null);
}

test "CircuitBuilder: conditionalMul - selector enabled" {
    var builder = try Builder.init(testing.allocator, 4);
    defer builder.deinit();

    const sel = try builder.addWitness();
    const a = try builder.addWitness();
    const b = try builder.addWitness();
    const c = try builder.addWitness();

    try builder.conditionalMul(sel, a, b, c);

    // Row 0: sel=1, so a*b must equal c
    try builder.set(sel, 0, M31.one);
    try builder.set(a, 0, M31.fromU64(4));
    try builder.set(b, 0, M31.fromU64(5));
    try builder.set(c, 0, M31.fromU64(20)); // 4*5=20

    for (1..4) |row| {
        try builder.set(sel, row, M31.zero);
        try builder.set(a, row, M31.zero);
        try builder.set(b, row, M31.zero);
        try builder.set(c, row, M31.zero);
    }

    const result = builder.build();
    const evals = try CSTester.evaluate(result.constraints, result.trace, testing.allocator);
    defer testing.allocator.free(evals);

    try testing.expect(CSTester.isSatisfied(evals) == null);
}

test "CircuitBuilder: conditionalMul - selector disabled" {
    var builder = try Builder.init(testing.allocator, 4);
    defer builder.deinit();

    const sel = try builder.addWitness();
    const a = try builder.addWitness();
    const b = try builder.addWitness();
    const c = try builder.addWitness();

    try builder.conditionalMul(sel, a, b, c);

    // Row 0: sel=0, so constraint is 0 regardless of a, b, c
    try builder.set(sel, 0, M31.zero);
    try builder.set(a, 0, M31.fromU64(999));
    try builder.set(b, 0, M31.fromU64(888));
    try builder.set(c, 0, M31.fromU64(777)); // Wrong, but selector is off

    for (1..4) |row| {
        try builder.set(sel, row, M31.zero);
        try builder.set(a, row, M31.zero);
        try builder.set(b, row, M31.zero);
        try builder.set(c, row, M31.zero);
    }

    const result = builder.build();
    const evals = try CSTester.evaluate(result.constraints, result.trace, testing.allocator);
    defer testing.allocator.free(evals);

    try testing.expect(CSTester.isSatisfied(evals) == null);
}

test "CircuitBuilder: conditionalTransition" {
    var builder = try Builder.init(testing.allocator, 4);
    defer builder.deinit();

    const sel = try builder.addWitness();
    const state = try builder.addWitness();
    const delta = try builder.addWitness();

    try builder.conditionalTransition(sel, state, delta);

    // Row 0: sel=1, state transitions from 10 to 15 with delta=5
    // Row 1: sel=0, state can be anything (15 -> 100, doesn't matter)
    // Row 2: sel=1, state transitions from 100 to 110 with delta=10
    try builder.set(sel, 0, M31.one);
    try builder.set(sel, 1, M31.zero);
    try builder.set(sel, 2, M31.one);
    try builder.set(sel, 3, M31.zero); // unused for transition

    try builder.set(state, 0, M31.fromU64(10));
    try builder.set(state, 1, M31.fromU64(15));
    try builder.set(state, 2, M31.fromU64(100));
    try builder.set(state, 3, M31.fromU64(110));

    try builder.set(delta, 0, M31.fromU64(5));
    try builder.set(delta, 1, M31.fromU64(999)); // ignored
    try builder.set(delta, 2, M31.fromU64(10));
    try builder.set(delta, 3, M31.zero);

    const result = builder.build();
    const evals = try CSTester.evaluate(result.constraints, result.trace, testing.allocator);
    defer testing.allocator.free(evals);

    try testing.expect(CSTester.isSatisfied(evals) == null);
}

test "CircuitBuilder: addConstraint for custom constraint" {
    var builder = try Builder.init(testing.allocator, 4);
    defer builder.deinit();

    const a = try builder.addWitness();
    const b = try builder.addWitness();

    // Custom: a^2 - b = 0 (a squared equals b)
    // Use the builder's arena allocator so memory is cleaned up with the builder
    const arena_alloc = builder.arena.allocator();
    const terms = try arena_alloc.alloc(CSTester.Term, 2);
    terms[0] = .{ .product = .{ .coeff = M31.one, .cells = try arena_alloc.dupe(Cell, &.{ .{ .col = a }, .{ .col = a } }) } };
    terms[1] = .{ .product = .{ .coeff = M31.one.neg(), .cells = try arena_alloc.dupe(Cell, &.{.{ .col = b }}) } };
    try builder.addConstraint(terms);

    // 5^2 = 25
    try builder.set(a, 0, M31.fromU64(5));
    try builder.set(b, 0, M31.fromU64(25));

    for (1..4) |row| {
        try builder.set(a, row, M31.zero);
        try builder.set(b, row, M31.zero);
    }

    const result = builder.build();
    const evals = try CSTester.evaluate(result.constraints, result.trace, testing.allocator);
    defer testing.allocator.free(evals);

    try testing.expect(CSTester.isSatisfied(evals) == null);
}
