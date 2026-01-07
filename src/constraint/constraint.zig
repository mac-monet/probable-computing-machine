const std = @import("std");

/// A cell reference: column at row offset (field-independent)
pub const Cell = struct {
    col: usize,
    rot: i32 = 0, // 0 = current row, 1 = next, -1 = prev
};

/// Constraint system parameterized by field type.
/// Usage:
///   const F = @import("../fields/mersenne31.zig").Mersenne31;
///   const CS = ConstraintSystem(F);
///   const trace = CS.Trace;
///   const term = CS.Term;
pub fn ConstraintSystem(comptime F: type) type {
    return struct {
        pub const Field = F;

        // Forward declaration for Trace (defined in trace.zig)
        pub const Trace = @import("trace.zig").Trace(F);

        /// A term: constant or coefficient × product of cells
        pub const Term = union(enum) {
            constant: F,
            product: struct {
                coeff: F,
                cells: []const Cell,
            },

            /// Evaluate this term against a trace at the given row.
            /// For constants, returns the constant value.
            /// For products, returns coeff * product(trace[cell] for cell in cells).
            pub fn evaluate(self: Term, trace: *const Trace, row: usize) !F {
                return switch (self) {
                    .constant => |c| c,
                    .product => |p| {
                        var result = p.coeff;
                        for (p.cells) |cell| {
                            const val = try trace.get(cell.col, row, cell.rot);
                            result = result.mul(val);
                        }
                        return result;
                    },
                };
            }
        };

        /// A constraint: sum of terms that must equal zero
        pub const Constraint = struct {
            terms: []const Term,

            /// Compute the valid row range for evaluating this constraint.
            /// Returns the range [start, end) of rows where all rotations are in bounds.
            /// Returns error.InvalidRowRange if the constraint cannot be evaluated on any rows.
            pub fn validRowRange(self: Constraint, num_rows: usize) !struct { start: usize, end: usize } {
                var min_rot: i32 = 0;
                var max_rot: i32 = 0;

                for (self.terms) |term| {
                    switch (term) {
                        .constant => {},
                        .product => |p| {
                            for (p.cells) |cell| {
                                min_rot = @min(min_rot, cell.rot);
                                max_rot = @max(max_rot, cell.rot);
                            }
                        },
                    }
                }

                const start: usize = if (min_rot < 0) @intCast(-min_rot) else 0;
                const end_offset: usize = @intCast(max_rot);

                if (start >= num_rows or end_offset >= num_rows) {
                    return error.InvalidRowRange;
                }

                const end: usize = num_rows - end_offset;
                if (start >= end) {
                    return error.InvalidRowRange;
                }

                return .{ .start = start, .end = end };
            }

            /// Evaluate this constraint at a specific row.
            /// Returns the sum of all terms evaluated at the given row.
            /// For a satisfied constraint, this should return zero.
            pub fn evaluate(self: Constraint, trace: *const Trace, row: usize) !F {
                var sum = F.zero;
                for (self.terms) |term| {
                    const val = try term.evaluate(trace, row);
                    sum = sum.add(val);
                }
                return sum;
            }
        };

        /// Collection of constraints (simple storage).
        /// Validation of column references should be done by the caller (e.g., CircuitBuilder).
        pub const ConstraintSet = struct {
            constraints: std.ArrayListUnmanaged(Constraint),
            allocator: std.mem.Allocator,

            pub fn init(allocator: std.mem.Allocator) ConstraintSet {
                return .{
                    .constraints = .{},
                    .allocator = allocator,
                };
            }

            pub fn deinit(self: *ConstraintSet) void {
                self.constraints.deinit(self.allocator);
            }

            /// Add a constraint to the set.
            pub fn add(self: *ConstraintSet, terms: []const Term) !void {
                try self.constraints.append(self.allocator, .{ .terms = terms });
            }

            /// Returns the number of constraints in the set.
            pub fn len(self: ConstraintSet) usize {
                return self.constraints.items.len;
            }
        };

        /// Evaluate constraints over trace, returns polynomial evaluations.
        /// Each constraint is only evaluated on rows where all rotations are valid.
        /// Result[row] = sum of all constraint evaluations at that row.
        /// For valid witness: all values should be zero.
        pub fn evaluate(
            constraints: []const Constraint,
            trace: *const Trace,
            allocator: std.mem.Allocator,
        ) ![]F {
            const num_rows = trace.num_rows;
            var result = try allocator.alloc(F, num_rows);
            @memset(result, F.zero);

            for (constraints) |constraint| {
                // Only evaluate on valid rows for this constraint
                const range = try constraint.validRowRange(num_rows);

                for (range.start..range.end) |row| {
                    const val = try constraint.evaluate(trace, row);
                    result[row] = result[row].add(val);
                }
            }

            return result;
        }

        /// Evaluate constraints with per-constraint granularity (debug mode).
        /// Returns [constraint_idx][row] for debugging which constraint failed where.
        pub fn evaluateDebug(
            constraints: []const Constraint,
            trace: *const Trace,
            allocator: std.mem.Allocator,
        ) ![][]F {
            const num_rows = trace.num_rows;
            var result = try allocator.alloc([]F, constraints.len);
            errdefer {
                for (result[0..]) |maybe_row| {
                    if (maybe_row.len > 0) allocator.free(maybe_row);
                }
                allocator.free(result);
            }

            for (constraints, 0..) |constraint, ci| {
                result[ci] = try allocator.alloc(F, num_rows);
                @memset(result[ci], F.zero);

                const range = try constraint.validRowRange(num_rows);
                for (range.start..range.end) |row| {
                    result[ci][row] = try constraint.evaluate(trace, row);
                }
            }

            return result;
        }

        /// Check if all constraint evaluations are zero.
        /// Returns null if satisfied, or the first non-zero row index.
        pub fn isSatisfied(evals: []const F) ?usize {
            for (evals, 0..) |v, i| {
                if (!v.isZero()) return i;
            }
            return null;
        }
    };
}

// ============ Tests ============ //

const testing = std.testing;
const M31 = @import("../fields/mersenne31.zig").Mersenne31;
const CS = ConstraintSystem(M31);

test "Cell: default rotation is zero" {
    const cell = Cell{ .col = 5 };
    try testing.expectEqual(@as(usize, 5), cell.col);
    try testing.expectEqual(@as(i32, 0), cell.rot);
}

test "Cell: explicit rotation" {
    const next = Cell{ .col = 0, .rot = 1 };
    const prev = Cell{ .col = 0, .rot = -1 };

    try testing.expectEqual(@as(i32, 1), next.rot);
    try testing.expectEqual(@as(i32, -1), prev.rot);
}

test "Term: constant evaluation" {
    var trace = try CS.Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const term = CS.Term{ .constant = M31.fromU64(42) };
    const result = try term.evaluate(&trace, 0);

    try testing.expect(result.eql(M31.fromU64(42)));
}

test "Term: product with single cell" {
    var trace = try CS.Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const col = try trace.addColumn();
    try trace.set(col, 0, M31.fromU64(7));

    const cells = &[_]Cell{.{ .col = col }};
    const term = CS.Term{ .product = .{ .coeff = M31.fromU64(3), .cells = cells } };
    const result = try term.evaluate(&trace, 0);

    // 3 * 7 = 21
    try testing.expect(result.eql(M31.fromU64(21)));
}

test "Term: product with multiple cells" {
    var trace = try CS.Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const a = try trace.addColumn();
    const b = try trace.addColumn();
    try trace.set(a, 0, M31.fromU64(5));
    try trace.set(b, 0, M31.fromU64(7));

    const cells = &[_]Cell{ .{ .col = a }, .{ .col = b } };
    const term = CS.Term{ .product = .{ .coeff = M31.one, .cells = cells } };
    const result = try term.evaluate(&trace, 0);

    // 1 * 5 * 7 = 35
    try testing.expect(result.eql(M31.fromU64(35)));
}

test "Term: product with rotation" {
    var trace = try CS.Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const col = try trace.addColumn();
    try trace.set(col, 0, M31.fromU64(10));
    try trace.set(col, 1, M31.fromU64(20));

    // Access col[row+1] from row 0
    const cells = &[_]Cell{.{ .col = col, .rot = 1 }};
    const term = CS.Term{ .product = .{ .coeff = M31.one, .cells = cells } };
    const result = try term.evaluate(&trace, 0);

    try testing.expect(result.eql(M31.fromU64(20)));
}

test "Term: product with empty cells is just coefficient" {
    var trace = try CS.Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const cells: []const Cell = &.{};
    const term = CS.Term{ .product = .{ .coeff = M31.fromU64(99), .cells = cells } };
    const result = try term.evaluate(&trace, 0);

    try testing.expect(result.eql(M31.fromU64(99)));
}

test "Term: evaluate returns error on invalid column" {
    var trace = try CS.Trace.init(testing.allocator, 4);
    defer trace.deinit();

    // No columns added, try to access column 0
    const cells = &[_]Cell{.{ .col = 0 }};
    const term = CS.Term{ .product = .{ .coeff = M31.one, .cells = cells } };

    try testing.expectError(error.InvalidColumn, term.evaluate(&trace, 0));
}

test "Term: evaluate returns error on rotation out of bounds" {
    var trace = try CS.Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const col = try trace.addColumn();
    try trace.set(col, 3, M31.one);

    // Try to access row 4 from row 3 with rot=1
    const cells = &[_]Cell{.{ .col = col, .rot = 1 }};
    const term = CS.Term{ .product = .{ .coeff = M31.one, .cells = cells } };

    try testing.expectError(error.RotationOutOfBounds, term.evaluate(&trace, 3));
}

test "Term: evaluate returns error on unset cell" {
    var trace = try CS.Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const col = try trace.addColumn();
    // Don't set any values

    const cells = &[_]Cell{.{ .col = col }};
    const term = CS.Term{ .product = .{ .coeff = M31.one, .cells = cells } };

    try testing.expectError(error.CellNotSet, term.evaluate(&trace, 0));
}

// ============ Constraint Tests ============ //

test "Constraint: validRowRange with no rotations" {
    // Constraint with rot=0 should be valid on all rows
    const terms = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = 0, .rot = 0 }} } },
    };
    const constraint = CS.Constraint{ .terms = terms };

    const range = try constraint.validRowRange(4);
    try testing.expectEqual(@as(usize, 0), range.start);
    try testing.expectEqual(@as(usize, 4), range.end);
}

test "Constraint: validRowRange with positive rotation" {
    // Constraint with rot=1 should be valid on rows [0, num_rows-1)
    const terms = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = 0, .rot = 1 }} } },
    };
    const constraint = CS.Constraint{ .terms = terms };

    const range = try constraint.validRowRange(4);
    try testing.expectEqual(@as(usize, 0), range.start);
    try testing.expectEqual(@as(usize, 3), range.end);
}

test "Constraint: validRowRange with negative rotation" {
    // Constraint with rot=-1 should be valid on rows [1, num_rows)
    const terms = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = 0, .rot = -1 }} } },
    };
    const constraint = CS.Constraint{ .terms = terms };

    const range = try constraint.validRowRange(4);
    try testing.expectEqual(@as(usize, 1), range.start);
    try testing.expectEqual(@as(usize, 4), range.end);
}

test "Constraint: validRowRange with mixed rotations" {
    // rot=-1 and rot=2 should give valid range [1, num_rows-2)
    const terms = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = 0, .rot = -1 }} } },
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = 0, .rot = 2 }} } },
    };
    const constraint = CS.Constraint{ .terms = terms };

    const range = try constraint.validRowRange(8);
    try testing.expectEqual(@as(usize, 1), range.start);
    try testing.expectEqual(@as(usize, 6), range.end);
}

test "Constraint: validRowRange with constants ignored" {
    // Constants don't affect the row range
    const terms = &[_]CS.Term{
        .{ .constant = M31.fromU64(42) },
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = 0, .rot = 1 }} } },
    };
    const constraint = CS.Constraint{ .terms = terms };

    const range = try constraint.validRowRange(4);
    try testing.expectEqual(@as(usize, 0), range.start);
    try testing.expectEqual(@as(usize, 3), range.end);
}

test "Constraint: validRowRange error on too few rows" {
    // With rot=2, need at least 3 rows (rows 0,1 can evaluate)
    const terms = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = 0, .rot = 2 }} } },
    };
    const constraint = CS.Constraint{ .terms = terms };

    try testing.expectError(error.InvalidRowRange, constraint.validRowRange(2));
}

test "Constraint: validRowRange error when start >= end" {
    // With rot=-2 and rot=2 on 4 rows: start=2, end=2, invalid
    const terms = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = 0, .rot = -2 }} } },
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = 0, .rot = 2 }} } },
    };
    const constraint = CS.Constraint{ .terms = terms };

    try testing.expectError(error.InvalidRowRange, constraint.validRowRange(4));
}

test "Constraint: evaluate multiplication gate" {
    var trace = try CS.Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const a = try trace.addColumn();
    const b = try trace.addColumn();
    const c = try trace.addColumn();

    // 3 * 7 = 21
    try trace.set(a, 0, M31.fromU64(3));
    try trace.set(b, 0, M31.fromU64(7));
    try trace.set(c, 0, M31.fromU64(21));

    // Constraint: a * b - c = 0
    const terms = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{ .{ .col = a }, .{ .col = b } } } },
        .{ .product = .{ .coeff = M31.one.neg(), .cells = &.{.{ .col = c }} } },
    };
    const constraint = CS.Constraint{ .terms = terms };

    const result = try constraint.evaluate(&trace, 0);
    try testing.expect(result.isZero());
}

test "Constraint: evaluate invalid witness produces non-zero" {
    var trace = try CS.Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const a = try trace.addColumn();
    const b = try trace.addColumn();
    const c = try trace.addColumn();

    // 3 * 7 = 20 (WRONG - should be 21)
    try trace.set(a, 0, M31.fromU64(3));
    try trace.set(b, 0, M31.fromU64(7));
    try trace.set(c, 0, M31.fromU64(20));

    // Constraint: a * b - c = 0
    const terms = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{ .{ .col = a }, .{ .col = b } } } },
        .{ .product = .{ .coeff = M31.one.neg(), .cells = &.{.{ .col = c }} } },
    };
    const constraint = CS.Constraint{ .terms = terms };

    const result = try constraint.evaluate(&trace, 0);
    // 3 * 7 - 20 = 1
    try testing.expect(result.eql(M31.one));
}

test "Constraint: evaluate state transition with rotation" {
    var trace = try CS.Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const state = try trace.addColumn();

    // state[i+1] = state[i] + 1 (counting)
    try trace.set(state, 0, M31.fromU64(0));
    try trace.set(state, 1, M31.fromU64(1));
    try trace.set(state, 2, M31.fromU64(2));
    try trace.set(state, 3, M31.fromU64(3));

    // Constraint: state[row+1] - state[row] - 1 = 0
    const terms = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = state, .rot = 1 }} } },
        .{ .product = .{ .coeff = M31.one.neg(), .cells = &.{.{ .col = state, .rot = 0 }} } },
        .{ .constant = M31.one.neg() },
    };
    const constraint = CS.Constraint{ .terms = terms };

    // Valid range should be [0, 3)
    const range = try constraint.validRowRange(4);
    try testing.expectEqual(@as(usize, 0), range.start);
    try testing.expectEqual(@as(usize, 3), range.end);

    // Evaluate at each valid row
    for (range.start..range.end) |row| {
        const result = try constraint.evaluate(&trace, row);
        try testing.expect(result.isZero());
    }
}

test "Constraint: evaluate fibonacci via rotation" {
    var trace = try CS.Trace.init(testing.allocator, 8);
    defer trace.deinit();

    const fib = try trace.addColumn();

    // Fill: 1, 1, 2, 3, 5, 8, 13, 21
    const fibs = [_]u32{ 1, 1, 2, 3, 5, 8, 13, 21 };
    for (fibs, 0..) |v, i| {
        try trace.set(fib, i, M31.fromU64(v));
    }

    // Constraint: fib[row] + fib[row+1] - fib[row+2] = 0
    const terms = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = fib, .rot = 0 }} } },
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = fib, .rot = 1 }} } },
        .{ .product = .{ .coeff = M31.one.neg(), .cells = &.{.{ .col = fib, .rot = 2 }} } },
    };
    const constraint = CS.Constraint{ .terms = terms };

    // Valid range: rot=2 means rows [0, 6)
    const range = try constraint.validRowRange(8);
    try testing.expectEqual(@as(usize, 0), range.start);
    try testing.expectEqual(@as(usize, 6), range.end);

    // Evaluate at each valid row
    for (range.start..range.end) |row| {
        const result = try constraint.evaluate(&trace, row);
        try testing.expect(result.isZero());
    }
}

// ============ ConstraintSet Tests ============ //

test "ConstraintSet: init and deinit" {
    var cs = CS.ConstraintSet.init(testing.allocator);
    defer cs.deinit();

    try testing.expectEqual(@as(usize, 0), cs.len());
}

test "ConstraintSet: add constraint" {
    var cs = CS.ConstraintSet.init(testing.allocator);
    defer cs.deinit();

    // a + b = 0 constraint
    const terms = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = 0 }} } },
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = 1 }} } },
    };

    try cs.add(terms);
    try testing.expectEqual(@as(usize, 1), cs.len());
}

test "ConstraintSet: add multiple constraints" {
    var cs = CS.ConstraintSet.init(testing.allocator);
    defer cs.deinit();

    // First constraint: a + b = 0
    const terms1 = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = 0 }} } },
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = 1 }} } },
    };
    try cs.add(terms1);

    // Second constraint: b * c = 0
    const terms2 = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{ .{ .col = 1 }, .{ .col = 2 } } } },
    };
    try cs.add(terms2);

    try testing.expectEqual(@as(usize, 2), cs.len());
}

test "ConstraintSet: add constraint with constant term" {
    var cs = CS.ConstraintSet.init(testing.allocator);
    defer cs.deinit();

    // a - 42 = 0 constraint
    const terms = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = 0 }} } },
        .{ .constant = M31.fromU64(42).neg() },
    };

    try cs.add(terms);
    try testing.expectEqual(@as(usize, 1), cs.len());
}

// ============ Evaluation Function Tests ============ //

test "evaluate: multiplication constraint satisfied" {
    var trace = try CS.Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const a = try trace.addColumn();
    const b = try trace.addColumn();
    const c = try trace.addColumn();

    // 3 * 7 = 21
    try trace.set(a, 0, M31.fromU64(3));
    try trace.set(b, 0, M31.fromU64(7));
    try trace.set(c, 0, M31.fromU64(21));

    // Pad remaining rows
    for (1..4) |row| {
        try trace.set(a, row, M31.zero);
        try trace.set(b, row, M31.zero);
        try trace.set(c, row, M31.zero);
    }

    // Constraint: a * b - c = 0
    const terms = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{ .{ .col = a }, .{ .col = b } } } },
        .{ .product = .{ .coeff = M31.one.neg(), .cells = &.{.{ .col = c }} } },
    };
    const constraints = &[_]CS.Constraint{.{ .terms = terms }};

    const evals = try CS.evaluate(constraints, &trace, testing.allocator);
    defer testing.allocator.free(evals);

    try testing.expect(CS.isSatisfied(evals) == null);
}

test "evaluate: invalid witness produces non-zero" {
    var trace = try CS.Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const a = try trace.addColumn();
    const b = try trace.addColumn();
    const c = try trace.addColumn();

    // 3 * 7 = 20 (WRONG)
    try trace.set(a, 0, M31.fromU64(3));
    try trace.set(b, 0, M31.fromU64(7));
    try trace.set(c, 0, M31.fromU64(20));

    for (1..4) |row| {
        try trace.set(a, row, M31.zero);
        try trace.set(b, row, M31.zero);
        try trace.set(c, row, M31.zero);
    }

    const terms = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{ .{ .col = a }, .{ .col = b } } } },
        .{ .product = .{ .coeff = M31.one.neg(), .cells = &.{.{ .col = c }} } },
    };
    const constraints = &[_]CS.Constraint{.{ .terms = terms }};

    const evals = try CS.evaluate(constraints, &trace, testing.allocator);
    defer testing.allocator.free(evals);

    // Should fail at row 0
    try testing.expect(CS.isSatisfied(evals) == 0);
}

test "evaluate: state transition with rotation" {
    var trace = try CS.Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const state = try trace.addColumn();

    // state[i+1] = state[i] + 1 (counting)
    try trace.set(state, 0, M31.fromU64(0));
    try trace.set(state, 1, M31.fromU64(1));
    try trace.set(state, 2, M31.fromU64(2));
    try trace.set(state, 3, M31.fromU64(3));

    // Constraint: state[row+1] - state[row] - 1 = 0
    const terms = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = state, .rot = 1 }} } },
        .{ .product = .{ .coeff = M31.one.neg(), .cells = &.{.{ .col = state, .rot = 0 }} } },
        .{ .constant = M31.one.neg() },
    };
    const constraints = &[_]CS.Constraint{.{ .terms = terms }};

    const evals = try CS.evaluate(constraints, &trace, testing.allocator);
    defer testing.allocator.free(evals);

    try testing.expect(CS.isSatisfied(evals) == null);
}

test "evaluate: fibonacci via rotation" {
    var trace = try CS.Trace.init(testing.allocator, 8);
    defer trace.deinit();

    const fib = try trace.addColumn();

    // Fill: 1, 1, 2, 3, 5, 8, 13, 21
    const fibs = [_]u32{ 1, 1, 2, 3, 5, 8, 13, 21 };
    for (fibs, 0..) |v, i| {
        try trace.set(fib, i, M31.fromU64(v));
    }

    // Constraint: fib[row] + fib[row+1] - fib[row+2] = 0
    const terms = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = fib, .rot = 0 }} } },
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = fib, .rot = 1 }} } },
        .{ .product = .{ .coeff = M31.one.neg(), .cells = &.{.{ .col = fib, .rot = 2 }} } },
    };
    const constraints = &[_]CS.Constraint{.{ .terms = terms }};

    const evals = try CS.evaluate(constraints, &trace, testing.allocator);
    defer testing.allocator.free(evals);

    try testing.expect(CS.isSatisfied(evals) == null);
}

test "evaluate: multiple constraints combined" {
    var trace = try CS.Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const a = try trace.addColumn();
    const b = try trace.addColumn();
    const c = try trace.addColumn();
    const d = try trace.addColumn();

    // Set up: a + b = c AND a * b = d on all rows
    // Row 0: a=2, b=3, c=5, d=6
    try trace.set(a, 0, M31.fromU64(2));
    try trace.set(b, 0, M31.fromU64(3));
    try trace.set(c, 0, M31.fromU64(5));
    try trace.set(d, 0, M31.fromU64(6));

    // Pad remaining rows with zeros (0 + 0 = 0, 0 * 0 = 0)
    for (1..4) |row| {
        try trace.set(a, row, M31.zero);
        try trace.set(b, row, M31.zero);
        try trace.set(c, row, M31.zero);
        try trace.set(d, row, M31.zero);
    }

    // Constraint 1: a + b - c = 0
    const terms1 = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = a }} } },
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = b }} } },
        .{ .product = .{ .coeff = M31.one.neg(), .cells = &.{.{ .col = c }} } },
    };

    // Constraint 2: a * b - d = 0
    const terms2 = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{ .{ .col = a }, .{ .col = b } } } },
        .{ .product = .{ .coeff = M31.one.neg(), .cells = &.{.{ .col = d }} } },
    };

    const constraints = &[_]CS.Constraint{
        .{ .terms = terms1 },
        .{ .terms = terms2 },
    };

    const evals = try CS.evaluate(constraints, &trace, testing.allocator);
    defer testing.allocator.free(evals);

    try testing.expect(CS.isSatisfied(evals) == null);
}

test "evaluateDebug: per-constraint granularity" {
    var trace = try CS.Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const a = try trace.addColumn();
    const b = try trace.addColumn();

    // Row 0: a=1, b=1 (both constraints satisfied)
    // Row 1: a=2, b=3 (constraint 1 fails: a-b != 0)
    try trace.set(a, 0, M31.fromU64(1));
    try trace.set(b, 0, M31.fromU64(1));
    try trace.set(a, 1, M31.fromU64(2));
    try trace.set(b, 1, M31.fromU64(3));
    try trace.set(a, 2, M31.zero);
    try trace.set(b, 2, M31.zero);
    try trace.set(a, 3, M31.zero);
    try trace.set(b, 3, M31.zero);

    // Constraint 1: a - b = 0
    const terms1 = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{.{ .col = a }} } },
        .{ .product = .{ .coeff = M31.one.neg(), .cells = &.{.{ .col = b }} } },
    };

    // Constraint 2: a * b - a = 0 (satisfied when b=1)
    const terms2 = &[_]CS.Term{
        .{ .product = .{ .coeff = M31.one, .cells = &.{ .{ .col = a }, .{ .col = b } } } },
        .{ .product = .{ .coeff = M31.one.neg(), .cells = &.{.{ .col = a }} } },
    };

    const constraints = &[_]CS.Constraint{
        .{ .terms = terms1 },
        .{ .terms = terms2 },
    };

    const debug_evals = try CS.evaluateDebug(constraints, &trace, testing.allocator);
    defer {
        for (debug_evals) |row| {
            testing.allocator.free(row);
        }
        testing.allocator.free(debug_evals);
    }

    try testing.expectEqual(@as(usize, 2), debug_evals.len);
    try testing.expectEqual(@as(usize, 4), debug_evals[0].len);

    // Constraint 1, row 0: 1 - 1 = 0
    try testing.expect(debug_evals[0][0].isZero());
    // Constraint 1, row 1: 2 - 3 = -1 (non-zero)
    try testing.expect(!debug_evals[0][1].isZero());

    // Constraint 2, row 0: 1*1 - 1 = 0
    try testing.expect(debug_evals[1][0].isZero());
    // Constraint 2, row 1: 2*3 - 2 = 4 (non-zero)
    try testing.expect(!debug_evals[1][1].isZero());
}

test "isSatisfied: returns null for all zeros" {
    const evals = &[_]M31{ M31.zero, M31.zero, M31.zero, M31.zero };
    try testing.expect(CS.isSatisfied(evals) == null);
}

test "isSatisfied: returns first non-zero index" {
    const evals = &[_]M31{ M31.zero, M31.zero, M31.one, M31.zero };
    try testing.expectEqual(@as(?usize, 2), CS.isSatisfied(evals));
}

test "isSatisfied: returns 0 for first element non-zero" {
    const evals = &[_]M31{ M31.one, M31.zero, M31.zero };
    try testing.expectEqual(@as(?usize, 0), CS.isSatisfied(evals));
}

test "isSatisfied: empty slice returns null" {
    const evals: []const M31 = &.{};
    try testing.expect(CS.isSatisfied(evals) == null);
}
