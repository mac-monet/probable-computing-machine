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
