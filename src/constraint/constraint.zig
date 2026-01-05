//! Plonkish constraint types for the constraint system frontend.
//!
//! Core types:
//! - Cell: reference to a column at a row offset (rotation)
//! - Term: coefficient times product of cells
//! - Constraint: sum of terms that must equal zero
//!
//! Constraints compute their valid row range from rotations at definition time.
//! No wraparound - evaluator only runs constraints on valid rows.

const std = @import("std");
const Mersenne31 = @import("../fields/mersenne31.zig").Mersenne31;

/// Field type used throughout the constraint system.
pub const F = Mersenne31;

/// A cell reference: column at row offset.
///
/// The rotation (rot) specifies the row offset from the current row:
/// - rot = 0: current row
/// - rot = 1: next row
/// - rot = -1: previous row
/// - rot = 2: two rows ahead, etc.
pub const Cell = struct {
    col: usize,
    rot: i8 = 0,
};

/// A term: coefficient times product of cells.
///
/// When evaluated at a row, computes: coeff * prod(trace[cell.col, row + cell.rot])
/// An empty cells slice represents the constant 1 (so coeff alone is the value).
pub const Term = struct {
    coeff: F,
    cells: []const Cell,
};

/// A constraint: sum of terms that must equal zero.
///
/// When evaluated at a row, computes: sum(term.evaluate(row)) which must equal 0.
/// The valid row range is automatically computed from the rotations used.
pub const Constraint = struct {
    terms: []const Term,

    /// Compute valid row range from rotations in terms.
    ///
    /// A constraint is only valid on rows where all cell accesses are in bounds:
    /// - rot = 1 means we can't evaluate at the last row (would access row+1)
    /// - rot = -1 means we can't evaluate at row 0 (would access row-1)
    ///
    /// Returns [start, end) range where the constraint can be evaluated.
    pub fn validRowRange(self: Constraint, num_rows: usize) struct { start: usize, end: usize } {
        var min_rot: i8 = 0;
        var max_rot: i8 = 0;

        for (self.terms) |term| {
            for (term.cells) |cell| {
                min_rot = @min(min_rot, cell.rot);
                max_rot = @max(max_rot, cell.rot);
            }
        }

        // If min_rot = -1, start = 1 (can't access row -1)
        // If max_rot = 2, end = num_rows - 2 (can't access row num_rows+1)
        const start: usize = if (min_rot < 0) @intCast(-@as(i16, min_rot)) else 0;
        const end: usize = if (max_rot > 0)
            num_rows - @as(usize, @intCast(max_rot))
        else
            num_rows;

        return .{ .start = start, .end = end };
    }
};

// ============ Tests ============ //

test "Cell construction" {
    const c1 = Cell{ .col = 0 };
    try std.testing.expectEqual(@as(usize, 0), c1.col);
    try std.testing.expectEqual(@as(i8, 0), c1.rot);

    const c2 = Cell{ .col = 3, .rot = 1 };
    try std.testing.expectEqual(@as(usize, 3), c2.col);
    try std.testing.expectEqual(@as(i8, 1), c2.rot);

    const c3 = Cell{ .col = 5, .rot = -1 };
    try std.testing.expectEqual(@as(usize, 5), c3.col);
    try std.testing.expectEqual(@as(i8, -1), c3.rot);
}

test "Term construction" {
    const cells = [_]Cell{
        .{ .col = 0 },
        .{ .col = 1, .rot = 1 },
    };
    const term = Term{
        .coeff = F.one,
        .cells = &cells,
    };
    try std.testing.expectEqual(@as(usize, 2), term.cells.len);
    try std.testing.expect(term.coeff.eql(F.one));

    // Empty cells = constant term
    const const_term = Term{
        .coeff = F.fromU32(42),
        .cells = &.{},
    };
    try std.testing.expectEqual(@as(usize, 0), const_term.cells.len);
}

test "Constraint.validRowRange - no rotation" {
    // Constraint with rot=0 only: a * b - c = 0
    const constraint = Constraint{
        .terms = &.{
            .{ .coeff = F.one, .cells = &.{ .{ .col = 0 }, .{ .col = 1 } } },
            .{ .coeff = F.neg(F.one), .cells = &.{.{ .col = 2 }} },
        },
    };

    const range = constraint.validRowRange(8);
    try std.testing.expectEqual(@as(usize, 0), range.start);
    try std.testing.expectEqual(@as(usize, 8), range.end);
}

test "Constraint.validRowRange - positive rotation" {
    // Constraint with rot=1: state[row+1] - state[row] = 0
    const constraint = Constraint{
        .terms = &.{
            .{ .coeff = F.one, .cells = &.{.{ .col = 0, .rot = 1 }} },
            .{ .coeff = F.neg(F.one), .cells = &.{.{ .col = 0, .rot = 0 }} },
        },
    };

    const range = constraint.validRowRange(8);
    try std.testing.expectEqual(@as(usize, 0), range.start);
    try std.testing.expectEqual(@as(usize, 7), range.end); // can't evaluate at row 7
}

test "Constraint.validRowRange - negative rotation" {
    // Constraint with rot=-1: state[row] - state[row-1] = 0
    const constraint = Constraint{
        .terms = &.{
            .{ .coeff = F.one, .cells = &.{.{ .col = 0, .rot = 0 }} },
            .{ .coeff = F.neg(F.one), .cells = &.{.{ .col = 0, .rot = -1 }} },
        },
    };

    const range = constraint.validRowRange(8);
    try std.testing.expectEqual(@as(usize, 1), range.start); // can't evaluate at row 0
    try std.testing.expectEqual(@as(usize, 8), range.end);
}

test "Constraint.validRowRange - rot=2" {
    // Fibonacci: fib[row] + fib[row+1] = fib[row+2]
    const constraint = Constraint{
        .terms = &.{
            .{ .coeff = F.one, .cells = &.{.{ .col = 0, .rot = 0 }} },
            .{ .coeff = F.one, .cells = &.{.{ .col = 0, .rot = 1 }} },
            .{ .coeff = F.neg(F.one), .cells = &.{.{ .col = 0, .rot = 2 }} },
        },
    };

    const range = constraint.validRowRange(8);
    try std.testing.expectEqual(@as(usize, 0), range.start);
    try std.testing.expectEqual(@as(usize, 6), range.end); // can't evaluate at rows 6, 7
}

test "Constraint.validRowRange - mixed rotations" {
    // Constraint with both rot=-1 and rot=1
    const constraint = Constraint{
        .terms = &.{
            .{ .coeff = F.one, .cells = &.{.{ .col = 0, .rot = -1 }} },
            .{ .coeff = F.one, .cells = &.{.{ .col = 0, .rot = 0 }} },
            .{ .coeff = F.neg(F.one), .cells = &.{.{ .col = 0, .rot = 1 }} },
        },
    };

    const range = constraint.validRowRange(8);
    try std.testing.expectEqual(@as(usize, 1), range.start); // can't evaluate at row 0
    try std.testing.expectEqual(@as(usize, 7), range.end); // can't evaluate at row 7
}

test "Constraint.validRowRange - empty constraint" {
    // Edge case: constraint with no terms
    const constraint = Constraint{
        .terms = &.{},
    };

    const range = constraint.validRowRange(8);
    try std.testing.expectEqual(@as(usize, 0), range.start);
    try std.testing.expectEqual(@as(usize, 8), range.end);
}

test "Constraint.validRowRange - constant only" {
    // Constraint that's just a constant (empty cells in term)
    const constraint = Constraint{
        .terms = &.{
            .{ .coeff = F.fromU32(42), .cells = &.{} },
        },
    };

    const range = constraint.validRowRange(8);
    try std.testing.expectEqual(@as(usize, 0), range.start);
    try std.testing.expectEqual(@as(usize, 8), range.end);
}
