const std = @import("std");
const Allocator = std.mem.Allocator;

/// Trace parameterized by field type.
/// Column-major storage for constraint evaluation.
pub fn Trace(comptime F: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,

        /// Number of rows (must be power of 2)
        num_rows: usize,

        /// Column data - each column is values[row]
        /// Column-major for cache-friendly polynomial ops
        columns: std.ArrayListUnmanaged([]F),

        /// Track which cells have been set (for error detection)
        set_flags: std.ArrayListUnmanaged([]bool),

        pub fn init(allocator: Allocator, num_rows: usize) !Self {
            // Validate power of 2
            if (num_rows == 0 or (num_rows & (num_rows - 1)) != 0) {
                return error.NotPowerOfTwo;
            }
            return .{
                .allocator = allocator,
                .num_rows = num_rows,
                .columns = .{},
                .set_flags = .{},
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.columns.items) |col| {
                self.allocator.free(col);
            }
            for (self.set_flags.items) |flags| {
                self.allocator.free(flags);
            }
            self.columns.deinit(self.allocator);
            self.set_flags.deinit(self.allocator);
        }

        /// Add a column, returns column index
        pub fn addColumn(self: *Self) !usize {
            const col = try self.allocator.alloc(F, self.num_rows);
            @memset(col, F.zero);

            const flags = try self.allocator.alloc(bool, self.num_rows);
            @memset(flags, false);

            try self.columns.append(self.allocator, col);
            try self.set_flags.append(self.allocator, flags);

            return self.columns.items.len - 1;
        }

        /// Number of columns
        pub fn numColumns(self: Self) usize {
            return self.columns.items.len;
        }

        /// Set a cell value
        pub fn set(self: *Self, col: usize, row: usize, val: F) !void {
            if (col >= self.columns.items.len) {
                return error.InvalidColumn;
            }
            if (row >= self.num_rows) {
                return error.InvalidRow;
            }
            self.columns.items[col][row] = val;
            self.set_flags.items[col][row] = true;
        }

        /// Get a cell value with rotation
        /// Returns error if cell was never set or rotation is out of bounds
        pub fn get(self: *const Self, col: usize, row: usize, rot: i32) !F {
            if (col >= self.columns.items.len) {
                return error.InvalidColumn;
            }

            const row_i64: i64 = @intCast(row);
            const actual_row_i64 = row_i64 + rot;

            if (actual_row_i64 < 0 or actual_row_i64 >= self.num_rows) {
                return error.RotationOutOfBounds;
            }

            const actual_row: usize = @intCast(actual_row_i64);

            if (!self.set_flags.items[col][actual_row]) {
                return error.CellNotSet;
            }

            return self.columns.items[col][actual_row];
        }
    };
}

// ============ Tests ============ //

const testing = std.testing;
const M31 = @import("../fields/mersenne31.zig").Mersenne31;

test "Trace: init requires power of 2" {
    try testing.expectError(error.NotPowerOfTwo, Trace(M31).init(testing.allocator, 0));
    try testing.expectError(error.NotPowerOfTwo, Trace(M31).init(testing.allocator, 3));
    try testing.expectError(error.NotPowerOfTwo, Trace(M31).init(testing.allocator, 5));

    // Valid powers of 2
    var t1 = try Trace(M31).init(testing.allocator, 1);
    defer t1.deinit();
    var t2 = try Trace(M31).init(testing.allocator, 4);
    defer t2.deinit();
    var t4 = try Trace(M31).init(testing.allocator, 16);
    defer t4.deinit();
}

test "Trace: addColumn allocates storage" {
    var trace = try Trace(M31).init(testing.allocator, 4);
    defer trace.deinit();

    try testing.expectEqual(@as(usize, 0), trace.numColumns());

    const col0 = try trace.addColumn();
    try testing.expectEqual(@as(usize, 0), col0);
    try testing.expectEqual(@as(usize, 1), trace.numColumns());

    const col1 = try trace.addColumn();
    try testing.expectEqual(@as(usize, 1), col1);
    try testing.expectEqual(@as(usize, 2), trace.numColumns());
}

test "Trace: set and get" {
    var trace = try Trace(M31).init(testing.allocator, 4);
    defer trace.deinit();

    const col = try trace.addColumn();
    try trace.set(col, 2, M31.fromU64(42));

    const val = try trace.get(col, 2, 0);
    try testing.expect(val.eql(M31.fromU64(42)));
}

test "Trace: get with rotation" {
    var trace = try Trace(M31).init(testing.allocator, 4);
    defer trace.deinit();

    const col = try trace.addColumn();
    try trace.set(col, 0, M31.fromU64(10));
    try trace.set(col, 1, M31.fromU64(20));
    try trace.set(col, 2, M31.fromU64(30));
    try trace.set(col, 3, M31.fromU64(40));

    // From row 1, access row 0 (rot = -1)
    try testing.expect((try trace.get(col, 1, -1)).eql(M31.fromU64(10)));

    // From row 1, access row 2 (rot = 1)
    try testing.expect((try trace.get(col, 1, 1)).eql(M31.fromU64(30)));

    // From row 1, access row 3 (rot = 2)
    try testing.expect((try trace.get(col, 1, 2)).eql(M31.fromU64(40)));
}

test "Trace: get returns error on invalid column" {
    var trace = try Trace(M31).init(testing.allocator, 4);
    defer trace.deinit();

    try testing.expectError(error.InvalidColumn, trace.get(0, 0, 0));
}

test "Trace: set returns error on invalid column" {
    var trace = try Trace(M31).init(testing.allocator, 4);
    defer trace.deinit();

    try testing.expectError(error.InvalidColumn, trace.set(0, 0, M31.one));
}

test "Trace: set returns error on invalid row" {
    var trace = try Trace(M31).init(testing.allocator, 4);
    defer trace.deinit();

    const col = try trace.addColumn();
    try testing.expectError(error.InvalidRow, trace.set(col, 4, M31.one));
}

test "Trace: get returns error on rotation out of bounds" {
    var trace = try Trace(M31).init(testing.allocator, 4);
    defer trace.deinit();

    const col = try trace.addColumn();
    try trace.set(col, 0, M31.one);
    try trace.set(col, 3, M31.one);

    // From row 0, cannot go to row -1
    try testing.expectError(error.RotationOutOfBounds, trace.get(col, 0, -1));

    // From row 3, cannot go to row 4
    try testing.expectError(error.RotationOutOfBounds, trace.get(col, 3, 1));
}

test "Trace: get returns error on unset cell" {
    var trace = try Trace(M31).init(testing.allocator, 4);
    defer trace.deinit();

    const col = try trace.addColumn();
    // Column added but no values set

    try testing.expectError(error.CellNotSet, trace.get(col, 0, 0));
}
