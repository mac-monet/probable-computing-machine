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

        /// Which columns are public (for verification)
        public_columns: std.ArrayListUnmanaged(usize),

        pub fn init(allocator: Allocator, num_rows: usize) !Self {
            // Validate power of 2
            if (num_rows == 0 or (num_rows & (num_rows - 1)) != 0) {
                return error.NotPowerOfTwo;
            }
            return .{
                .allocator = allocator,
                .num_rows = num_rows,
                .columns = .{},
                .public_columns = .{},
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.columns.items) |col| {
                self.allocator.free(col);
            }
            self.columns.deinit(self.allocator);
            self.public_columns.deinit(self.allocator);
        }

        /// Add a column, returns column index
        pub fn addColumn(self: *Self) !usize {
            const col = try self.allocator.alloc(F, self.num_rows);
            @memset(col, F.zero);
            try self.columns.append(self.allocator, col);
            return self.columns.items.len - 1;
        }

        /// Number of columns
        pub fn numColumns(self: Self) usize {
            return self.columns.items.len;
        }

        /// Mark a column as public
        pub fn markPublic(self: *Self, col: usize) !void {
            if (col >= self.columns.items.len) {
                return error.InvalidColumn;
            }
            try self.public_columns.append(self.allocator, col);
        }

        /// Extract public column values (for verifier)
        pub fn getPublicValues(self: Self, allocator: Allocator) ![][]F {
            var result = try allocator.alloc([]F, self.public_columns.items.len);
            errdefer allocator.free(result);

            for (self.public_columns.items, 0..) |col_idx, i| {
                result[i] = try allocator.dupe(F, self.columns.items[col_idx]);
            }
            return result;
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
        }

        /// Get a cell value with rotation
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

test "Trace: markPublic validates column" {
    var trace = try Trace(M31).init(testing.allocator, 4);
    defer trace.deinit();

    // No columns exist yet
    try testing.expectError(error.InvalidColumn, trace.markPublic(0));

    const col = try trace.addColumn();
    try trace.markPublic(col);

    // Invalid column index
    try testing.expectError(error.InvalidColumn, trace.markPublic(99));
}

test "Trace: getPublicValues returns empty for no public columns" {
    var trace = try Trace(M31).init(testing.allocator, 4);
    defer trace.deinit();

    _ = try trace.addColumn();

    const public_vals = try trace.getPublicValues(testing.allocator);
    defer testing.allocator.free(public_vals);

    try testing.expectEqual(@as(usize, 0), public_vals.len);
}

test "Trace: getPublicValues extracts marked columns" {
    var trace = try Trace(M31).init(testing.allocator, 4);
    defer trace.deinit();

    const a = try trace.addColumn();
    const b = try trace.addColumn();
    const c = try trace.addColumn();

    // Set values
    try trace.set(a, 0, M31.fromU64(10));
    try trace.set(a, 1, M31.fromU64(20));
    try trace.set(a, 2, M31.fromU64(30));
    try trace.set(a, 3, M31.fromU64(40));

    try trace.set(b, 0, M31.fromU64(100));
    try trace.set(b, 1, M31.fromU64(200));
    try trace.set(b, 2, M31.fromU64(300));
    try trace.set(b, 3, M31.fromU64(400));

    try trace.set(c, 0, M31.fromU64(1));
    try trace.set(c, 1, M31.fromU64(2));
    try trace.set(c, 2, M31.fromU64(3));
    try trace.set(c, 3, M31.fromU64(4));

    // Mark columns a and c as public (not b)
    try trace.markPublic(a);
    try trace.markPublic(c);

    const public_vals = try trace.getPublicValues(testing.allocator);
    defer {
        for (public_vals) |col| {
            testing.allocator.free(col);
        }
        testing.allocator.free(public_vals);
    }

    try testing.expectEqual(@as(usize, 2), public_vals.len);

    // First public column is 'a'
    try testing.expect(public_vals[0][0].eql(M31.fromU64(10)));
    try testing.expect(public_vals[0][3].eql(M31.fromU64(40)));

    // Second public column is 'c'
    try testing.expect(public_vals[1][0].eql(M31.fromU64(1)));
    try testing.expect(public_vals[1][3].eql(M31.fromU64(4)));
}
