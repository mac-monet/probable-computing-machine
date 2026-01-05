//! Dynamic column storage for the Plonkish constraint system.
//!
//! Core types:
//! - ColumnKind: witness, public, selector, fixed
//! - Column: named column with values and kind
//! - Trace: collection of columns with dynamic sizing
//!
//! The trace stores column-major data for cache-friendly polynomial operations.
//! Power-of-2 padding is applied when finalizing for multilinear polynomial use.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Mersenne31 = @import("../fields/mersenne31.zig").Mersenne31;

/// Field type used throughout the trace system.
pub const F = Mersenne31;

/// Kind of column in the trace.
pub const ColumnKind = enum {
    /// Private witness values computed by the prover.
    witness,
    /// Public inputs visible to both prover and verifier.
    public,
    /// Selector columns (0 or 1) that enable/disable constraints.
    selector,
    /// Fixed columns with values known at circuit definition time.
    fixed,
};

/// A column in the trace.
pub const Column = struct {
    /// Optional name for debugging/identification.
    name: ?[]const u8,
    /// Column values indexed by row.
    values: []F,
    /// Kind of column.
    kind: ColumnKind,

    /// Check if this column contains public values.
    pub fn isPublic(self: Column) bool {
        return self.kind == .public;
    }
};

/// Execution trace with dynamic columns.
///
/// Column-major storage for cache-friendly polynomial operations.
/// Rows must be a power of 2 for multilinear polynomial compatibility.
pub const Trace = struct {
    allocator: Allocator,

    /// Number of rows (may not be power of 2 until padded).
    num_rows: usize,

    /// Dynamic list of columns.
    columns: std.ArrayList(Column),

    /// Initialize a new trace with the given number of rows.
    pub fn init(allocator: Allocator, num_rows: usize) Trace {
        return .{
            .allocator = allocator,
            .num_rows = num_rows,
            .columns = .{},
        };
    }

    /// Free all allocated memory.
    pub fn deinit(self: *Trace) void {
        for (self.columns.items) |col| {
            self.allocator.free(col.values);
        }
        self.columns.deinit(self.allocator);
    }

    /// Add a new column with the given kind. Returns the column index.
    pub fn addColumn(self: *Trace, kind: ColumnKind) !usize {
        return self.addColumnNamed(null, kind);
    }

    /// Add a new column with name and kind. Returns the column index.
    pub fn addColumnNamed(self: *Trace, name: ?[]const u8, kind: ColumnKind) !usize {
        const values = try self.allocator.alloc(F, self.num_rows);
        @memset(values, F.zero);

        try self.columns.append(self.allocator, .{
            .name = name,
            .values = values,
            .kind = kind,
        });

        return self.columns.items.len - 1;
    }

    /// Add a witness column. Convenience method.
    pub fn addWitness(self: *Trace) !usize {
        return self.addColumn(.witness);
    }

    /// Add a public column. Convenience method.
    pub fn addPublic(self: *Trace) !usize {
        return self.addColumn(.public);
    }

    /// Add a selector column. Convenience method.
    pub fn addSelector(self: *Trace) !usize {
        return self.addColumn(.selector);
    }

    /// Add a fixed column. Convenience method.
    pub fn addFixed(self: *Trace) !usize {
        return self.addColumn(.fixed);
    }

    /// Set a cell value.
    pub fn set(self: *Trace, col: usize, row: usize, val: F) void {
        std.debug.assert(col < self.columns.items.len);
        std.debug.assert(row < self.num_rows);
        self.columns.items[col].values[row] = val;
    }

    /// Get a cell value with rotation.
    ///
    /// Rotation specifies offset from the given row:
    /// - rot = 0: current row
    /// - rot = 1: next row
    /// - rot = -1: previous row
    ///
    /// Caller must ensure the resulting row is in bounds (no wraparound).
    pub fn get(self: Trace, col: usize, row: usize, rot: i8) F {
        std.debug.assert(col < self.columns.items.len);

        const actual_row: usize = if (rot >= 0)
            row + @as(usize, @intCast(rot))
        else
            row - @as(usize, @intCast(-rot));

        std.debug.assert(actual_row < self.num_rows);
        return self.columns.items[col].values[actual_row];
    }

    /// Get column values directly (without rotation).
    pub fn getColumn(self: Trace, col: usize) []F {
        std.debug.assert(col < self.columns.items.len);
        return self.columns.items[col].values;
    }

    /// Get column info.
    pub fn getColumnInfo(self: Trace, col: usize) Column {
        std.debug.assert(col < self.columns.items.len);
        return self.columns.items[col];
    }

    /// Number of columns.
    pub fn numColumns(self: Trace) usize {
        return self.columns.items.len;
    }

    /// Extract public column values for verification.
    ///
    /// Returns a slice of slices, one per public column.
    /// Caller owns the returned memory.
    pub fn getPublicValues(self: Trace, allocator: Allocator) ![][]F {
        // Count public columns
        var count: usize = 0;
        for (self.columns.items) |col| {
            if (col.kind == .public) count += 1;
        }

        if (count == 0) {
            return try allocator.alloc([]F, 0);
        }

        var result = try allocator.alloc([]F, count);
        errdefer allocator.free(result);

        var idx: usize = 0;
        for (self.columns.items) |col| {
            if (col.kind == .public) {
                // Copy values
                const copy = try allocator.alloc(F, col.values.len);
                @memcpy(copy, col.values);
                result[idx] = copy;
                idx += 1;
            }
        }

        return result;
    }

    /// Free public values allocated by getPublicValues.
    pub fn freePublicValues(values: [][]F, allocator: Allocator) void {
        for (values) |v| {
            allocator.free(v);
        }
        allocator.free(values);
    }

    /// Pad all columns to a power of 2 length.
    ///
    /// This is required for multilinear polynomial compatibility.
    /// Padding values are set to zero.
    pub fn padToPowerOfTwo(self: *Trace) !void {
        if (self.num_rows == 0) {
            // Special case: pad to 1 row
            try self.resizeColumns(1);
            return;
        }

        const padded_len = std.math.ceilPowerOfTwo(usize, self.num_rows) catch return error.Overflow;
        if (padded_len == self.num_rows) return; // Already power of 2

        try self.resizeColumns(padded_len);
    }

    /// Resize all columns to a new length, zero-filling new rows.
    fn resizeColumns(self: *Trace, new_len: usize) !void {
        for (self.columns.items) |*col| {
            const new_values = try self.allocator.alloc(F, new_len);

            // Copy existing values
            const copy_len = @min(col.values.len, new_len);
            @memcpy(new_values[0..copy_len], col.values[0..copy_len]);

            // Zero-fill new rows
            if (new_len > col.values.len) {
                @memset(new_values[col.values.len..], F.zero);
            }

            self.allocator.free(col.values);
            col.values = new_values;
        }

        self.num_rows = new_len;
    }

    /// Check if num_rows is a power of 2.
    pub fn isPowerOfTwo(self: Trace) bool {
        return self.num_rows > 0 and std.math.isPowerOfTwo(self.num_rows);
    }

    /// Get all column values as a slice of slices.
    /// Useful for passing to polynomial operations.
    pub fn getAllColumnValues(self: Trace, allocator: Allocator) ![][]F {
        var result = try allocator.alloc([]F, self.columns.items.len);
        for (self.columns.items, 0..) |col, i| {
            result[i] = col.values;
        }
        return result;
    }
};

// ============ Tests ============ //

const testing = std.testing;

test "ColumnKind enum values" {
    try testing.expectEqual(@as(usize, 0), @intFromEnum(ColumnKind.witness));
    try testing.expectEqual(@as(usize, 1), @intFromEnum(ColumnKind.public));
    try testing.expectEqual(@as(usize, 2), @intFromEnum(ColumnKind.selector));
    try testing.expectEqual(@as(usize, 3), @intFromEnum(ColumnKind.fixed));
}

test "Column.isPublic" {
    const witness_col = Column{
        .name = null,
        .values = &.{},
        .kind = .witness,
    };
    try testing.expect(!witness_col.isPublic());

    const public_col = Column{
        .name = "input",
        .values = &.{},
        .kind = .public,
    };
    try testing.expect(public_col.isPublic());
}

test "Trace: init and deinit" {
    var trace = Trace.init(testing.allocator, 4);
    defer trace.deinit();

    try testing.expectEqual(@as(usize, 4), trace.num_rows);
    try testing.expectEqual(@as(usize, 0), trace.numColumns());
}

test "Trace: addColumn" {
    var trace = Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const col0 = try trace.addColumn(.witness);
    const col1 = try trace.addColumn(.public);
    const col2 = try trace.addColumnNamed("selector_a", .selector);

    try testing.expectEqual(@as(usize, 0), col0);
    try testing.expectEqual(@as(usize, 1), col1);
    try testing.expectEqual(@as(usize, 2), col2);
    try testing.expectEqual(@as(usize, 3), trace.numColumns());

    // Check kinds
    try testing.expectEqual(ColumnKind.witness, trace.getColumnInfo(col0).kind);
    try testing.expectEqual(ColumnKind.public, trace.getColumnInfo(col1).kind);
    try testing.expectEqual(ColumnKind.selector, trace.getColumnInfo(col2).kind);

    // Check name
    try testing.expectEqualStrings("selector_a", trace.getColumnInfo(col2).name.?);
}

test "Trace: convenience column methods" {
    var trace = Trace.init(testing.allocator, 2);
    defer trace.deinit();

    const w = try trace.addWitness();
    const p = try trace.addPublic();
    const s = try trace.addSelector();
    const f = try trace.addFixed();

    try testing.expectEqual(ColumnKind.witness, trace.getColumnInfo(w).kind);
    try testing.expectEqual(ColumnKind.public, trace.getColumnInfo(p).kind);
    try testing.expectEqual(ColumnKind.selector, trace.getColumnInfo(s).kind);
    try testing.expectEqual(ColumnKind.fixed, trace.getColumnInfo(f).kind);
}

test "Trace: set and get" {
    var trace = Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const col = try trace.addWitness();

    trace.set(col, 0, F.fromU32(10));
    trace.set(col, 1, F.fromU32(20));
    trace.set(col, 2, F.fromU32(30));
    trace.set(col, 3, F.fromU32(40));

    // Get without rotation
    try testing.expect(trace.get(col, 0, 0).eql(F.fromU32(10)));
    try testing.expect(trace.get(col, 1, 0).eql(F.fromU32(20)));
    try testing.expect(trace.get(col, 2, 0).eql(F.fromU32(30)));
    try testing.expect(trace.get(col, 3, 0).eql(F.fromU32(40)));

    // Get with positive rotation
    try testing.expect(trace.get(col, 0, 1).eql(F.fromU32(20)));
    try testing.expect(trace.get(col, 0, 2).eql(F.fromU32(30)));
    try testing.expect(trace.get(col, 1, 2).eql(F.fromU32(40)));

    // Get with negative rotation
    try testing.expect(trace.get(col, 1, -1).eql(F.fromU32(10)));
    try testing.expect(trace.get(col, 3, -2).eql(F.fromU32(20)));
}

test "Trace: getColumn direct access" {
    var trace = Trace.init(testing.allocator, 4);
    defer trace.deinit();

    const col = try trace.addWitness();
    trace.set(col, 0, F.fromU32(100));
    trace.set(col, 2, F.fromU32(300));

    const values = trace.getColumn(col);
    try testing.expectEqual(@as(usize, 4), values.len);
    try testing.expect(values[0].eql(F.fromU32(100)));
    try testing.expect(values[1].eql(F.zero));
    try testing.expect(values[2].eql(F.fromU32(300)));
    try testing.expect(values[3].eql(F.zero));
}

test "Trace: getPublicValues" {
    var trace = Trace.init(testing.allocator, 2);
    defer trace.deinit();

    _ = try trace.addWitness();
    const pub1 = try trace.addPublic();
    _ = try trace.addWitness();
    const pub2 = try trace.addPublic();

    trace.set(pub1, 0, F.fromU32(11));
    trace.set(pub1, 1, F.fromU32(12));
    trace.set(pub2, 0, F.fromU32(21));
    trace.set(pub2, 1, F.fromU32(22));

    const public_values = try trace.getPublicValues(testing.allocator);
    defer Trace.freePublicValues(public_values, testing.allocator);

    try testing.expectEqual(@as(usize, 2), public_values.len);

    // First public column
    try testing.expect(public_values[0][0].eql(F.fromU32(11)));
    try testing.expect(public_values[0][1].eql(F.fromU32(12)));

    // Second public column
    try testing.expect(public_values[1][0].eql(F.fromU32(21)));
    try testing.expect(public_values[1][1].eql(F.fromU32(22)));
}

test "Trace: getPublicValues with no public columns" {
    var trace = Trace.init(testing.allocator, 2);
    defer trace.deinit();

    _ = try trace.addWitness();
    _ = try trace.addSelector();

    const public_values = try trace.getPublicValues(testing.allocator);
    defer Trace.freePublicValues(public_values, testing.allocator);

    try testing.expectEqual(@as(usize, 0), public_values.len);
}

test "Trace: padToPowerOfTwo from 3 to 4" {
    var trace = Trace.init(testing.allocator, 3);
    defer trace.deinit();

    const col = try trace.addWitness();
    trace.set(col, 0, F.fromU32(1));
    trace.set(col, 1, F.fromU32(2));
    trace.set(col, 2, F.fromU32(3));

    try testing.expect(!trace.isPowerOfTwo());
    try testing.expectEqual(@as(usize, 3), trace.num_rows);

    try trace.padToPowerOfTwo();

    try testing.expect(trace.isPowerOfTwo());
    try testing.expectEqual(@as(usize, 4), trace.num_rows);

    // Original values preserved
    try testing.expect(trace.get(col, 0, 0).eql(F.fromU32(1)));
    try testing.expect(trace.get(col, 1, 0).eql(F.fromU32(2)));
    try testing.expect(trace.get(col, 2, 0).eql(F.fromU32(3)));

    // Padding is zero
    try testing.expect(trace.get(col, 3, 0).eql(F.zero));
}

test "Trace: padToPowerOfTwo from 5 to 8" {
    var trace = Trace.init(testing.allocator, 5);
    defer trace.deinit();

    const col = try trace.addWitness();
    for (0..5) |i| {
        trace.set(col, i, F.fromU32(@intCast(i + 1)));
    }

    try trace.padToPowerOfTwo();

    try testing.expectEqual(@as(usize, 8), trace.num_rows);
    try testing.expect(trace.isPowerOfTwo());

    // Original values
    for (0..5) |i| {
        try testing.expect(trace.get(col, i, 0).eql(F.fromU32(@intCast(i + 1))));
    }

    // Padding
    for (5..8) |i| {
        try testing.expect(trace.get(col, i, 0).eql(F.zero));
    }
}

test "Trace: padToPowerOfTwo already power of 2" {
    var trace = Trace.init(testing.allocator, 8);
    defer trace.deinit();

    const col = try trace.addWitness();
    trace.set(col, 0, F.fromU32(42));

    try testing.expect(trace.isPowerOfTwo());

    try trace.padToPowerOfTwo();

    // Unchanged
    try testing.expectEqual(@as(usize, 8), trace.num_rows);
    try testing.expect(trace.get(col, 0, 0).eql(F.fromU32(42)));
}

test "Trace: padToPowerOfTwo empty trace" {
    var trace = Trace.init(testing.allocator, 0);
    defer trace.deinit();

    _ = try trace.addWitness();

    try trace.padToPowerOfTwo();

    try testing.expectEqual(@as(usize, 1), trace.num_rows);
    try testing.expect(trace.isPowerOfTwo());
}

test "Trace: multiple columns with padding" {
    var trace = Trace.init(testing.allocator, 3);
    defer trace.deinit();

    const a = try trace.addWitness();
    const b = try trace.addPublic();
    const c = try trace.addSelector();

    trace.set(a, 0, F.fromU32(10));
    trace.set(b, 1, F.fromU32(20));
    trace.set(c, 2, F.fromU32(1)); // selector

    try trace.padToPowerOfTwo();

    try testing.expectEqual(@as(usize, 4), trace.num_rows);

    // All columns padded
    try testing.expectEqual(@as(usize, 4), trace.getColumn(a).len);
    try testing.expectEqual(@as(usize, 4), trace.getColumn(b).len);
    try testing.expectEqual(@as(usize, 4), trace.getColumn(c).len);

    // Values preserved
    try testing.expect(trace.get(a, 0, 0).eql(F.fromU32(10)));
    try testing.expect(trace.get(b, 1, 0).eql(F.fromU32(20)));
    try testing.expect(trace.get(c, 2, 0).eql(F.fromU32(1)));
}

test "Trace: isPowerOfTwo" {
    {
        var trace = Trace.init(testing.allocator, 1);
        defer trace.deinit();
        try testing.expect(trace.isPowerOfTwo());
    }
    {
        var trace = Trace.init(testing.allocator, 2);
        defer trace.deinit();
        try testing.expect(trace.isPowerOfTwo());
    }
    {
        var trace = Trace.init(testing.allocator, 4);
        defer trace.deinit();
        try testing.expect(trace.isPowerOfTwo());
    }
    {
        var trace = Trace.init(testing.allocator, 8);
        defer trace.deinit();
        try testing.expect(trace.isPowerOfTwo());
    }
    {
        var trace = Trace.init(testing.allocator, 3);
        defer trace.deinit();
        try testing.expect(!trace.isPowerOfTwo());
    }
    {
        var trace = Trace.init(testing.allocator, 5);
        defer trace.deinit();
        try testing.expect(!trace.isPowerOfTwo());
    }
    {
        var trace = Trace.init(testing.allocator, 0);
        defer trace.deinit();
        try testing.expect(!trace.isPowerOfTwo());
    }
}

test "Trace: getAllColumnValues" {
    var trace = Trace.init(testing.allocator, 2);
    defer trace.deinit();

    const a = try trace.addWitness();
    const b = try trace.addPublic();

    trace.set(a, 0, F.fromU32(1));
    trace.set(b, 1, F.fromU32(2));

    const all = try trace.getAllColumnValues(testing.allocator);
    defer testing.allocator.free(all);

    try testing.expectEqual(@as(usize, 2), all.len);
    try testing.expect(all[0][0].eql(F.fromU32(1)));
    try testing.expect(all[1][1].eql(F.fromU32(2)));
}
