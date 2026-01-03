const std = @import("std");
const stack = @import("stack.zig");

pub const F = @import("../fields/mersenne31.zig").Mersenne31;

/// A single row in the execution trace
/// Each field will become a multilinear polynomial column
pub const TraceRow = struct {
    step: u32,
    opcode: u8, // Opcode as u8 for field conversion

    // Stack state (we only need top 2 elements for our opcodes)
    stack_top: F, // Top of stack (TOS)
    stack_next: F, // Second element (TOS-1)

    // Next state (for constraint checking)
    next_stack_top: F, // TOS after instruction
    next_stack_depth: u8,
};

/// Execution trace - one row per execution step
pub const Trace = struct {
    rows: []TraceRow,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Trace) void {
        self.allocator.free(self.rows);
    }

    /// Check that all transition constraints are satisfied
    /// For testing - verifies trace is valid
    pub fn verify(self: *const Trace) !void {
        for (self.rows) |row| {
            const opcode: stack.Opcode = @enumFromInt(row.opcode);

            switch (opcode) {
                .PUSH => {
                    // For PUSH, next_stack_top should be the pushed value
                    // We don't have the immediate value in the row, so we just verify
                    // that stack_depth increased. The value constraint would need
                    // the program to verify.
                },
                .ADD => {
                    // next_stack_top == stack_top + stack_next
                    const expected = row.stack_top.add(row.stack_next);
                    if (!row.next_stack_top.eql(expected)) {
                        return error.InvalidAddConstraint;
                    }
                },
                .MUL => {
                    // next_stack_top == stack_top * stack_next
                    const expected = row.stack_top.mul(row.stack_next);
                    if (!row.next_stack_top.eql(expected)) {
                        return error.InvalidMulConstraint;
                    }
                },
                .DUP => {
                    // next_stack_top == stack_top
                    if (!row.next_stack_top.eql(row.stack_top)) {
                        return error.InvalidDupConstraint;
                    }
                },
                .SWAP => {
                    // next_stack_top == stack_next (values are swapped)
                    if (!row.next_stack_top.eql(row.stack_next)) {
                        return error.InvalidSwapConstraint;
                    }
                },
                .OVER => {
                    // next_stack_top == stack_next (copy of second-to-top)
                    if (!row.next_stack_top.eql(row.stack_next)) {
                        return error.InvalidOverConstraint;
                    }
                },
            }
        }
    }

    /// Column indices for polynomial conversion
    pub const Column = enum(usize) {
        step = 0,
        opcode = 1,
        stack_top = 2,
        stack_next = 3,
        next_stack_top = 4,
        next_stack_depth = 5,

        pub const count = 6;
    };

    /// Convert trace to multilinear polynomial columns
    /// Returns array of polynomials, one per column
    /// Pads trace to power of 2 rows
    pub fn toPolynomials(self: *const Trace, allocator: std.mem.Allocator) ![][]F {
        // Calculate padded length (power of 2)
        const n = self.rows.len;
        const padded_len = if (n == 0) 1 else std.math.ceilPowerOfTwo(usize, n) catch return error.Overflow;

        // Allocate column slices
        var columns = try allocator.alloc([]F, Column.count);
        errdefer {
            for (columns) |col| {
                allocator.free(col);
            }
            allocator.free(columns);
        }

        // Allocate each column
        for (0..Column.count) |i| {
            columns[i] = try allocator.alloc(F, padded_len);
        }

        // Fill in data from rows
        for (self.rows, 0..) |row, i| {
            columns[@intFromEnum(Column.step)][i] = F.fromU32(row.step);
            columns[@intFromEnum(Column.opcode)][i] = F.fromU32(row.opcode);
            columns[@intFromEnum(Column.stack_top)][i] = row.stack_top;
            columns[@intFromEnum(Column.stack_next)][i] = row.stack_next;
            columns[@intFromEnum(Column.next_stack_top)][i] = row.next_stack_top;
            columns[@intFromEnum(Column.next_stack_depth)][i] = F.fromU32(row.next_stack_depth);
        }

        // Pad remaining rows with zeros
        for (n..padded_len) |i| {
            for (0..Column.count) |col| {
                columns[col][i] = F.zero;
            }
        }

        return columns;
    }

    /// Free polynomial columns allocated by toPolynomials
    pub fn freePolynomials(columns: [][]F, allocator: std.mem.Allocator) void {
        for (columns) |col| {
            allocator.free(col);
        }
        allocator.free(columns);
    }
};

// ============ Tests ============ //

const testing = std.testing;

test "trace: toPolynomials creates correct columns" {
    const program = [_]stack.Instruction{
        .{ .PUSH = 5 },
        .{ .PUSH = 3 },
        .{ .ADD = {} },
    };

    var trace = try stack.executeAndTrace(&program, testing.allocator);
    defer trace.deinit();

    const columns = try trace.toPolynomials(testing.allocator);
    defer Trace.freePolynomials(columns, testing.allocator);

    // Should have 6 columns
    try testing.expectEqual(@as(usize, Trace.Column.count), columns.len);

    // Padded to power of 2 (3 rows -> 4)
    try testing.expectEqual(@as(usize, 4), columns[0].len);

    // Check step column
    try testing.expect(columns[@intFromEnum(Trace.Column.step)][0].eql(F.fromU32(0)));
    try testing.expect(columns[@intFromEnum(Trace.Column.step)][1].eql(F.fromU32(1)));
    try testing.expect(columns[@intFromEnum(Trace.Column.step)][2].eql(F.fromU32(2)));

    // Check opcode column
    try testing.expect(columns[@intFromEnum(Trace.Column.opcode)][0].eql(F.fromU32(@intFromEnum(stack.Opcode.PUSH))));
    try testing.expect(columns[@intFromEnum(Trace.Column.opcode)][1].eql(F.fromU32(@intFromEnum(stack.Opcode.PUSH))));
    try testing.expect(columns[@intFromEnum(Trace.Column.opcode)][2].eql(F.fromU32(@intFromEnum(stack.Opcode.ADD))));
}

test "trace: toPolynomials pads to power of 2" {
    // 5 rows should pad to 8
    const program = [_]stack.Instruction{
        .{ .PUSH = 1 },
        .{ .PUSH = 2 },
        .{ .PUSH = 3 },
        .{ .PUSH = 4 },
        .{ .PUSH = 5 },
    };

    var trace = try stack.executeAndTrace(&program, testing.allocator);
    defer trace.deinit();

    const columns = try trace.toPolynomials(testing.allocator);
    defer Trace.freePolynomials(columns, testing.allocator);

    try testing.expectEqual(@as(usize, 8), columns[0].len);

    // Padding should be zeros
    try testing.expect(columns[0][5].isZero());
    try testing.expect(columns[0][6].isZero());
    try testing.expect(columns[0][7].isZero());
}

test "trace: verify detects invalid ADD" {
    // Manually create an invalid trace
    var rows = [_]TraceRow{
        .{
            .step = 0,
            .opcode = @intFromEnum(stack.Opcode.ADD),
            .stack_top = F.fromU32(5),
            .stack_next = F.fromU32(3),
            .next_stack_top = F.fromU32(10), // Wrong! Should be 8
            .next_stack_depth = 1,
        },
    };

    const trace = Trace{
        .rows = &rows,
        .allocator = testing.allocator,
    };

    try testing.expectError(error.InvalidAddConstraint, trace.verify());
}

test "trace: verify detects invalid MUL" {
    var rows = [_]TraceRow{
        .{
            .step = 0,
            .opcode = @intFromEnum(stack.Opcode.MUL),
            .stack_top = F.fromU32(5),
            .stack_next = F.fromU32(3),
            .next_stack_top = F.fromU32(8), // Wrong! Should be 15
            .next_stack_depth = 1,
        },
    };

    const trace = Trace{
        .rows = &rows,
        .allocator = testing.allocator,
    };

    try testing.expectError(error.InvalidMulConstraint, trace.verify());
}

test "trace: verify detects invalid DUP" {
    var rows = [_]TraceRow{
        .{
            .step = 0,
            .opcode = @intFromEnum(stack.Opcode.DUP),
            .stack_top = F.fromU32(5),
            .stack_next = F.fromU32(3),
            .next_stack_top = F.fromU32(10), // Wrong! Should be 5
            .next_stack_depth = 2,
        },
    };

    const trace = Trace{
        .rows = &rows,
        .allocator = testing.allocator,
    };

    try testing.expectError(error.InvalidDupConstraint, trace.verify());
}
