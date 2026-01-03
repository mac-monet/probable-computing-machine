const std = @import("std");
const trace = @import("trace.zig");
const F = @import("../fields/mersenne31.zig").Mersenne31;

/// Instruction opcodes
pub const Opcode = enum(u8) {
    PUSH, // Push constant to stack
    ADD, // Pop two, push sum
    MUL, // Pop two, push product
    DUP, // Duplicate top of stack
    SWAP, // Swap top two elements
    OVER, // Copy second-to-top to top
};

/// A single instruction
pub const Instruction = union(Opcode) {
    PUSH: u32, // Immediate value to push
    ADD: void,
    MUL: void,
    DUP: void,
    SWAP: void,
    OVER: void,
};

/// A program is a fixed sequence of instructions
pub const Program = []const Instruction;

/// VM state at a single step
pub const State = struct {
    step: u32, // Step number (0-indexed)
    pc: u32, // Program counter
    stack: [8]F, // Fixed-size stack (max depth 8)
    stack_depth: u8, // Current stack depth (0-8)

    pub fn init() State {
        return .{
            .step = 0,
            .pc = 0,
            .stack = [_]F{F.zero} ** 8,
            .stack_depth = 0,
        };
    }

    /// Get top of stack (TOS)
    pub fn top(self: *const State) F {
        if (self.stack_depth == 0) return F.zero;
        return self.stack[self.stack_depth - 1];
    }

    /// Get second element (TOS-1)
    pub fn next(self: *const State) F {
        if (self.stack_depth < 2) return F.zero;
        return self.stack[self.stack_depth - 2];
    }
};

pub const Error = error{
    StackUnderflow,
    StackOverflow,
};

/// Execute a program and return the final state
pub fn execute(program: Program, allocator: std.mem.Allocator) Error!State {
    _ = allocator;
    var state = State.init();

    for (program) |inst| {
        try executeInstruction(&state, inst);
    }

    return state;
}

/// Execute program and generate full execution trace
pub fn executeAndTrace(program: Program, allocator: std.mem.Allocator) (Error || std.mem.Allocator.Error)!trace.Trace {
    const rows = try allocator.alloc(trace.TraceRow, program.len);
    errdefer allocator.free(rows);

    var state = State.init();

    for (program, 0..) |inst, i| {
        // Record current state before execution
        const stack_top = state.top();
        const stack_next = state.next();
        const opcode: u8 = @intFromEnum(inst);

        // Execute instruction
        try executeInstruction(&state, inst);

        // Record row with before/after state
        rows[i] = .{
            .step = state.step - 1, // step was incremented in executeInstruction
            .opcode = opcode,
            .stack_top = stack_top,
            .stack_next = stack_next,
            .next_stack_top = state.top(),
            .next_stack_depth = state.stack_depth,
        };
    }

    return trace.Trace{
        .rows = rows,
        .allocator = allocator,
    };
}

/// Helper: execute single instruction, update state
fn executeInstruction(state: *State, inst: Instruction) Error!void {
    switch (inst) {
        .PUSH => |value| {
            if (state.stack_depth >= 8) return Error.StackOverflow;
            state.stack[state.stack_depth] = F.fromU32(value);
            state.stack_depth += 1;
        },
        .ADD => {
            if (state.stack_depth < 2) return Error.StackUnderflow;
            const b = state.stack[state.stack_depth - 1];
            const a = state.stack[state.stack_depth - 2];
            state.stack_depth -= 1;
            state.stack[state.stack_depth - 1] = a.add(b);
        },
        .MUL => {
            if (state.stack_depth < 2) return Error.StackUnderflow;
            const b = state.stack[state.stack_depth - 1];
            const a = state.stack[state.stack_depth - 2];
            state.stack_depth -= 1;
            state.stack[state.stack_depth - 1] = a.mul(b);
        },
        .DUP => {
            if (state.stack_depth < 1) return Error.StackUnderflow;
            if (state.stack_depth >= 8) return Error.StackOverflow;
            state.stack[state.stack_depth] = state.stack[state.stack_depth - 1];
            state.stack_depth += 1;
        },
        .SWAP => {
            if (state.stack_depth < 2) return Error.StackUnderflow;
            const tmp = state.stack[state.stack_depth - 1];
            state.stack[state.stack_depth - 1] = state.stack[state.stack_depth - 2];
            state.stack[state.stack_depth - 2] = tmp;
        },
        .OVER => {
            if (state.stack_depth < 2) return Error.StackUnderflow;
            if (state.stack_depth >= 8) return Error.StackOverflow;
            state.stack[state.stack_depth] = state.stack[state.stack_depth - 2];
            state.stack_depth += 1;
        },
    }

    state.step += 1;
    state.pc += 1;
}

// ============ Tests ============ //

const testing = std.testing;

test "execute: push and add" {
    const program = [_]Instruction{
        .{ .PUSH = 5 },
        .{ .PUSH = 3 },
        .{ .ADD = {} },
    };

    const state = try execute(&program, testing.allocator);
    // Expected: stack_top = 8, depth = 1
    try testing.expect(state.stack[0].eql(F.fromU64(8)));
    try testing.expectEqual(@as(u8, 1), state.stack_depth);
}

test "execute: push and mul" {
    const program = [_]Instruction{
        .{ .PUSH = 7 },
        .{ .PUSH = 6 },
        .{ .MUL = {} },
    };

    const state = try execute(&program, testing.allocator);
    // Expected: stack_top = 42, depth = 1
    try testing.expect(state.stack[0].eql(F.fromU64(42)));
    try testing.expectEqual(@as(u8, 1), state.stack_depth);
}

test "execute: dup" {
    const program = [_]Instruction{
        .{ .PUSH = 10 },
        .{ .DUP = {} },
    };

    const state = try execute(&program, testing.allocator);
    // Expected: stack = [10, 10], depth = 2
    try testing.expect(state.stack[0].eql(F.fromU64(10)));
    try testing.expect(state.stack[1].eql(F.fromU64(10)));
    try testing.expectEqual(@as(u8, 2), state.stack_depth);
}

test "execute: doubling sequence (DUP+ADD pattern)" {
    // Note: DUP+ADD doubles the top of stack, it doesn't compute Fibonacci.
    // To compute Fibonacci we'd need a SWAP instruction.
    // This test verifies the DUP+ADD pattern works correctly.
    const program = [_]Instruction{
        .{ .PUSH = 1 }, // [1]
        .{ .PUSH = 1 }, // [1, 1]
        .{ .DUP = {} }, // [1, 1, 1]
        .{ .ADD = {} }, // [1, 2]     (1+1=2)
        .{ .DUP = {} }, // [1, 2, 2]
        .{ .ADD = {} }, // [1, 4]     (2+2=4)
        .{ .DUP = {} }, // [1, 4, 4]
        .{ .ADD = {} }, // [1, 8]     (4+4=8)
    };

    const state = try execute(&program, testing.allocator);
    // Top of stack should be 8 (doubling: 1->2->4->8)
    try testing.expect(state.top().eql(F.fromU64(8)));
    // Second should still be 1
    try testing.expect(state.next().eql(F.fromU64(1)));
    try testing.expectEqual(@as(u8, 2), state.stack_depth);
}

test "execute: stack underflow on ADD" {
    const program = [_]Instruction{
        .{ .PUSH = 5 },
        .{ .ADD = {} }, // Only 1 element on stack
    };

    const result = execute(&program, testing.allocator);
    try testing.expectError(Error.StackUnderflow, result);
}

test "execute: stack underflow on MUL" {
    const program = [_]Instruction{
        .{ .MUL = {} }, // Empty stack
    };

    const result = execute(&program, testing.allocator);
    try testing.expectError(Error.StackUnderflow, result);
}

test "execute: stack underflow on DUP" {
    const program = [_]Instruction{
        .{ .DUP = {} }, // Empty stack
    };

    const result = execute(&program, testing.allocator);
    try testing.expectError(Error.StackUnderflow, result);
}

test "execute: stack overflow" {
    const program = [_]Instruction{
        .{ .PUSH = 1 },
        .{ .PUSH = 2 },
        .{ .PUSH = 3 },
        .{ .PUSH = 4 },
        .{ .PUSH = 5 },
        .{ .PUSH = 6 },
        .{ .PUSH = 7 },
        .{ .PUSH = 8 },
        .{ .PUSH = 9 }, // 9th push should overflow
    };

    const result = execute(&program, testing.allocator);
    try testing.expectError(Error.StackOverflow, result);
}

test "trace: generates correct row count" {
    const program = [_]Instruction{
        .{ .PUSH = 1 },
        .{ .PUSH = 2 },
        .{ .ADD = {} },
    };

    var t = try executeAndTrace(&program, testing.allocator);
    defer t.deinit();

    // 3 instructions = 3 rows
    try testing.expectEqual(@as(usize, 3), t.rows.len);
}

test "trace: verify constraints" {
    const program = [_]Instruction{
        .{ .PUSH = 10 },
        .{ .PUSH = 20 },
        .{ .MUL = {} },
    };

    var t = try executeAndTrace(&program, testing.allocator);
    defer t.deinit();

    // Should not error - trace is valid
    try t.verify();
}

test "trace: doubling trace verification" {
    const program = [_]Instruction{
        .{ .PUSH = 1 },
        .{ .PUSH = 1 },
        .{ .DUP = {} },
        .{ .ADD = {} },
        .{ .DUP = {} },
        .{ .ADD = {} },
        .{ .DUP = {} },
        .{ .ADD = {} },
    };

    var t = try executeAndTrace(&program, testing.allocator);
    defer t.deinit();

    try t.verify();
    try testing.expectEqual(@as(usize, 8), t.rows.len);
}

test "execute: swap" {
    const program = [_]Instruction{
        .{ .PUSH = 10 },
        .{ .PUSH = 20 },
        .{ .SWAP = {} },
    };

    const state = try execute(&program, testing.allocator);
    // After swap: [20, 10], TOS = 10
    try testing.expect(state.top().eql(F.fromU64(10)));
    try testing.expect(state.next().eql(F.fromU64(20)));
    try testing.expectEqual(@as(u8, 2), state.stack_depth);
}

test "execute: stack underflow on SWAP" {
    const program = [_]Instruction{
        .{ .PUSH = 5 },
        .{ .SWAP = {} }, // Only 1 element on stack
    };

    const result = execute(&program, testing.allocator);
    try testing.expectError(Error.StackUnderflow, result);
}

test "execute: fibonacci fib(7)" {
    // Correct Fibonacci using SWAP + OVER:
    // State: [fib(n-1), fib(n)] = [a, b]
    // Step: [fib(n), fib(n+1)] = [b, a+b]
    //
    // [a, b]
    // SWAP -> [b, a]
    // OVER -> [b, a, b]
    // ADD  -> [b, a+b] = [fib(n), fib(n+1)] ✓
    //
    // fib(1)=1, fib(2)=1, fib(3)=2, fib(4)=3, fib(5)=5, fib(6)=8, fib(7)=13
    const program = [_]Instruction{
        .{ .PUSH = 1 }, // [1]           fib(1)
        .{ .PUSH = 1 }, // [1, 1]        [fib(1), fib(2)]
        // Step to [fib(2), fib(3)] = [1, 2]
        .{ .SWAP = {} }, // [1, 1]
        .{ .OVER = {} }, // [1, 1, 1]
        .{ .ADD = {} },  // [1, 2]       [fib(2), fib(3)]
        // Step to [fib(3), fib(4)] = [2, 3]
        .{ .SWAP = {} }, // [2, 1]
        .{ .OVER = {} }, // [2, 1, 2]
        .{ .ADD = {} },  // [2, 3]       [fib(3), fib(4)]
        // Step to [fib(4), fib(5)] = [3, 5]
        .{ .SWAP = {} }, // [3, 2]
        .{ .OVER = {} }, // [3, 2, 3]
        .{ .ADD = {} },  // [3, 5]       [fib(4), fib(5)]
        // Step to [fib(5), fib(6)] = [5, 8]
        .{ .SWAP = {} }, // [5, 3]
        .{ .OVER = {} }, // [5, 3, 5]
        .{ .ADD = {} },  // [5, 8]       [fib(5), fib(6)]
        // Step to [fib(6), fib(7)] = [8, 13]
        .{ .SWAP = {} }, // [8, 5]
        .{ .OVER = {} }, // [8, 5, 8]
        .{ .ADD = {} },  // [8, 13]      [fib(6), fib(7)]
    };

    const state = try execute(&program, testing.allocator);
    // Top of stack should be fib(7) = 13
    try testing.expect(state.top().eql(F.fromU64(13)));
    // Second should be fib(6) = 8
    try testing.expect(state.next().eql(F.fromU64(8)));
    try testing.expectEqual(@as(u8, 2), state.stack_depth);
}

test "execute: expression with SWAP" {
    const program = [_]Instruction{
        .{ .PUSH = 3 },
        .{ .PUSH = 5 },
        .{ .SWAP = {} }, // [5, 3]
        .{ .PUSH = 2 },  // [5, 3, 2]
        .{ .MUL = {} },  // [5, 6]
        .{ .ADD = {} },  // [11]
    };

    const state = try execute(&program, testing.allocator);
    // 5 + (3 * 2) = 11
    try testing.expect(state.top().eql(F.fromU64(11)));
    try testing.expectEqual(@as(u8, 1), state.stack_depth);
}

test "trace: swap trace verification" {
    const program = [_]Instruction{
        .{ .PUSH = 10 },
        .{ .PUSH = 20 },
        .{ .SWAP = {} },
    };

    var t = try executeAndTrace(&program, testing.allocator);
    defer t.deinit();

    try t.verify();
    try testing.expectEqual(@as(usize, 3), t.rows.len);

    // Check SWAP row constraints
    const swap_row = t.rows[2];
    try testing.expectEqual(@as(u8, @intFromEnum(Opcode.SWAP)), swap_row.opcode);
    // Before: stack_top=20, stack_next=10
    // After: next_stack_top=10 (swapped)
    try testing.expect(swap_row.stack_top.eql(F.fromU64(20)));
    try testing.expect(swap_row.stack_next.eql(F.fromU64(10)));
    try testing.expect(swap_row.next_stack_top.eql(F.fromU64(10)));
}
