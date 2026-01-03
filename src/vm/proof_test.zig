const std = @import("std");
const F = @import("../fields/mersenne31.zig").Mersenne31;
const stack = @import("stack.zig");
const prover = @import("prover.zig");
const verifier = @import("verifier.zig");

const testing = std.testing;

// ============ End-to-End Integration Tests ============ //

test "e2e: simple addition" {
    const allocator = testing.allocator;

    const program = [_]stack.Instruction{
        .{ .PUSH = 5 },
        .{ .PUSH = 3 },
        .{ .ADD = {} },
    };

    // Generate proof
    const proof = try prover.prove(&program, allocator, .{});
    defer proof.deinit(allocator);

    // Expected output: 5 + 3 = 8
    const expected = F.fromU32(8);

    // Verify
    const valid = try verifier.verify(expected, &proof, allocator, .{});
    try testing.expect(valid);
}

test "e2e: multiplication" {
    const allocator = testing.allocator;

    const program = [_]stack.Instruction{
        .{ .PUSH = 12 },
        .{ .PUSH = 7 },
        .{ .MUL = {} },
    };

    const proof = try prover.prove(&program, allocator, .{});
    defer proof.deinit(allocator);

    // 12 * 7 = 84
    const expected = F.fromU32(84);
    const valid = try verifier.verify(expected, &proof, allocator, .{});
    try testing.expect(valid);
}

test "e2e: fibonacci fib(7) = 13" {
    const allocator = testing.allocator;

    // Compute fib(7) using SWAP+OVER+ADD pattern
    // State: [fib(n-1), fib(n)] -> SWAP -> OVER -> ADD -> [fib(n), fib(n+1)]
    const program = [_]stack.Instruction{
        .{ .PUSH = 1 }, // [1]           fib(1)
        .{ .PUSH = 1 }, // [1, 1]        [fib(1), fib(2)]
        // Step to fib(3) = 2
        .{ .SWAP = {} },
        .{ .OVER = {} },
        .{ .ADD = {} },
        // Step to fib(4) = 3
        .{ .SWAP = {} },
        .{ .OVER = {} },
        .{ .ADD = {} },
        // Step to fib(5) = 5
        .{ .SWAP = {} },
        .{ .OVER = {} },
        .{ .ADD = {} },
        // Step to fib(6) = 8
        .{ .SWAP = {} },
        .{ .OVER = {} },
        .{ .ADD = {} },
        // Step to fib(7) = 13
        .{ .SWAP = {} },
        .{ .OVER = {} },
        .{ .ADD = {} },
    };

    const proof = try prover.prove(&program, allocator, .{});
    defer proof.deinit(allocator);

    // fib(7) = 13
    const expected = F.fromU32(13);
    const valid = try verifier.verify(expected, &proof, allocator, .{});
    try testing.expect(valid);
}

test "e2e: complex expression (2*3 + 4*5)" {
    const allocator = testing.allocator;

    // Compute (2*3) + (4*5) = 6 + 20 = 26
    const program = [_]stack.Instruction{
        .{ .PUSH = 2 },
        .{ .PUSH = 3 },
        .{ .MUL = {} }, // [6]
        .{ .PUSH = 4 },
        .{ .PUSH = 5 },
        .{ .MUL = {} }, // [6, 20]
        .{ .ADD = {} }, // [26]
    };

    const proof = try prover.prove(&program, allocator, .{});
    defer proof.deinit(allocator);

    const expected = F.fromU32(26);
    const valid = try verifier.verify(expected, &proof, allocator, .{});
    try testing.expect(valid);
}

test "e2e: proof rejects wrong output" {
    const allocator = testing.allocator;

    const program = [_]stack.Instruction{
        .{ .PUSH = 10 },
        .{ .PUSH = 20 },
        .{ .ADD = {} },
    };

    const proof = try prover.prove(&program, allocator, .{});
    defer proof.deinit(allocator);

    // Real output is 30, but we claim 999
    const wrong_output = F.fromU32(999);
    const valid = try verifier.verify(wrong_output, &proof, allocator, .{});
    try testing.expect(!valid);
}

test "e2e: DUP and ADD for squaring" {
    const allocator = testing.allocator;

    // Compute 7^2 = 49
    const program = [_]stack.Instruction{
        .{ .PUSH = 7 },
        .{ .DUP = {} }, // [7, 7]
        .{ .MUL = {} }, // [49]
    };

    const proof = try prover.prove(&program, allocator, .{});
    defer proof.deinit(allocator);

    const expected = F.fromU32(49);
    const valid = try verifier.verify(expected, &proof, allocator, .{});
    try testing.expect(valid);
}

test "e2e: longer program with all opcodes" {
    const allocator = testing.allocator;

    // Use all non-PUSH opcodes in a computation
    const program = [_]stack.Instruction{
        .{ .PUSH = 2 }, // [2]
        .{ .PUSH = 3 }, // [2, 3]
        .{ .SWAP = {} }, // [3, 2]
        .{ .OVER = {} }, // [3, 2, 3]
        .{ .MUL = {} }, // [3, 6]
        .{ .DUP = {} }, // [3, 6, 6]
        .{ .ADD = {} }, // [3, 12]
        .{ .ADD = {} }, // [15]
    };

    const proof = try prover.prove(&program, allocator, .{});
    defer proof.deinit(allocator);

    // 2*3 + 2*3 + 3 = 6 + 6 + 3 = 15
    const expected = F.fromU32(15);
    const valid = try verifier.verify(expected, &proof, allocator, .{});
    try testing.expect(valid);
}

test "e2e: verifyProof ignores output" {
    const allocator = testing.allocator;

    const program = [_]stack.Instruction{
        .{ .PUSH = 100 },
        .{ .PUSH = 200 },
        .{ .ADD = {} },
    };

    const proof = try prover.prove(&program, allocator, .{});
    defer proof.deinit(allocator);

    // verifyProof doesn't check output
    const valid = try verifier.verifyProof(&proof, allocator, .{});
    try testing.expect(valid);

    // But the output is correct
    try testing.expect(proof.output.eql(F.fromU32(300)));
}

test "e2e: power of 2 trace length" {
    const allocator = testing.allocator;

    // Exactly 4 instructions -> 4 rows -> already power of 2
    const program = [_]stack.Instruction{
        .{ .PUSH = 1 },
        .{ .PUSH = 2 },
        .{ .PUSH = 3 },
        .{ .ADD = {} },
    };

    const proof = try prover.prove(&program, allocator, .{});
    defer proof.deinit(allocator);

    // 2 + 3 = 5
    const expected = F.fromU32(5);
    const valid = try verifier.verify(expected, &proof, allocator, .{});
    try testing.expect(valid);
}

test "e2e: non-power-of-2 trace length" {
    const allocator = testing.allocator;

    // 5 instructions -> padded to 8
    const program = [_]stack.Instruction{
        .{ .PUSH = 1 },
        .{ .PUSH = 2 },
        .{ .PUSH = 3 },
        .{ .ADD = {} },
        .{ .ADD = {} },
    };

    const proof = try prover.prove(&program, allocator, .{});
    defer proof.deinit(allocator);

    // (2 + 3) + 1 = 6
    // Wait, that's not right. Let me trace through:
    // PUSH 1 -> [1]
    // PUSH 2 -> [1, 2]
    // PUSH 3 -> [1, 2, 3]
    // ADD -> [1, 5]  (2+3)
    // ADD -> [6]     (1+5)
    const expected = F.fromU32(6);
    const valid = try verifier.verify(expected, &proof, allocator, .{});
    try testing.expect(valid);
}
