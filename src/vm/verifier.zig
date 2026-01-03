const std = @import("std");
const F = @import("../fields/mersenne31.zig").Mersenne31;
const prover = @import("prover.zig");
const Basefold = @import("../pcs/basefold.zig").Basefold(F);
const Transcript = @import("../transcript.zig").Transcript;

pub const VerifierConfig = struct {
    /// Number of FRI queries (must match prover config)
    num_queries: usize = 16,
};

/// Verify a VM execution proof.
/// Returns true if proof is valid and output matches expected.
pub fn verify(
    expected_output: F,
    proof: *const prover.VMProof,
    allocator: std.mem.Allocator,
    config: VerifierConfig,
) !bool {
    // 1. Check output matches expected
    if (!proof.output.eql(expected_output)) {
        return false;
    }

    // 2. Derive the same random point as prover
    const r_point = try deriveRandomPoint(proof.num_vars, allocator);
    defer allocator.free(r_point);

    // 3. Verify Basefold proof that constraint polynomial sums to 0
    // The claimed value is 0 (zerocheck)
    const claimed_value = F.zero;
    const basefold_config = Basefold.Config{ .num_queries = config.num_queries };

    return try Basefold.verify(
        allocator,
        claimed_value,
        r_point,
        &proof.basefold_proof,
        basefold_config,
    );
}

/// Verify proof without checking output.
/// Useful when output is not known in advance.
pub fn verifyProof(
    proof: *const prover.VMProof,
    allocator: std.mem.Allocator,
    config: VerifierConfig,
) !bool {
    // Derive the same random point as prover
    const r_point = try deriveRandomPoint(proof.num_vars, allocator);
    defer allocator.free(r_point);

    // Verify Basefold proof that constraint polynomial sums to 0
    const claimed_value = F.zero;
    const basefold_config = Basefold.Config{ .num_queries = config.num_queries };

    return try Basefold.verify(
        allocator,
        claimed_value,
        r_point,
        &proof.basefold_proof,
        basefold_config,
    );
}

/// Derive a random evaluation point for zerocheck.
/// Must match prover's deriveRandomPoint exactly.
fn deriveRandomPoint(num_vars: usize, allocator: std.mem.Allocator) ![]F {
    var transcript = Transcript(F).init("vm-zerocheck");

    // Add domain separation for num_vars
    transcript.absorb(F.fromU32(@intCast(num_vars)));

    const r_point = try allocator.alloc(F, num_vars);
    for (r_point) |*r| {
        r.* = transcript.squeeze();
    }

    return r_point;
}

// ============ Tests ============ //

const testing = std.testing;
const stack = @import("stack.zig");

test "verifier: accepts valid proof" {
    const allocator = testing.allocator;

    const program = [_]stack.Instruction{
        .{ .PUSH = 10 },
        .{ .PUSH = 20 },
        .{ .ADD = {} },
    };

    const proof = try prover.prove(&program, allocator, .{});
    defer proof.deinit(allocator);

    const expected = F.fromU32(30);
    const valid = try verify(expected, &proof, allocator, .{});

    try testing.expect(valid);
}

test "verifier: rejects wrong output" {
    const allocator = testing.allocator;

    const program = [_]stack.Instruction{
        .{ .PUSH = 10 },
        .{ .PUSH = 20 },
        .{ .ADD = {} },
    };

    const proof = try prover.prove(&program, allocator, .{});
    defer proof.deinit(allocator);

    // Wrong expected output
    const wrong_expected = F.fromU32(999);
    const valid = try verify(wrong_expected, &proof, allocator, .{});

    try testing.expect(!valid);
}

test "verifier: verifyProof without output check" {
    const allocator = testing.allocator;

    const program = [_]stack.Instruction{
        .{ .PUSH = 7 },
        .{ .PUSH = 6 },
        .{ .MUL = {} },
    };

    const proof = try prover.prove(&program, allocator, .{});
    defer proof.deinit(allocator);

    // Verify just the proof validity
    const valid = try verifyProof(&proof, allocator, .{});

    try testing.expect(valid);
    // Output is 42
    try testing.expect(proof.output.eql(F.fromU32(42)));
}

test "verifier: fibonacci proof" {
    const allocator = testing.allocator;

    // fib(7) = 13
    const program = [_]stack.Instruction{
        .{ .PUSH = 1 },
        .{ .PUSH = 1 },
        .{ .SWAP = {} },
        .{ .OVER = {} },
        .{ .ADD = {} },
        .{ .SWAP = {} },
        .{ .OVER = {} },
        .{ .ADD = {} },
        .{ .SWAP = {} },
        .{ .OVER = {} },
        .{ .ADD = {} },
        .{ .SWAP = {} },
        .{ .OVER = {} },
        .{ .ADD = {} },
        .{ .SWAP = {} },
        .{ .OVER = {} },
        .{ .ADD = {} },
    };

    const proof = try prover.prove(&program, allocator, .{});
    defer proof.deinit(allocator);

    const expected = F.fromU32(13);
    const valid = try verify(expected, &proof, allocator, .{});

    try testing.expect(valid);
}
