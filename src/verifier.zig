//! Minimal verifier: verifies proofs from the prover
//!
//! Verification steps:
//! 1. Verify zerocheck (constraint polynomial is zero on hypercube)
//! 2. Verify PCS opening (evaluation claim is correct)

const std = @import("std");
const Allocator = std.mem.Allocator;
const zerocheck = @import("iop/zerocheck.zig");
const basefold = @import("pcs/basefold.zig");
const Transcript = @import("core/transcript.zig").Transcript;
const prover = @import("prover.zig");

/// Verifier parameterized by field and configuration.
/// Must use same config as prover.
pub fn Verifier(comptime F: type, comptime config: prover.Config) type {
    const ZC = zerocheck.Zerocheck(F, config.max_vars);
    const PCS = basefold.Basefold(F, .{
        .max_vars = config.max_vars,
        .num_queries = config.num_queries,
    });
    const Proof = prover.Prover(F, config).Proof;

    return struct {
        const Self = @This();

        /// Verify a proof.
        ///
        /// Args:
        ///   allocator: For small scratch allocations
        ///   proof: The proof to verify
        ///
        /// Returns true if proof is valid, false otherwise.
        pub fn verify(
            allocator: Allocator,
            proof: *const Proof,
        ) !bool {
            // 1. Initialize transcript with same commitment as prover
            var transcript = Transcript(F).init("hyperplonk");
            transcript.absorbBytes(&proof.commitment);

            // 2. Verify zerocheck
            //    Returns the evaluation claim if valid
            const eval_claim = try ZC.verify(allocator, &proof.zerocheck, &transcript);
            if (eval_claim == null) {
                return false;
            }

            // 3. Verify PCS opening matches the evaluation claim
            //    The zerocheck claims: constraint_poly(eval_point) = claimed_value
            //    We need to verify this against the commitment
            const pcs_valid = try PCS.verify(
                allocator,
                eval_claim.?.value,
                eval_claim.?.point,
                &proof.pcs_opening,
            );

            return pcs_valid;
        }
    };
}

// ============ Tests ============ //

const M31 = @import("fields/mersenne31.zig").Mersenne31;
const CircuitBuilder = @import("constraint/builder.zig").CircuitBuilder(M31);

const TestConfig = prover.Config{ .max_vars = 8, .num_queries = 4 };
const TestProver = prover.Prover(M31, TestConfig);
const TestVerifier = Verifier(M31, TestConfig);

test "verifier: accepts valid proof" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Build circuit: a * b = c
    var builder = try CircuitBuilder.init(std.testing.allocator, 4);
    defer builder.deinit();

    const a = try builder.addWitness();
    const b = try builder.addWitness();
    const c = try builder.addPublic();

    try builder.mulGate(a, b, c);

    // Witness: 3 * 7 = 21
    try builder.set(a, 0, M31.fromU64(3));
    try builder.set(b, 0, M31.fromU64(7));
    try builder.set(c, 0, M31.fromU64(21));

    for (1..4) |row| {
        try builder.set(a, row, M31.zero);
        try builder.set(b, row, M31.zero);
        try builder.set(c, row, M31.zero);
    }

    const circuit = builder.build();

    // Prove
    const proof = try TestProver.prove(allocator, circuit.constraints, circuit.trace);

    // Verify with fresh allocator
    var verify_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer verify_arena.deinit();

    const valid = try TestVerifier.verify(verify_arena.allocator(), &proof);
    try std.testing.expect(valid);
}

test "verifier: larger circuit (8 rows)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var builder = try CircuitBuilder.init(std.testing.allocator, 8);
    defer builder.deinit();

    const a = try builder.addWitness();
    const b = try builder.addWitness();
    const c = try builder.addPublic();

    try builder.mulGate(a, b, c);

    // Fill all rows with valid witnesses
    const test_cases = [_][3]u64{
        .{ 2, 3, 6 },
        .{ 4, 5, 20 },
        .{ 7, 8, 56 },
        .{ 1, 9, 9 },
        .{ 10, 10, 100 },
        .{ 0, 0, 0 },
        .{ 1, 1, 1 },
        .{ 2, 2, 4 },
    };

    for (test_cases, 0..) |tc, row| {
        try builder.set(a, row, M31.fromU64(tc[0]));
        try builder.set(b, row, M31.fromU64(tc[1]));
        try builder.set(c, row, M31.fromU64(tc[2]));
    }

    const circuit = builder.build();
    const proof = try TestProver.prove(allocator, circuit.constraints, circuit.trace);

    var verify_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer verify_arena.deinit();

    const valid = try TestVerifier.verify(verify_arena.allocator(), &proof);
    try std.testing.expect(valid);
}
