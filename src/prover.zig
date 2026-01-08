//! Minimal prover: orchestrates constraint → zerocheck → PCS
//!
//! This is the composition layer that wires together:
//! - Constraint arithmetization
//! - Zerocheck IOP
//! - Basefold PCS

const std = @import("std");
const Allocator = std.mem.Allocator;
const constraint = @import("constraint/constraint.zig");
const zerocheck = @import("iop/zerocheck.zig");
const basefold = @import("pcs/basefold.zig");
const Transcript = @import("core/transcript.zig").Transcript;

/// Prover configuration
pub const Config = struct {
    /// Maximum number of variables (log2 of max rows)
    max_vars: comptime_int = 16,
    /// Number of PCS queries for soundness
    num_queries: comptime_int = 32,
};

/// Minimal prover parameterized by field and configuration.
pub fn Prover(comptime F: type, comptime config: Config) type {
    const CS = constraint.ConstraintSystem(F);
    const ZC = zerocheck.Zerocheck(F, config.max_vars);
    const PCS = basefold.Basefold(F, .{
        .max_vars = config.max_vars,
        .num_queries = config.num_queries,
    });

    return struct {
        const Self = @This();

        /// Complete proof structure
        pub const Proof = struct {
            /// Commitment to constraint polynomial
            commitment: PCS.Commitment,
            /// Zerocheck proof
            zerocheck: ZC.Proof,
            /// PCS opening proof for final evaluation claim
            pcs_opening: PCS.OpeningProof,
        };

        /// Generate a proof that constraints are satisfied.
        ///
        /// Args:
        ///   allocator: Arena recommended for efficient scratch allocation
        ///   constraints: The constraint set to prove
        ///   trace: Witness data satisfying constraints
        ///
        /// Returns proof if witness is valid, error otherwise.
        pub fn prove(
            allocator: Allocator,
            constraints: []const CS.Constraint,
            trace: *const CS.Trace,
        ) !Proof {
            // 1. Evaluate: constraints + trace → polynomial evaluations
            const evals = try CS.evaluate(constraints, trace, allocator);

            // Quick sanity check: if witness is invalid, fail fast
            if (CS.isSatisfied(evals)) |_| {
                return error.ConstraintNotSatisfied;
            }

            // 2. Commit to constraint polynomial
            const commitment = PCS.commit(evals);

            // 3. Initialize transcript with commitment
            var transcript = Transcript(F).init("hyperplonk");
            transcript.absorbBytes(&commitment);

            // 4. Run zerocheck to prove polynomial is zero on hypercube
            const zc_result = try ZC.prove(allocator, evals, &transcript);

            // 5. Run PCS to prove evaluation claim from zerocheck
            const pcs_opening = try PCS.open(allocator, evals, zc_result.eval_point);

            return Proof{
                .commitment = commitment,
                .zerocheck = zc_result.proof,
                .pcs_opening = pcs_opening,
            };
        }
    };
}

// ============ Tests ============ //

const M31 = @import("fields/mersenne31.zig").Mersenne31;
const CircuitBuilder = @import("constraint/builder.zig").CircuitBuilder(M31);

const TestConfig = Config{ .max_vars = 8, .num_queries = 4 };
const TestProver = Prover(M31, TestConfig);

test "prover: simple multiplication circuit" {
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

    // Generate proof
    const proof = try TestProver.prove(allocator, circuit.constraints, circuit.trace);

    // Proof should exist (basic sanity)
    try std.testing.expect(proof.zerocheck.num_vars == 2); // log2(4) = 2
}

test "prover: rejects invalid witness" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var builder = try CircuitBuilder.init(std.testing.allocator, 4);
    defer builder.deinit();

    const a = try builder.addWitness();
    const b = try builder.addWitness();
    const c = try builder.addPublic();

    try builder.mulGate(a, b, c);

    // WRONG witness: 3 * 7 = 20 (should be 21)
    try builder.set(a, 0, M31.fromU64(3));
    try builder.set(b, 0, M31.fromU64(7));
    try builder.set(c, 0, M31.fromU64(20));

    for (1..4) |row| {
        try builder.set(a, row, M31.zero);
        try builder.set(b, row, M31.zero);
        try builder.set(c, row, M31.zero);
    }

    const circuit = builder.build();

    // Should fail
    const result = TestProver.prove(allocator, circuit.constraints, circuit.trace);
    try std.testing.expectError(error.ConstraintNotSatisfied, result);
}
