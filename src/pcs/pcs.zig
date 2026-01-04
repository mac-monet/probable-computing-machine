const std = @import("std");

/// Configuration for polynomial commitment schemes.
pub const PCSConfig = struct {
    /// Number of queries for soundness
    num_queries: comptime_int = 80,
};

/// Generic PCS interface.
/// Implementations (Basefold, FRI, Brakedown) conform to this interface.
///
/// A PCS implementation must provide:
/// - Commitment: type of commitment (e.g., Merkle root)
/// - OpeningProof: type of opening proof
/// - commit(ctx, evals) -> Commitment
/// - open(ctx, evals, point, claimed_value) -> OpeningProof
/// - verify(ctx, commitment, point, claimed_value, proof) -> bool
pub fn PCS(comptime F: type, comptime Impl: type) type {
    return struct {
        pub const Commitment = Impl.Commitment;
        pub const OpeningProof = Impl.OpeningProof;
        pub const Config = Impl.Config;

        /// Commit to polynomial evaluations.
        pub fn commit(evals: []const F) Commitment {
            return Impl.commit(evals);
        }

        /// Open polynomial at a point.
        pub fn open(
            allocator: std.mem.Allocator,
            evals: []const F,
            point: []const F,
            config: Config,
        ) !OpeningProof {
            return Impl.prove(allocator, evals, point, config);
        }

        /// Verify opening proof.
        pub fn verify(
            allocator: std.mem.Allocator,
            claimed_value: F,
            point: []const F,
            proof: *const OpeningProof,
            config: Config,
        ) !bool {
            return Impl.verify(allocator, claimed_value, point, proof, config);
        }
    };
}

/// Trait verification for PCS implementations.
/// Use this to check that a type conforms to the PCS interface at comptime.
pub fn verifyPCSImpl(comptime Impl: type) void {
    // Check required types exist
    _ = Impl.Commitment;
    _ = Impl.OpeningProof;
    _ = Impl.Config;

    // Check required functions exist with correct signatures
    // Note: actual signature verification is done by the compiler when used
}
