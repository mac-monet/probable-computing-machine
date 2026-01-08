const std = @import("std");

/// Generic PCS interface.
/// Implementations (Basefold, FRI, Brakedown) conform to this interface.
///
/// Configuration is comptime - baked into the Impl type at instantiation.
/// Example: `const MyPCS = PCS(M31, Basefold(M31, .{ .num_queries = 32 }));`
///
/// A PCS implementation must provide:
/// - Commitment: type of commitment (e.g., Merkle root)
/// - OpeningProof: type of opening proof (fully bounded, no allocation)
/// - commit(evals) -> Commitment
/// - prove(allocator, evals, point) -> OpeningProof
/// - verify(allocator, claimed_value, point, proof) -> bool
pub fn PCS(comptime F: type, comptime Impl: type) type {
    // Compile-time interface verification
    comptime {
        _ = Impl.Commitment;
        _ = Impl.OpeningProof;
    }

    return struct {
        pub const Commitment = Impl.Commitment;
        pub const OpeningProof = Impl.OpeningProof;

        /// Commit to polynomial evaluations.
        pub fn commit(evals: []const F) Commitment {
            return Impl.commit(evals);
        }

        /// Open polynomial at a point.
        /// Allocator used for scratch space only - proof is fully bounded.
        pub fn open(
            allocator: std.mem.Allocator,
            evals: []const F,
            point: []const F,
        ) !OpeningProof {
            return Impl.open(allocator, evals, point);
        }

        /// Verify opening proof.
        pub fn verify(
            allocator: std.mem.Allocator,
            claimed_value: F,
            point: []const F,
            proof: *const OpeningProof,
        ) !bool {
            return Impl.verify(allocator, claimed_value, point, proof);
        }
    };
}
