const std = @import("std");

/// Protocol configuration.
pub const ProtocolConfig = struct {
    /// Number of FRI/Basefold queries for soundness
    num_queries: usize = 80,
};

/// Generic protocol that composes IOP + PCS.
/// Currently uses Basefold as the PCS.
pub fn Protocol(comptime F: type) type {
    const Basefold = @import("../pcs/basefold.zig").Basefold(F);
    const Transcript = @import("../core/transcript.zig").Transcript;

    return struct {
        const Self = @This();

        pub const Proof = Basefold.Proof;

        /// Prove that a constraint polynomial sums to zero (zerocheck).
        /// constraint_evals: evaluations of constraint polynomial over {0,1}^n
        /// r_point: random evaluation point for zerocheck
        pub fn proveZerocheck(
            allocator: std.mem.Allocator,
            constraint_evals: []const F,
            r_point: []const F,
            config: ProtocolConfig,
        ) !Proof {
            const basefold_config = Basefold.Config{ .num_queries = config.num_queries };
            return Basefold.prove(allocator, constraint_evals, r_point, basefold_config);
        }

        /// Verify a zerocheck proof.
        /// claimed_value: should be F.zero for zerocheck
        /// r_point: same random point used during proving
        pub fn verifyZerocheck(
            allocator: std.mem.Allocator,
            claimed_value: F,
            r_point: []const F,
            proof: *const Proof,
            config: ProtocolConfig,
        ) !bool {
            const basefold_config = Basefold.Config{ .num_queries = config.num_queries };
            return Basefold.verify(allocator, claimed_value, r_point, proof, basefold_config);
        }

        /// Derive a random evaluation point using Fiat-Shamir.
        pub fn deriveRandomPoint(
            domain: []const u8,
            num_vars: usize,
            allocator: std.mem.Allocator,
        ) ![]F {
            var transcript = Transcript(F).init(domain);
            transcript.absorb(F.fromU32(@intCast(num_vars)));

            const r_point = try allocator.alloc(F, num_vars);
            for (r_point) |*r| {
                r.* = transcript.squeeze();
            }
            return r_point;
        }
    };
}
