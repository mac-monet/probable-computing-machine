const std = @import("std");

pub fn MerkleTree(comptime F: type) type {
    return struct {
        const Self = @This();
        const Hash = [32]u8;

        /// Commitment is just the root hash
        pub const Commitment = Hash;

        /// Build tree from evaluations, return root
        pub fn commit(evals: []const F) Commitment {
            const Blake3 = std.crypto.hash.Blake3;

            // For now, just hash all evals concatenated
            // (This is the simplest "commitment" - not a real tree yet)
            var hasher = Blake3.init(.{});
            for (evals) |e| {
                hasher.update(&e.toBytes());
            }
            var root: [32]u8 = undefined;
            hasher.final(&root);
            return root;
        }

        /// Verify that evals should hash to the given commitment
        pub fn verifyAll(evals: []const F, commitment: Commitment) bool {
            // Recompute and compare
            return std.mem.eql(u8, &commit(evals), &commitment);
        }
    };
}
