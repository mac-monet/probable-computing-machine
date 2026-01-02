const std = @import("std");
const Blake3 = std.crypto.hash.Blake3;

pub fn MerkleTree(comptime F: type) type {
    return struct {
        const Self = @This();
        const Hash = [32]u8;

        /// Commitment is just the root hash
        pub const Commitment = Hash;

        /// Build tree from evaluations, return root
        pub fn commit(evals: []const F) Commitment {
            if (evals.len == 0) return std.mem.zeroes(Commitment);
            if (evals.len == 1) return hashLeaf(evals[0]);

            // Build tree bottom-up
            const allocator = std.heap.page_allocator;
            var current = allocator.alloc(Commitment, evals.len) catch unreachable;
            defer allocator.free(current);

            // Hash leaves
            for (evals, 0..) |e, i| {
                current[i] = hashLeaf(e);
            }

            // Build layers
            var size = evals.len;
            while (size > 1) {
                const half = size / 2;
                for (0..half) |i| {
                    current[i] = hashPair(current[2 * i], current[2 * i + 1]);
                }
                size = half;
            }

            return current[0];
        }

        fn hashLeaf(elem: F) Commitment {
            var hasher = Blake3.init(.{});
            hasher.update(&[_]u8{0x00}); // Leaf domain separator
            hasher.update(&elem.toBytes());

            var result: Commitment = undefined;
            hasher.final(&result);
            return result;
        }

        fn hashPair(left: Commitment, right: Commitment) Commitment {
            var hasher = Blake3.init(.{});
            hasher.update(&[_]u8{0x01}); // Internal node domain separator
            hasher.update(&left);
            hasher.update(&right);

            var result: Commitment = undefined;
            hasher.final(&result);
            return result;
        }

        // TODO deprecate
        /// Verify that evals should hash to the given commitment
        pub fn verifyAll(evals: []const F, commitment: Commitment) bool {
            // Recompute and compare
            return std.mem.eql(u8, &commit(evals), &commitment);
        }
    };
}
