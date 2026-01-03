const std = @import("std");
const Blake3 = std.crypto.hash.Blake3;

pub fn MerkleTree(comptime F: type) type {
    return struct {
        const Self = @This();
        const Hash = [32]u8;

        /// Commitment is just the root hash
        pub const Commitment = Hash;

        pub const MerkleProof = struct {
            /// Sibling hashes from leaf to root
            path: []const Commitment,
            /// Direction bits: false = left, true = right
            indices: []const bool,
        };

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
        /// Open a leaf and return its authentication path
        pub fn open(evals: []const F, index: usize, allocator: std.mem.Allocator) !struct {
            value: F,
            proof: MerkleProof,
        } {
            const depth = std.math.log2_int(usize, evals.len);

            // Build full tree (we need sibling hashes)
            const tree = try buildTree(evals, allocator);
            defer allocator.free(tree);

            var path = try allocator.alloc(Commitment, depth);
            var indices = try allocator.alloc(bool, depth);

            var idx = index;
            var level_start: usize = 0;
            var level_size = evals.len;

            for (0..depth) |i| {
                const sibling_idx = idx ^ 1; // Flip last bit to get sibling
                path[i] = tree[level_start + sibling_idx];
                indices[i] = (idx & 1) == 1; // Am I the right child?

                idx >>= 1;
                level_start += level_size;
                level_size >>= 1;
            }

            return .{
                .value = evals[index],
                .proof = .{ .path = path, .indices = indices },
            };
        }

        /// Verify a Merkle opening
        pub fn verifyOpening(
            root: Commitment,
            index: usize,
            value: F,
            proof: MerkleProof,
        ) bool {
            _ = index;
            var current = hashLeaf(value);

            for (proof.path, proof.indices) |sibling, is_right| {
                if (is_right) {
                    current = hashPair(sibling, current);
                } else {
                    current = hashPair(current, sibling);
                }
            }

            return std.mem.eql(u8, &current, &root);
        }

        /// Build complete Merkle tree, returns all nodes level by level
        fn buildTree(evals: []const F, allocator: std.mem.Allocator) ![]Commitment {
            const n = evals.len;
            // Total nodes: n + n/2 + n/4 + ... + 1 = 2n - 1
            const total_nodes = 2 * n - 1;
            var tree = try allocator.alloc(Commitment, total_nodes);

            // Leaves
            for (evals, 0..) |e, i| {
                tree[i] = hashLeaf(e);
            }

            // Internal nodes
            var level_start: usize = 0;
            var level_size = n;
            var write_pos = n;

            while (level_size > 1) {
                const half = level_size / 2;
                for (0..half) |i| {
                    tree[write_pos + i] = hashPair(tree[level_start + 2 * i], tree[level_start + 2 * i + 1]);
                }
                level_start += level_size;
                level_size = half;
                write_pos += half;
            }

            return tree;
        }

        // TODO deprecate
        /// Verify that evals should hash to the given commitment
        pub fn verifyAll(evals: []const F, commitment: Commitment) bool {
            // Recompute and compare
            return std.mem.eql(u8, &commit(evals), &commitment);
        }
    };
}

const M31 = @import("fields/mersenne31.zig").Mersenne31;
const merkle = MerkleTree(M31);

test "merkle open and verify" {
    const allocator = std.testing.allocator;

    const evals = [_]M31{
        M31.fromU64(1), M31.fromU64(2), M31.fromU64(3), M31.fromU64(4),
    };

    const root = merkle.commit(&evals);

    // Open each leaf and verify
    for (0..evals.len) |i| {
        const opening = try merkle.open(&evals, i, allocator);
        defer allocator.free(opening.proof.path);
        defer allocator.free(opening.proof.indices);

        try std.testing.expect(opening.value.eql(evals[i]));
        try std.testing.expect(merkle.verifyOpening(root, i, opening.value, opening.proof));
    }
}
