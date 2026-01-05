//! DOD-compliant Merkle tree for polynomial commitments.
//!
//! Design principles:
//! - Zero allocation in hot paths (caller-provided scratch)
//! - Reusable hasher context
//! - Bounded proofs (fixed buffer, no heap)
//! - Direction bits derived from leaf index

const std = @import("std");

/// Maximum tree depth (supports up to 2^32 leaves)
pub const MAX_DEPTH = 32;

/// Create a Merkle tree type parameterized by field and hasher.
pub fn MerkleTree(comptime F: type, comptime Hasher: type) type {
    return struct {
        const Self = @This();

        pub const Digest = Hasher.Digest;
        pub const Commitment = Digest;

        /// Authentication path with bounded capacity (no heap allocation)
        pub const Proof = struct {
            /// Sibling hashes from leaf to root (fixed buffer)
            path_buffer: [MAX_DEPTH]Digest,
            /// Number of valid entries in path
            path_len: usize,
            /// Leaf index - direction bits derived from this
            leaf_index: u32,

            pub fn init(leaf_index: u32) Proof {
                return .{
                    .path_buffer = undefined,
                    .path_len = 0,
                    .leaf_index = leaf_index,
                };
            }

            pub fn depth(self: Proof) usize {
                return self.path_len;
            }

            pub fn path(self: *const Proof) []const Digest {
                return self.path_buffer[0..self.path_len];
            }

            pub fn append(self: *Proof, digest: Digest) void {
                std.debug.assert(self.path_len < MAX_DEPTH);
                self.path_buffer[self.path_len] = digest;
                self.path_len += 1;
            }

            /// Returns true if current node is right child at given level
            pub fn isRight(self: Proof, level: usize) bool {
                return (self.leaf_index >> @intCast(level)) & 1 == 1;
            }
        };

        // ====================================================================
        // Commit
        // ====================================================================

        /// Commit evaluations to a Merkle root.
        ///
        /// Uses batch hashing for efficiency. Builds tree in-place in scratch buffer.
        ///
        /// Requirements:
        /// - scratch.len >= evals.len
        /// - evals.len must be power of 2
        ///
        /// Returns: root commitment
        pub fn commit(evals: []const F, scratch: []Digest, hasher: *Hasher) Commitment {
            const n = evals.len;
            if (n == 0) return std.mem.zeroes(Digest);

            std.debug.assert(scratch.len >= n);
            std.debug.assert(std.math.isPowerOfTwo(n));

            // Hash leaves using batch API
            hashLeaves(evals, scratch[0..n], hasher);

            // Build tree in-place
            var size = n;
            while (size > 1) {
                const half = size / 2;
                hasher.hashPairsInterleaved(scratch[0..size], scratch[0..half]);
                size = half;
            }

            return scratch[0];
        }

        // ====================================================================
        // Open
        // ====================================================================

        /// Open leaf at index, returning proof.
        ///
        /// Requirements:
        /// - tree_scratch.len >= 2 * evals.len - 1
        /// - evals.len must be power of 2
        /// - index < evals.len
        ///
        /// Returns: proof with path stored in BoundedArray (no allocation)
        pub fn open(evals: []const F, index: u32, tree_scratch: []Digest, hasher: *Hasher) Proof {
            const n = evals.len;
            std.debug.assert(tree_scratch.len >= 2 * n - 1);
            std.debug.assert(std.math.isPowerOfTwo(n));
            std.debug.assert(index < n);

            const depth = std.math.log2_int(usize, n);

            // Build full tree
            buildTree(evals, tree_scratch, hasher);

            // Extract authentication path
            var proof = Proof.init(index);

            var idx: usize = index;
            var level_start: usize = 0;
            var level_size: usize = n;

            for (0..depth) |_| {
                const sibling_idx = idx ^ 1;
                proof.append(tree_scratch[level_start + sibling_idx]);
                idx >>= 1;
                level_start += level_size;
                level_size >>= 1;
            }

            return proof;
        }

        // ====================================================================
        // Open Batch
        // ====================================================================

        /// Open multiple leaves at once (shared tree construction).
        ///
        /// Requirements:
        /// - tree_scratch.len >= 2 * evals.len - 1
        /// - proofs_out.len >= indices.len
        /// - all indices < evals.len
        pub fn openBatch(
            evals: []const F,
            indices: []const u32,
            tree_scratch: []Digest,
            proofs_out: []Proof,
            hasher: *Hasher,
        ) void {
            const n = evals.len;
            std.debug.assert(tree_scratch.len >= 2 * n - 1);
            std.debug.assert(proofs_out.len >= indices.len);
            std.debug.assert(std.math.isPowerOfTwo(n));

            const depth = std.math.log2_int(usize, n);

            // Build tree once
            buildTree(evals, tree_scratch, hasher);

            // Extract all paths
            for (indices, 0..) |index, i| {
                std.debug.assert(index < n);
                proofs_out[i] = extractPath(tree_scratch, n, depth, index);
            }
        }

        // ====================================================================
        // Verify
        // ====================================================================

        /// Verify opening proof against root commitment.
        ///
        /// Returns: true if proof is valid
        pub fn verify(root: Commitment, value: F, proof: Proof, hasher: *Hasher) bool {
            var current: Digest = undefined;
            hasher.hashLeaf(&value.toBytes(), &current);

            for (proof.path(), 0..) |sibling, level| {
                if (proof.isRight(level)) {
                    hasher.hashPair(&sibling, &current, &current);
                } else {
                    hasher.hashPair(&current, &sibling, &current);
                }
            }

            return std.mem.eql(u8, &current, &root);
        }

        // ====================================================================
        // Internals
        // ====================================================================

        /// Hash all leaves using batch API
        fn hashLeaves(evals: []const F, out: []Digest, hasher: *Hasher) void {
            // Convert field elements to bytes and hash
            // Using single-item API since we need field.toBytes() per element
            for (evals, 0..) |e, i| {
                hasher.hashLeaf(&e.toBytes(), &out[i]);
            }
        }

        /// Build complete Merkle tree into scratch buffer.
        /// Layout: [leaves][level1][level2]...[root]
        fn buildTree(evals: []const F, tree: []Digest, hasher: *Hasher) void {
            const n = evals.len;

            // Hash leaves
            hashLeaves(evals, tree[0..n], hasher);

            // Build internal nodes level by level
            var level_start: usize = 0;
            var level_size: usize = n;
            var write_pos: usize = n;

            while (level_size > 1) {
                const half = level_size / 2;
                hasher.hashPairsInterleaved(
                    tree[level_start .. level_start + level_size],
                    tree[write_pos .. write_pos + half],
                );
                level_start += level_size;
                level_size = half;
                write_pos += half;
            }
        }

        /// Extract authentication path from pre-built tree
        fn extractPath(tree: []const Digest, n: usize, depth: usize, index: u32) Proof {
            var proof = Proof.init(index);

            var idx: usize = index;
            var level_start: usize = 0;
            var level_size: usize = n;

            for (0..depth) |_| {
                const sibling_idx = idx ^ 1;
                proof.append(tree[level_start + sibling_idx]);
                idx >>= 1;
                level_start += level_size;
                level_size >>= 1;
            }

            return proof;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const M31 = @import("../fields/mersenne31.zig").Mersenne31;
const Blake3Hasher = @import("hashers.zig").Blake3Hasher;

test "commit and verify single leaf" {
    var hasher = Blake3Hasher.init();
    const Merkle = MerkleTree(M31, Blake3Hasher);

    const evals = [_]M31{M31.fromU64(42)};
    var scratch: [1]Merkle.Digest = undefined;

    const root = Merkle.commit(&evals, &scratch, &hasher);

    // For single leaf, root should be hash of that leaf
    var expected: Merkle.Digest = undefined;
    hasher.hashLeaf(&evals[0].toBytes(), &expected);
    try std.testing.expectEqualSlices(u8, &expected, &root);
}

test "commit deterministic" {
    var hasher = Blake3Hasher.init();
    const Merkle = MerkleTree(M31, Blake3Hasher);

    const evals = [_]M31{
        M31.fromU64(1),
        M31.fromU64(2),
        M31.fromU64(3),
        M31.fromU64(4),
    };

    var scratch1: [4]Merkle.Digest = undefined;
    var scratch2: [4]Merkle.Digest = undefined;

    const root1 = Merkle.commit(&evals, &scratch1, &hasher);
    const root2 = Merkle.commit(&evals, &scratch2, &hasher);

    try std.testing.expectEqualSlices(u8, &root1, &root2);
}

test "open and verify all leaves" {
    var hasher = Blake3Hasher.init();
    const Merkle = MerkleTree(M31, Blake3Hasher);

    const evals = [_]M31{
        M31.fromU64(1),
        M31.fromU64(2),
        M31.fromU64(3),
        M31.fromU64(4),
    };

    var scratch: [4]Merkle.Digest = undefined;
    const root = Merkle.commit(&evals, &scratch, &hasher);

    var tree_scratch: [7]Merkle.Digest = undefined; // 2*4 - 1

    for (0..evals.len) |i| {
        const proof = Merkle.open(&evals, @intCast(i), &tree_scratch, &hasher);

        try std.testing.expect(proof.leaf_index == i);
        try std.testing.expect(proof.depth() == 2); // log2(4) = 2
        try std.testing.expect(Merkle.verify(root, evals[i], proof, &hasher));
    }
}

test "open batch" {
    var hasher = Blake3Hasher.init();
    const Merkle = MerkleTree(M31, Blake3Hasher);

    const evals = [_]M31{
        M31.fromU64(1),
        M31.fromU64(2),
        M31.fromU64(3),
        M31.fromU64(4),
        M31.fromU64(5),
        M31.fromU64(6),
        M31.fromU64(7),
        M31.fromU64(8),
    };

    var scratch: [8]Merkle.Digest = undefined;
    const root = Merkle.commit(&evals, &scratch, &hasher);

    var tree_scratch: [15]Merkle.Digest = undefined; // 2*8 - 1
    const indices = [_]u32{ 0, 3, 5, 7 };
    var proofs: [4]Merkle.Proof = undefined;

    Merkle.openBatch(&evals, &indices, &tree_scratch, &proofs, &hasher);

    for (indices, 0..) |idx, i| {
        try std.testing.expect(proofs[i].leaf_index == idx);
        try std.testing.expect(Merkle.verify(root, evals[idx], proofs[i], &hasher));
    }
}

test "verify rejects wrong value" {
    var hasher = Blake3Hasher.init();
    const Merkle = MerkleTree(M31, Blake3Hasher);

    const evals = [_]M31{
        M31.fromU64(1),
        M31.fromU64(2),
        M31.fromU64(3),
        M31.fromU64(4),
    };

    var scratch: [4]Merkle.Digest = undefined;
    const root = Merkle.commit(&evals, &scratch, &hasher);

    var tree_scratch: [7]Merkle.Digest = undefined;
    const proof = Merkle.open(&evals, 0, &tree_scratch, &hasher);

    // Correct value verifies
    try std.testing.expect(Merkle.verify(root, evals[0], proof, &hasher));

    // Wrong value fails
    try std.testing.expect(!Merkle.verify(root, M31.fromU64(999), proof, &hasher));
}

test "verify rejects wrong proof" {
    var hasher = Blake3Hasher.init();
    const Merkle = MerkleTree(M31, Blake3Hasher);

    const evals = [_]M31{
        M31.fromU64(1),
        M31.fromU64(2),
        M31.fromU64(3),
        M31.fromU64(4),
    };

    var scratch: [4]Merkle.Digest = undefined;
    const root = Merkle.commit(&evals, &scratch, &hasher);

    var tree_scratch: [7]Merkle.Digest = undefined;
    const proof0 = Merkle.open(&evals, 0, &tree_scratch, &hasher);
    const proof1 = Merkle.open(&evals, 1, &tree_scratch, &hasher);

    // Correct proof verifies
    try std.testing.expect(Merkle.verify(root, evals[0], proof0, &hasher));

    // Wrong proof (proof for index 1 used with value at index 0) fails
    try std.testing.expect(!Merkle.verify(root, evals[0], proof1, &hasher));
}

test "large tree" {
    var hasher = Blake3Hasher.init();
    const Merkle = MerkleTree(M31, Blake3Hasher);

    const N = 1024;
    var evals: [N]M31 = undefined;
    for (0..N) |i| {
        evals[i] = M31.fromU64(i);
    }

    var scratch: [N]Merkle.Digest = undefined;
    const root = Merkle.commit(&evals, &scratch, &hasher);

    var tree_scratch: [2 * N - 1]Merkle.Digest = undefined;

    // Test a few random indices
    const test_indices = [_]u32{ 0, 1, 512, 1023 };
    for (test_indices) |idx| {
        const proof = Merkle.open(&evals, idx, &tree_scratch, &hasher);
        try std.testing.expect(proof.depth() == 10); // log2(1024)
        try std.testing.expect(Merkle.verify(root, evals[idx], proof, &hasher));
    }
}

test "proof direction bits" {
    const Merkle = MerkleTree(M31, Blake3Hasher);

    // Index 5 = 0b101
    // Level 0: bit 0 = 1 (right)
    // Level 1: bit 1 = 0 (left)
    // Level 2: bit 2 = 1 (right)
    const proof = Merkle.Proof.init(5);

    try std.testing.expect(proof.isRight(0) == true);
    try std.testing.expect(proof.isRight(1) == false);
    try std.testing.expect(proof.isRight(2) == true);
}

test "empty tree returns zero root" {
    var hasher = Blake3Hasher.init();
    const Merkle = MerkleTree(M31, Blake3Hasher);

    const evals: []const M31 = &.{};
    var scratch: [0]Merkle.Digest = undefined;

    const root = Merkle.commit(evals, &scratch, &hasher);
    try std.testing.expectEqual(std.mem.zeroes(Merkle.Digest), root);
}

test "two leaf tree" {
    var hasher = Blake3Hasher.init();
    const Merkle = MerkleTree(M31, Blake3Hasher);

    const evals = [_]M31{ M31.fromU64(1), M31.fromU64(2) };
    var scratch: [2]Merkle.Digest = undefined;
    const root = Merkle.commit(&evals, &scratch, &hasher);

    var tree_scratch: [3]Merkle.Digest = undefined;
    for (0..2) |i| {
        const proof = Merkle.open(&evals, @intCast(i), &tree_scratch, &hasher);
        try std.testing.expect(proof.depth() == 1);
        try std.testing.expect(Merkle.verify(root, evals[i], proof, &hasher));
    }
}

test "commit root matches open tree root" {
    var hasher = Blake3Hasher.init();
    const Merkle = MerkleTree(M31, Blake3Hasher);

    const evals = [_]M31{ M31.fromU64(1), M31.fromU64(2), M31.fromU64(3), M31.fromU64(4) };

    var commit_scratch: [4]Merkle.Digest = undefined;
    const commit_root = Merkle.commit(&evals, &commit_scratch, &hasher);

    // Build full tree and extract root (at index 6: 4 leaves + 2 + 1 - 1)
    var tree_scratch: [7]Merkle.Digest = undefined;
    _ = Merkle.open(&evals, 0, &tree_scratch, &hasher);
    const tree_root = tree_scratch[6];

    try std.testing.expectEqualSlices(u8, &commit_root, &tree_root);
}

test "last leaf index" {
    var hasher = Blake3Hasher.init();
    const Merkle = MerkleTree(M31, Blake3Hasher);

    const N = 16;
    var evals: [N]M31 = undefined;
    for (0..N) |i| evals[i] = M31.fromU64(i);

    var scratch: [N]Merkle.Digest = undefined;
    const root = Merkle.commit(&evals, &scratch, &hasher);

    var tree_scratch: [2 * N - 1]Merkle.Digest = undefined;
    const proof = Merkle.open(&evals, N - 1, &tree_scratch, &hasher);

    try std.testing.expect(proof.leaf_index == N - 1);
    try std.testing.expect(Merkle.verify(root, evals[N - 1], proof, &hasher));
}

test "hasher reuse across multiple trees" {
    var hasher = Blake3Hasher.init();
    const Merkle = MerkleTree(M31, Blake3Hasher);

    // First tree
    const evals1 = [_]M31{ M31.fromU64(1), M31.fromU64(2) };
    var scratch1: [2]Merkle.Digest = undefined;
    const root1 = Merkle.commit(&evals1, &scratch1, &hasher);

    // Second tree (different data)
    const evals2 = [_]M31{ M31.fromU64(3), M31.fromU64(4) };
    var scratch2: [2]Merkle.Digest = undefined;
    const root2 = Merkle.commit(&evals2, &scratch2, &hasher);

    // Roots should differ
    try std.testing.expect(!std.mem.eql(u8, &root1, &root2));

    // Re-compute first tree, should match
    const root1_again = Merkle.commit(&evals1, &scratch1, &hasher);
    try std.testing.expectEqualSlices(u8, &root1, &root1_again);
}
