//! Merkle tree hashers with DOD-friendly interfaces.
//!
//! Design principles:
//! - Reusable context (no per-hash allocations)
//! - Batch operations for tree building
//! - Write to caller's buffer (no return-by-value for digests)
//! - Simple API for verification (few hashes)

const std = @import("std");

// ============================================================================
// Blake3 Hasher
// ============================================================================

pub const Blake3Hasher = struct {
    pub const digest_len = 32;
    pub const Digest = [digest_len]u8;

    /// Reusable hasher state (~1912 bytes, allocated once)
    state: std.crypto.hash.Blake3,

    /// Initialize hasher context (call once, reuse many times)
    pub fn init() Blake3Hasher {
        return .{ .state = std.crypto.hash.Blake3.init(.{}) };
    }

    // ========================================================================
    // Single-item API (for verification, small operations)
    // ========================================================================

    /// Hash a leaf, write to caller's buffer
    pub fn hashLeaf(self: *Blake3Hasher, data: []const u8, out: *Digest) void {
        self.state = std.crypto.hash.Blake3.init(.{});
        self.state.update(&[_]u8{0x00}); // leaf domain separator
        self.state.update(data);
        self.state.final(out);
    }

    /// Hash a pair, write to caller's buffer
    pub fn hashPair(self: *Blake3Hasher, left: *const Digest, right: *const Digest, out: *Digest) void {
        self.state = std.crypto.hash.Blake3.init(.{});
        self.state.update(&[_]u8{0x01}); // internal node domain separator
        self.state.update(left);
        self.state.update(right);
        self.state.final(out);
    }

    // ========================================================================
    // Batch API (for tree building, millions of hashes)
    // ========================================================================

    /// Hash multiple leaves in one pass
    /// Reuses hasher state across all items
    pub fn hashLeavesBatch(
        self: *Blake3Hasher,
        comptime field_size: usize,
        field_bytes: []const [field_size]u8,
        out: []Digest,
    ) void {
        std.debug.assert(out.len >= field_bytes.len);
        for (field_bytes, 0..) |data, i| {
            self.state = std.crypto.hash.Blake3.init(.{});
            self.state.update(&[_]u8{0x00});
            self.state.update(&data);
            self.state.final(&out[i]);
        }
    }

    /// Hash pairs for one tree level (in-place capable)
    /// lefts and rights are interleaved in `pairs`: [L0, R0, L1, R1, ...]
    pub fn hashPairsInterleaved(
        self: *Blake3Hasher,
        pairs: []const Digest,
        out: []Digest,
    ) void {
        const n_pairs = pairs.len / 2;
        std.debug.assert(out.len >= n_pairs);
        std.debug.assert(pairs.len % 2 == 0);

        for (0..n_pairs) |i| {
            self.state = std.crypto.hash.Blake3.init(.{});
            self.state.update(&[_]u8{0x01});
            self.state.update(&pairs[2 * i]);
            self.state.update(&pairs[2 * i + 1]);
            self.state.final(&out[i]);
        }
    }

    /// Hash pairs from separate left/right arrays
    pub fn hashPairsBatch(
        self: *Blake3Hasher,
        lefts: []const Digest,
        rights: []const Digest,
        out: []Digest,
    ) void {
        std.debug.assert(lefts.len == rights.len);
        std.debug.assert(out.len >= lefts.len);

        for (lefts, rights, 0..) |left, right, i| {
            self.state = std.crypto.hash.Blake3.init(.{});
            self.state.update(&[_]u8{0x01});
            self.state.update(&left);
            self.state.update(&right);
            self.state.final(&out[i]);
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "blake3: single hash operations" {
    var hasher = Blake3Hasher.init();

    var leaf: Blake3Hasher.Digest = undefined;
    hasher.hashLeaf("hello", &leaf);

    var pair: Blake3Hasher.Digest = undefined;
    hasher.hashPair(&leaf, &leaf, &pair);

    try std.testing.expect(!std.mem.eql(u8, &leaf, &pair));
}

test "blake3: batch leaf hashing" {
    var hasher = Blake3Hasher.init();

    const inputs = [_][4]u8{
        .{ 1, 0, 0, 0 },
        .{ 2, 0, 0, 0 },
        .{ 3, 0, 0, 0 },
        .{ 4, 0, 0, 0 },
    };

    var outputs: [4]Blake3Hasher.Digest = undefined;
    hasher.hashLeavesBatch(4, &inputs, &outputs);

    // All outputs should be different
    try std.testing.expect(!std.mem.eql(u8, &outputs[0], &outputs[1]));
    try std.testing.expect(!std.mem.eql(u8, &outputs[1], &outputs[2]));
    try std.testing.expect(!std.mem.eql(u8, &outputs[2], &outputs[3]));

    // Verify consistency with single-item API
    var single: Blake3Hasher.Digest = undefined;
    hasher.hashLeaf(&inputs[0], &single);
    try std.testing.expectEqualSlices(u8, &single, &outputs[0]);
}

test "blake3: batch pair hashing interleaved" {
    var hasher = Blake3Hasher.init();

    // First hash some leaves
    const inputs = [_][4]u8{
        .{ 1, 0, 0, 0 },
        .{ 2, 0, 0, 0 },
        .{ 3, 0, 0, 0 },
        .{ 4, 0, 0, 0 },
    };
    var leaves: [4]Blake3Hasher.Digest = undefined;
    hasher.hashLeavesBatch(4, &inputs, &leaves);

    // Hash pairs: (leaf0, leaf1), (leaf2, leaf3)
    var pairs: [2]Blake3Hasher.Digest = undefined;
    hasher.hashPairsInterleaved(&leaves, &pairs);

    // Verify against single-item API
    var expected: Blake3Hasher.Digest = undefined;
    hasher.hashPair(&leaves[0], &leaves[1], &expected);
    try std.testing.expectEqualSlices(u8, &expected, &pairs[0]);

    hasher.hashPair(&leaves[2], &leaves[3], &expected);
    try std.testing.expectEqualSlices(u8, &expected, &pairs[1]);
}

test "blake3: batch pair hashing separate arrays" {
    var hasher = Blake3Hasher.init();

    const lefts = [_]Blake3Hasher.Digest{
        .{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        .{ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    };
    const rights = [_]Blake3Hasher.Digest{
        .{ 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        .{ 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    };

    var outputs: [2]Blake3Hasher.Digest = undefined;
    hasher.hashPairsBatch(&lefts, &rights, &outputs);

    // Verify against single-item API
    var expected: Blake3Hasher.Digest = undefined;
    hasher.hashPair(&lefts[0], &rights[0], &expected);
    try std.testing.expectEqualSlices(u8, &expected, &outputs[0]);
}

test "blake3: domain separation leaf vs pair" {
    var hasher = Blake3Hasher.init();

    // Hash 64 bytes as a "leaf"
    const data: [64]u8 = [_]u8{0} ** 64;
    var leaf_hash: Blake3Hasher.Digest = undefined;
    hasher.hashLeaf(&data, &leaf_hash);

    // Hash same bytes as two 32-byte "digests" pair
    var pair_hash: Blake3Hasher.Digest = undefined;
    const left: *const Blake3Hasher.Digest = @ptrCast(data[0..32]);
    const right: *const Blake3Hasher.Digest = @ptrCast(data[32..64]);
    hasher.hashPair(left, right, &pair_hash);

    // Must be different due to domain separation (0x00 vs 0x01 prefix)
    try std.testing.expect(!std.mem.eql(u8, &leaf_hash, &pair_hash));
}

test "blake3: empty leaf hash" {
    var hasher = Blake3Hasher.init();

    var hash1: Blake3Hasher.Digest = undefined;
    var hash2: Blake3Hasher.Digest = undefined;

    hasher.hashLeaf(&[_]u8{}, &hash1);
    hasher.hashLeaf(&[_]u8{}, &hash2);

    // Deterministic
    try std.testing.expectEqualSlices(u8, &hash1, &hash2);

    // Not all zeros (hash of empty is still a real hash)
    try std.testing.expect(!std.mem.eql(u8, &hash1, &std.mem.zeroes(Blake3Hasher.Digest)));
}

test "blake3: hashPairsInterleaved in-place safe" {
    var hasher = Blake3Hasher.init();

    // Create 4 leaves
    var data: [4]Blake3Hasher.Digest = undefined;
    for (0..4) |i| {
        hasher.hashLeaf(&[_]u8{@intCast(i)}, &data[i]);
    }

    // Copy for comparison
    var expected: [2]Blake3Hasher.Digest = undefined;
    hasher.hashPair(&data[0], &data[1], &expected[0]);
    hasher.hashPair(&data[2], &data[3], &expected[1]);

    // In-place: write to first half while reading from all 4
    hasher.hashPairsInterleaved(&data, data[0..2]);

    try std.testing.expectEqualSlices(u8, &expected[0], &data[0]);
    try std.testing.expectEqualSlices(u8, &expected[1], &data[1]);
}
