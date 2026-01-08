//! Merkle tree commitment scheme.
//!
//! DOD-compliant implementation with:
//! - Zero allocation in hot paths
//! - Reusable hasher context
//! - Configurable hash algorithms
//! - Batch operations for tree building
//! - Comptime max_depth for right-sized proofs

pub const tree = @import("tree.zig");
pub const hashers = @import("hashers.zig");

// Re-export main types
pub const MerkleTree = tree.MerkleTree;

// Re-export hashers
pub const Blake3Hasher = hashers.Blake3Hasher;

test {
    _ = tree;
    _ = hashers;
}
