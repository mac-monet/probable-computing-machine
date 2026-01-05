//! Merkle tree commitment scheme.
//!
//! DOD-compliant implementation with:
//! - Zero allocation in hot paths
//! - Reusable hasher context
//! - Configurable hash algorithms
//! - Batch operations for tree building

pub const tree = @import("tree.zig");
pub const hashers = @import("hashers.zig");

// Re-export main types
pub const MerkleTree = tree.MerkleTree;
pub const MAX_DEPTH = tree.MAX_DEPTH;

// Re-export hashers
pub const Blake3Hasher = hashers.Blake3Hasher;

test {
    _ = tree;
    _ = hashers;
}
