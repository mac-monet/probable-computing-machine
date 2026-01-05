# Merkle Tree Specification

Binary Merkle tree for polynomial commitments. Designed for DOD principles: zero allocation in hot paths, caller-controlled memory, cache-friendly layout, reusable hasher context.

## Design Principles

### Zero-Allocation Hot Paths

All commit/open/verify operations use caller-provided scratch buffers. No internal allocations.

```text
commit(evals, scratch, hasher) → root
open(evals, index, tree_scratch, hasher) → proof
verify(root, value, proof, hasher) → bool
```

### Derive, Don't Store

Direction bits are derived from leaf index at verification time. No `[]bool` storage.

```zig
fn isRight(leaf_index: u32, level: usize) bool {
    return (leaf_index >> @intCast(level)) & 1 == 1;
}
```

### Bounded Proofs

Proof paths use `BoundedArray` with compile-time max depth (32 supports 2^32 leaves). No heap allocation for proofs.

### Reusable Hasher Context

Hasher state is created once and reused across all operations. Avoids per-hash initialization overhead.

---

## Hasher Interface

The Merkle tree is generic over a hasher type. This allows Blake3 for native verification or algebraic hashes (Poseidon) for in-circuit verification.

### DOD-Friendly Design

Hashers follow these principles:

| Principle | Implementation |
|-----------|---------------|
| Reusable state | Create once with `init()`, reuse for millions of hashes |
| Write to caller buffer | `out: *Digest` parameter, no return-by-value |
| Batch operations | `hashLeavesBatch`, `hashPairsInterleaved` for tree building |
| Single-item API | `hashLeaf`, `hashPair` for verification (few hashes) |

### Required Interface

```zig
/// A hasher must provide these declarations and functions
pub const HasherInterface = struct {
    // Constants
    digest_len: comptime_int,  // output size in bytes (e.g., 32)
    Digest: type,              // [digest_len]u8

    // Lifecycle
    fn init() Self;

    // Single-item API (for verification)
    fn hashLeaf(self: *Self, data: []const u8, out: *Digest) void;
    fn hashPair(self: *Self, left: *const Digest, right: *const Digest, out: *Digest) void;

    // Batch API (for tree building)
    fn hashLeavesBatch(self: *Self, comptime field_size: usize, inputs: []const [field_size]u8, out: []Digest) void;
    fn hashPairsInterleaved(self: *Self, pairs: []const Digest, out: []Digest) void;
    fn hashPairsBatch(self: *Self, lefts: []const Digest, rights: []const Digest, out: []Digest) void;
};
```

### Why Context-Based?

Zig's `std.crypto.hash` types have large internal state:

| Hash | State Size | Init Cost |
|------|------------|-----------|
| Blake3 | ~1912 bytes | Stack allocation + setup |
| Sha256 | ~112 bytes | Stack allocation + setup |
| Keccak256 | ~208 bytes | Stack allocation + setup |

For 1M leaves → 2M hashes. Without context reuse, that's 2M × 1.9KB = 3.8GB of state churn for Blake3.

### Blake3 Implementation

```zig
pub const Blake3Hasher = struct {
    pub const digest_len = 32;
    pub const Digest = [digest_len]u8;

    /// Reusable hasher state (~1912 bytes, allocated once)
    state: std.crypto.hash.Blake3,

    /// Initialize hasher context (call once, reuse many times)
    pub fn init() Blake3Hasher {
        return .{ .state = std.crypto.hash.Blake3.init(.{}) };
    }

    // === Single-item API ===

    pub fn hashLeaf(self: *Blake3Hasher, data: []const u8, out: *Digest) void {
        self.state = std.crypto.hash.Blake3.init(.{});
        self.state.update(&[_]u8{0x00});  // leaf domain separator
        self.state.update(data);
        self.state.final(out);
    }

    pub fn hashPair(self: *Blake3Hasher, left: *const Digest, right: *const Digest, out: *Digest) void {
        self.state = std.crypto.hash.Blake3.init(.{});
        self.state.update(&[_]u8{0x01});  // internal node domain separator
        self.state.update(left);
        self.state.update(right);
        self.state.final(out);
    }

    // === Batch API ===

    pub fn hashLeavesBatch(
        self: *Blake3Hasher,
        comptime field_size: usize,
        inputs: []const [field_size]u8,
        out: []Digest,
    ) void {
        for (inputs, 0..) |data, i| {
            self.state = std.crypto.hash.Blake3.init(.{});
            self.state.update(&[_]u8{0x00});
            self.state.update(&data);
            self.state.final(&out[i]);
        }
    }

    pub fn hashPairsInterleaved(self: *Blake3Hasher, pairs: []const Digest, out: []Digest) void {
        const n_pairs = pairs.len / 2;
        for (0..n_pairs) |i| {
            self.state = std.crypto.hash.Blake3.init(.{});
            self.state.update(&[_]u8{0x01});
            self.state.update(&pairs[2 * i]);
            self.state.update(&pairs[2 * i + 1]);
            self.state.final(&out[i]);
        }
    }
};
```

### Sha256 Implementation

```zig
pub const Sha256Hasher = struct {
    pub const digest_len = 32;
    pub const Digest = [digest_len]u8;

    state: std.crypto.hash.sha2.Sha256,

    pub fn init() Sha256Hasher {
        return .{ .state = std.crypto.hash.sha2.Sha256.init(.{}) };
    }

    // Same interface as Blake3Hasher...
};
```

### Keccak256 Implementation (Ethereum-compatible)

```zig
pub const Keccak256Hasher = struct {
    pub const digest_len = 32;
    pub const Digest = [digest_len]u8;

    state: std.crypto.hash.sha3.Keccak256,

    pub fn init() Keccak256Hasher {
        return .{ .state = std.crypto.hash.sha3.Keccak256.init(.{}) };
    }

    // Same interface as Blake3Hasher...
};
```

### Poseidon (ZK-Friendly, Future)

```zig
/// For in-circuit verification where algebraic hashes are cheaper
pub const PoseidonHasher = struct {
    pub const digest_len = 32;  // field element serialized
    pub const Digest = [digest_len]u8;

    // Poseidon state (field elements, not bytes)
    state: [3]FieldElement,

    pub fn init() PoseidonHasher { ... }
    pub fn hashLeaf(...) void { ... }
    pub fn hashPair(...) void { ... }
    // ... batch methods
};
```

---

## Data Types

### Digest

Fixed-size hash output, parameterized by hasher:

```zig
pub const Digest = [Hasher.digest_len]u8;
pub const Commitment = Digest;  // root hash
```

### Proof

Authentication path with bounded capacity:

```zig
pub const MAX_DEPTH = 32;  // supports up to 2^32 leaves

pub const Proof = struct {
    /// Sibling hashes from leaf to root
    path: std.BoundedArray(Digest, MAX_DEPTH),
    /// Leaf index - direction bits derived from this
    leaf_index: u32,

    pub fn depth(self: Proof) usize {
        return self.path.len;
    }

    pub fn isRight(self: Proof, level: usize) bool {
        return (self.leaf_index >> @intCast(level)) & 1 == 1;
    }
};
```

---

## Operations

### Commit

Build tree bottom-up, return root. In-place computation using caller's scratch buffer.

```zig
/// Commit evaluations to a Merkle root.
///
/// scratch: must have length >= evals.len
/// hasher: reusable hasher context
/// Returns: root commitment
pub fn commit(evals: []const F, scratch: []Digest, hasher: *Hasher) Commitment
```

**Algorithm:**

1. Convert evaluations to bytes
2. Batch hash leaves: `hasher.hashLeavesBatch(field_size, field_bytes, scratch)`
3. Build tree in-place using `hashPairsInterleaved`:

```zig
var size = evals.len;
while (size > 1) {
    hasher.hashPairsInterleaved(scratch[0..size], scratch[0..size/2]);
    size = size / 2;
}
```

4. Return `scratch[0]`

**Memory:** O(n) scratch, reused across levels. No allocation.

### Open

Generate authentication path for a leaf.

```zig
/// Open leaf at index, returning proof.
///
/// tree_scratch: must have length >= 2*evals.len - 1
/// hasher: reusable hasher context
/// Returns: proof with path stored in BoundedArray (no allocation)
pub fn open(evals: []const F, index: u32, tree_scratch: []Digest, hasher: *Hasher) Proof
```

**Algorithm:**

1. Build full tree into `tree_scratch` using batch APIs
2. Walk from leaf to root, collecting sibling at each level
3. Return proof with path in `BoundedArray`

**Memory:** O(2n) scratch for tree. Proof itself is stack-allocated.

### Open Batch

Open multiple leaves efficiently (shared tree construction).

```zig
/// Open multiple leaves at once.
///
/// tree_scratch: must have length >= 2*evals.len - 1
/// proofs_out: must have length >= indices.len
/// hasher: reusable hasher context
pub fn openBatch(
    evals: []const F,
    indices: []const u32,
    tree_scratch: []Digest,
    proofs_out: []Proof,
    hasher: *Hasher,
) void
```

**Amortization:** Tree construction is O(n), shared across all openings.

### Verify

Verify a proof against a root commitment.

```zig
/// Verify opening proof.
/// hasher: reusable hasher context
/// Returns: true if proof is valid
pub fn verify(root: Commitment, value: F, proof: Proof, hasher: *Hasher) bool
```

**Algorithm:**

1. `hasher.hashLeaf(value.toBytes(), &current)`
2. For each level:

```zig
if (proof.isRight(level)) {
    hasher.hashPair(&sibling, &current, &current);
} else {
    hasher.hashPair(&current, &sibling, &current);
}
```

3. Return `current == root`

**Memory:** O(1), single `Digest` on stack.

---

## Tree Layout

Level-by-level storage for cache-friendly access:

```text
[leaf₀][leaf₁][leaf₂][leaf₃][node₀₁][node₂₃][root]
 \_________leaves_________/ \____internal nodes___/
         level 0                  levels 1..log(n)
```

Indexing:

- Leaves: `tree[0..n]`
- Level k nodes: `tree[level_start(k)..level_start(k+1)]`
- Root: `tree[2n-2]`

---

## Domain Separation

Following RFC 6962 (Certificate Transparency):

| Node Type | Domain Separator |
|-----------|------------------|
| Leaf | `0x00` |
| Internal | `0x01` |

This prevents second-preimage attacks where an internal node could be reinterpreted as a leaf.

---

## Usage Examples

### Basic Commit and Verify

```zig
const Merkle = MerkleTree(M31, Blake3Hasher);

// Create hasher context (reuse for all operations)
var hasher = Blake3Hasher.init();

// Commit (stack scratch for small trees)
var scratch: [1024]Merkle.Digest = undefined;
const evals: [1024]M31 = ...;
const root = Merkle.commit(&evals, &scratch, &hasher);

// Open (larger scratch for full tree)
var tree_scratch: [2047]Merkle.Digest = undefined;
const proof = Merkle.open(&evals, 42, &tree_scratch, &hasher);

// Verify
const valid = Merkle.verify(root, evals[42], proof, &hasher);
```

### Batch Opening with Arena

```zig
var arena = std.heap.ArenaAllocator.init(allocator);
defer arena.deinit();

var hasher = Blake3Hasher.init();  // On stack, reused
const tree_scratch = try arena.allocator().alloc(Merkle.Digest, 2 * n - 1);
var proofs: [num_queries]Merkle.Proof = undefined;

Merkle.openBatch(&evals, &query_indices, tree_scratch, &proofs, &hasher);
```

### Swapping Hash Algorithms

```zig
// For native verification (fast)
const NativeMerkle = MerkleTree(M31, Blake3Hasher);

// For Ethereum compatibility
const EthMerkle = MerkleTree(M31, Keccak256Hasher);

// For in-circuit verification (when we implement Poseidon)
const ZkMerkle = MerkleTree(M31, PoseidonHasher);
```

---

## DOD Comparison

| Aspect | Old Implementation | New Design |
|--------|-------------------|------------|
| Hasher state | New per hash (~1.9KB × 2M) | Single context, reused |
| Allocation | Hardcoded `page_allocator` | Caller-provided scratch |
| Direction bits | `[]bool` (8x waste) | Derived from `leaf_index` |
| Proof allocation | Two heap allocations | `BoundedArray` on stack |
| Batch support | None | `hashLeavesBatch`, `hashPairsInterleaved` |
| Hash algorithm | Blake3 only | Generic over hasher |
| Output style | Return by value (32 bytes) | Write to caller buffer |

### Performance Impact

For 1M leaf tree (2M hashes):

| Metric | Pure Function | Context + Batch |
|--------|---------------|-----------------|
| State allocations | 2,000,000 | 1 |
| Stack churn | ~3.8 GB (Blake3) | ~1.9 KB |
| Cache behavior | Cold each hash | Hot state |

---

## Future: SIMD Parallelization

The batch interface enables future SIMD optimization:

```zig
/// 4-way parallel Blake3 for AVX2 (future)
pub fn hashPairs4(
    lefts: *const [4]Digest,
    rights: *const [4]Digest,
    outs: *[4]Digest,
) void {
    // Process 4 pairs simultaneously using SIMD
}
```

The current sequential batch implementation can be replaced with SIMD without changing the Merkle tree code.

---

## References

- RFC 6962: Certificate Transparency (domain separation)
- [Zig std.crypto.hash](https://github.com/ziglang/zig/blob/master/lib/std/crypto.zig)
- Data-Oriented Design principles
