# Fused Sumcheck Optimization

## Overview

The current sumcheck implementation uses **2 memory passes per round**:
1. `sumHalves()` — read entire array to compute sums
2. `bind()` — read entire array again to apply challenge

With **commit-then-challenge**, we can fuse these into **1 memory pass per round**, yielding ~33-50% speedup on the prover.

## Current Implementation

```zig
// src/sumcheck.zig - current structure
for (rounds) |*round| {
    const halves = multilinear.sumHalves(F, current);  // Pass 1: read N
    round.* = .{ .eval_0 = halves[0], .eval_1 = halves[1] };

    transcript.absorb(halves[0]);
    transcript.absorb(halves[1]);
    const c = transcript.squeeze();  // Challenge depends on round poly

    current = multilinear.bind(F, current, c);  // Pass 2: read N, write N/2
}
```

**Problem**: Challenge `c` is derived AFTER `sumHalves`, so we can't fuse.

## Commit-Then-Challenge Model

```zig
// New structure
commitment = commit(evals);  // Merkle root, Brakedown, etc.
transcript.absorb(commitment);
challenges = transcript.squeezeN(num_vars);  // ALL challenges upfront

for (challenges, rounds) |c, *round| {
    // Fused: single pass computes sums AND applies bind
    const result = multilinear.sumHalvesAndBind(F, current, c);
    round.* = .{ .eval_0 = result.sums[0], .eval_1 = result.sums[1] };
    current = result.bound;
}
```

## Implementation Steps

### Step 1a: Add `sumAndLinearCombineBatch` to mersenne31.zig

Keep all SIMD isolated in the field implementation:

**Aliasing note**: `dst` and `a` point to the same memory (`evals[0..half]`). This is safe
because we read `a[i]` before writing `dst[i]`. If we wrote first, subsequent reads of `a[i]`
would return the new value, not the original. The load-before-store order preserves zero-copy
in-place updates.

```zig
/// Fused sum + linear combine: accumulates sums while computing dst[i] = a[i] + r * (b[i] - a[i])
/// Returns [sum_a, sum_b] for the round polynomial.
/// All SIMD logic lives here, keeping multilinear.zig generic.
///
/// IMPORTANT: dst and a may alias (point to same memory). This is safe because
/// each element is read before being written. Do not reorder loads/stores.
pub fn sumAndLinearCombineBatch(
    dst: []Mersenne31,         // output, may alias a
    a: []const Mersenne31,     // first half (read before write)
    b: []const Mersenne31,     // second half
    r: Mersenne31,
) [2]Mersenne31 {
    std.debug.assert(dst.len == a.len and a.len == b.len);

    const simd = @import("../simd.zig");
    const VEC_LEN = simd.u32_len orelse 4;
    const VecU64 = @Vector(VEC_LEN, u64);

    const len = dst.len;
    const r_val: u64 = r.value;

    // Vector accumulators - horizontal sum only at the end
    var acc_a: VecU64 = @splat(0);
    var acc_b: VecU64 = @splat(0);

    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const result = sumAndLinearCombineChunk(VEC_LEN,
            dst[i..][0..VEC_LEN],
            a[i..][0..VEC_LEN],
            b[i..][0..VEC_LEN],
            r_val);
        acc_a +%= result.sum_a;
        acc_b +%= result.sum_b;
    }

    // Horizontal sum of vector accumulators
    var scalar_sum_a: u64 = @reduce(.Add, acc_a);
    var scalar_sum_b: u64 = @reduce(.Add, acc_b);

    // Tail loop: remaining elements
    while (i < len) : (i += 1) {
        scalar_sum_a +%= a[i].value;
        scalar_sum_b +%= b[i].value;
        const diff: u64 = @as(u64, b[i].value) +% MODULUS -% a[i].value;
        const prod = r_val *% diff;
        dst[i] = reduce64(@as(u64, a[i].value) +% prod);
    }

    return .{ reduce64(scalar_sum_a), reduce64(scalar_sum_b) };
}

/// Process one chunk: accumulate sums AND compute linear combination.
/// Load-before-store order is critical for aliasing safety.
inline fn sumAndLinearCombineChunk(
    comptime VEC_LEN: comptime_int,
    dst: *[VEC_LEN]Mersenne31,
    a: *const [VEC_LEN]Mersenne31,
    b: *const [VEC_LEN]Mersenne31,
    r_val: u64,
) struct { sum_a: @Vector(VEC_LEN, u64), sum_b: @Vector(VEC_LEN, u64) } {
    const VecU32 = @Vector(VEC_LEN, u32);
    const VecU64 = @Vector(VEC_LEN, u64);

    // LOAD FIRST: read a and b before any writes (dst may alias a)
    const a_vec: VecU32 = @bitCast(a.*);
    const b_vec: VecU32 = @bitCast(b.*);

    // Widen for accumulation (no reduction needed yet)
    const a_wide: VecU64 = a_vec;
    const b_wide: VecU64 = b_vec;

    // Compute diff and bind: dst = a + r * (b - a)
    const modulus_vec: VecU32 = @splat(MODULUS);
    const diff_vec: VecU32 = b_vec +% modulus_vec -% a_vec;
    const diff_wide: VecU64 = diff_vec;
    const r_vec: VecU64 = @splat(r_val);
    const prod: VecU64 = r_vec *% diff_wide;
    const result: VecU64 = a_wide +% prod;

    // STORE AFTER: write only after all reads complete
    dst.* = @bitCast(reduce64Vec(VEC_LEN, result));

    // Return original values (captured before the write)
    return .{ .sum_a = a_wide, .sum_b = b_wide };
}
```

### Step 1b: Add `sumHalvesAndBind` to multilinear.zig

The multilinear layer stays SIMD-free, delegating to the field:

```zig
/// Fused operation: compute round polynomial AND bind in single pass.
/// Returns sums for round polynomial and the bound slice.
pub fn sumHalvesAndBind(comptime F: type, evals: []F, r: F) struct { sums: [2]F, bound: []F } {
    std.debug.assert(evals.len >= 2);
    const half = evals.len / 2;

    // Delegate to field's fused batch operation (SIMD lives there)
    const sums = F.sumAndLinearCombineBatch(evals[0..half], evals[0..half], evals[half..], r);

    return .{ .sums = sums, .bound = evals[0..half] };
}
```

This mirrors the existing pattern: `sumHalves` calls `F.sumSlices`, `bind` calls `F.linearCombineBatch`.

### Step 2: Add commitment interface

```zig
// src/commit.zig
pub fn Commitment(comptime F: type) type {
    return struct {
        root: [32]u8,  // Merkle root or similar

        /// Commit to polynomial evaluations
        pub fn commit(evals: []const F) @This() {
            // Simple: hash all evaluations
            // Better: Merkle tree of evaluations
            // Best: Brakedown encoding + Merkle
        }
    };
}
```

### Step 3: New sumcheck prover API

```zig
/// Sumcheck with pre-derived challenges (commit-then-challenge model).
/// More efficient: enables fused memory access.
pub fn proveWithChallenges(
    evals: []F,
    challenges: []const F,
    rounds: []RoundPoly
) F {
    var current = evals;

    for (challenges, rounds) |c, *round| {
        const result = multilinear.sumHalvesAndBind(F, current, c);
        round.* = .{ .eval_0 = result.sums[0], .eval_1 = result.sums[1] };
        current = result.bound;
    }

    return current[0];
}
```

### Step 4: Update benchmarks

Add to `bench/sumcheck.zig`:

```zig
// Benchmark fused vs non-fused
try bench.benchmarkMem("n=20 prove (fused)", ...);
try bench.benchmarkMem("n=20 prove (standard)", ...);
```

## Expected Performance

### Memory Operations Per Round

| Operation | Standard | Fused |
|-----------|----------|-------|
| sumHalves | Read N | — |
| bind | Read N, Write N/2 | — |
| sumHalvesAndBind | — | Read N, Write N/2 |
| **Total** | Read 2N, Write N/2 | Read N, Write N/2 |

### Projected Speedup

For memory-bound workloads:
- Reads reduced by 50% per round
- Overall ~33% faster prover (writes unchanged)

Current n=24 prove: ~12.7 ms
Expected n=24 prove: ~8-9 ms

## Post-Fusion Optimizations

After implementing the fused algorithm, these additional single-threaded optimizations can be layered on:

### 1. Software Prefetching (Medium Impact)

The fused operation reads from two locations 32MB apart (for n=24). Prefetch both streams to hide memory latency:

```zig
const PREFETCH_DISTANCE = 16;  // Cache lines ahead (~1KB)

while (i + VEC_LEN <= len) : (i += VEC_LEN) {
    // Prefetch future data from both halves
    if (i + PREFETCH_DISTANCE * VEC_LEN < len) {
        @prefetch(a.ptr + i + PREFETCH_DISTANCE * VEC_LEN, .{ .rw = .read, .locality = 3 });
        @prefetch(b.ptr + i + PREFETCH_DISTANCE * VEC_LEN, .{ .rw = .read, .locality = 3 });
    }

    // Process current chunk...
}
```

### 2. Non-Temporal Stores (Small-Medium Impact)

After bind, written values won't be read until the next round (when they'll be cache-cold anyway). Streaming stores bypass cache pollution:

```zig
// Use non-temporal store hint (locality=0)
@prefetch(dst.ptr + i, .{ .rw = .write, .locality = 0 });
dst.* = @bitCast(reduce64Vec(VEC_LEN, result));
```

On x86 this can emit `movntdq`; on ARM it affects cache behavior.

### 3. Multiple Accumulators (Medium Impact)

Hide accumulator dependency latency with independent accumulators:

```zig
// Instead of 1 accumulator pair
var acc_a_0: VecU64 = @splat(0);
var acc_a_1: VecU64 = @splat(0);
var acc_b_0: VecU64 = @splat(0);
var acc_b_1: VecU64 = @splat(0);

while (i + 2 * VEC_LEN <= len) : (i += 2 * VEC_LEN) {
    // Iteration A - uses acc_*_0
    const r0 = sumAndLinearCombineChunk(...);
    acc_a_0 +%= r0.sum_a;
    acc_b_0 +%= r0.sum_b;

    // Iteration B - uses acc_*_1 (independent chain)
    const r1 = sumAndLinearCombineChunk(...);
    acc_a_1 +%= r1.sum_a;
    acc_b_1 +%= r1.sum_b;
}

// Combine at end
const sum_a = @reduce(.Add, acc_a_0) + @reduce(.Add, acc_a_1);
const sum_b = @reduce(.Add, acc_b_0) + @reduce(.Add, acc_b_1);
```

### 4. Loop Unrolling

Process multiple chunks per iteration to reduce loop overhead:

```zig
// Unroll 4x
while (i + 4 * VEC_LEN <= len) : (i += 4 * VEC_LEN) {
    inline for (0..4) |j| {
        const offset = i + j * VEC_LEN;
        const result = sumAndLinearCombineChunk(...);
        // accumulate...
    }
}
```

### Optimization Priority

| Optimization | Effort | Expected Gain | Notes |
|--------------|--------|---------------|-------|
| Fusion (Steps 1-3) | Medium | ~33% | Foundation - do first |
| SIMD fused loop | Medium | 2-4x vs scalar | Already in Step 1a |
| Prefetching | Low | 10-20% for n=24 | Hides RAM latency |
| Multiple accumulators | Low | 5-15% | Better ILP |
| Non-temporal stores | Low | 5-10% | Reduces cache pressure |
| Loop unrolling | Low | 0-5% | May help or hurt |

### Projected Final Performance

| Stage | n=24 Time | Throughput |
|-------|-----------|------------|
| Current (2-pass) | 12.7 ms | 21.0 GB/s |
| After fusion | ~8-9 ms | ~28-32 GB/s |
| + Prefetch/NT stores | ~7-8 ms | ~32-36 GB/s |

Memory bandwidth limit on M-series: ~100 GB/s. With fusion reading ~96MB total per round × 24 rounds = 2.3GB, theoretical minimum is ~23ms... but we're processing less due to halving each round. Actual data touched is ~192MB (geometric series), so ~2ms at peak bandwidth. Real gains limited by memory latency, not bandwidth.

## PCS Compatibility

This optimization requires commit-then-challenge. Compatible schemes:

| Scheme | Field | Notes |
|--------|-------|-------|
| Merkle (raw) | Any | Simple, large proofs |
| Brakedown | Any | Linear-time, practical |
| Ligero/Orion | Any | Code-based |
| Basefold | Any | Uses sumcheck internally |
| FRI | Small fields | Not planned |
| Binius | Binary only | Not for M31 |

## Testing Strategy

1. **Correctness**: Fused result matches non-fused
   ```zig
   test "fused matches standard" {
       // Run both, compare round polys and final eval
   }
   ```

2. **SIMD paths**: Test array sizes that hit vector and scalar paths
   ```zig
   test "fused SIMD and tail" {
       // Test n=32 (exact SIMD), n=37 (with tail)
   }
   ```

3. **Edge cases**: n=1 (single variable), large n

## Open Questions

1. **Transcript binding**: Should the commitment include domain separation?

2. **Streaming commitment**: Can we compute commitment while doing first sumcheck round?

3. **Parallel sumcheck**: With known challenges, could we parallelize across rounds?
   - Round i needs result of round i-1, so not directly
   - But could parallelize WITHIN each round (different cores process chunks)

4. **Basefold integration**: Basefold uses sumcheck internally — would this create a recursive optimization opportunity?

5. **Threading strategy**: With known challenges, each round can be parallelized internally:
   - Split data into L2-sized chunks across threads
   - Each thread computes partial sums + binds its chunk
   - Reduce partial sums at end of each round
   - Combines well with prefetching (each thread prefetches its own stream)

## Files to Modify

- [ ] `src/fields/mersenne31.zig` — add `sumAndLinearCombineBatch` (SIMD fused operation)
- [ ] `src/poly/multilinear.zig` — add `sumHalvesAndBind` (delegates to field)
- [ ] `src/sumcheck.zig` — add `proveWithChallenges`
- [ ] `src/commit.zig` — new file, commitment interface
- [ ] `bench/sumcheck.zig` — benchmark fused vs standard
- [ ] `ARCHITECTURE.md` — document commit-then-challenge model

## References

- BDDT paper: https://eprint.iacr.org/2024/1046
- Brakedown: https://eprint.iacr.org/2021/1043
- Basefold: https://eprint.iacr.org/2024/1571
- Ligero: https://eprint.iacr.org/2022/1608
