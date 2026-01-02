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

### Step 1: Add `sumHalvesAndBind` to multilinear.zig

```zig
/// Fused operation: compute round polynomial AND bind in single pass.
/// Returns sums for round polynomial and the bound slice.
pub fn sumHalvesAndBind(comptime F: type, evals: []F, r: F) struct {
    sums: [2]F,
    bound: []F
} {
    const half = evals.len / 2;

    // SIMD-friendly single pass
    var sum_lo: u64 = 0;
    var sum_hi: u64 = 0;

    for (0..half) |i| {
        const lo = evals[i];
        const hi = evals[i + half];

        // Accumulate sums
        sum_lo += lo.value;
        sum_hi += hi.value;

        // Compute bound value: lo + r * (hi - lo)
        evals[i] = lo.add(r.mul(hi.sub(lo)));
    }

    return .{
        .sums = .{ F.reduce64(sum_lo), F.reduce64(sum_hi) },
        .bound = evals[0..half],
    };
}
```

**SIMD version**: Process in chunks, accumulate sums in vector registers, apply bind in same loop.

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

## Files to Modify

- [ ] `src/poly/multilinear.zig` — add `sumHalvesAndBind`
- [ ] `src/sumcheck.zig` — add `proveWithChallenges`
- [ ] `src/commit.zig` — new file, commitment interface
- [ ] `bench/sumcheck.zig` — benchmark fused vs standard
- [ ] `ARCHITECTURE.md` — document commit-then-challenge model

## References

- BDDT paper: https://eprint.iacr.org/2024/1046
- Brakedown: https://eprint.iacr.org/2021/1043
- Basefold: https://eprint.iacr.org/2024/1571
- Ligero: https://eprint.iacr.org/2022/1608
