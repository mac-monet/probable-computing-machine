# SIMD Architecture Refactor

## Goal

Move generic SIMD loop structure to `multilinear.zig`, with fields providing minimal vector primitives. This cleanly separates:
- **Multilinear**: memory access patterns, loop orchestration
- **Field**: arithmetic primitives (scalar + vector)

## Current Architecture

```
multilinear.zig              mersenne31.zig
├── sumHalves → F.sumSlices  ├── add, sub, mul (scalar)
├── bind → F.linearCombineBatch  ├── sumSlices (SIMD)
└── evaluate                 ├── linearCombineBatch (SIMD)
                             └── dotProduct (SIMD)
```

**Problem**: Field does "too much" — batch operations are really polynomial-level, but SIMD is field-specific.

## Proposed Architecture

```
multilinear.zig                    mersenne31.zig
├── sumHalvesAndBind (SIMD loop)   ├── Scalar: add, sub, mul, reduce
├── bind (SIMD loop)               ├── Types: Int, Accum, MODULUS
├── sumHalves (SIMD loop)          ├── Vector: vecSub, reduceVec, widen
└── evaluate                       └── (no batch operations)
```

**Key insight**: SIMD loop structure is generic. Field just provides:
1. Type info (element size, accumulator size)
2. Vector primitives for field-specific operations

## Field Interface

```zig
// mersenne31.zig

// Types
pub const Int = u32;              // element storage
pub const Accum = u64;            // wide accumulator for sums/products
pub const MODULUS: u32 = 0x7FFFFFFF;

// Scalar ops (for verifier, transcript, small computations)
pub fn add(a: Mersenne31, b: Mersenne31) Mersenne31 { ... }
pub fn sub(a: Mersenne31, b: Mersenne31) Mersenne31 { ... }
pub fn mul(a: Mersenne31, b: Mersenne31) Mersenne31 { ... }

// Vector primitives (for prover batch operations)
pub fn widen(comptime N: usize, vec: @Vector(N, Int)) @Vector(N, Accum) {
    return vec;  // u32 -> u64 implicit
}

pub fn vecSub(comptime N: usize, a: @Vector(N, Int), b: @Vector(N, Int)) @Vector(N, Int) {
    // Mersenne trick: b + MODULUS - a avoids underflow
    return b +% @as(@Vector(N, Int), @splat(MODULUS)) -% a;
}

pub fn reduceVec(comptime N: usize, wide: @Vector(N, Accum)) @Vector(N, Int) {
    // Mersenne reduction: (x & MODULUS) + (x >> 31)
    const mask: @Vector(N, Accum) = @splat(MODULUS);
    const lo = wide & mask;
    const hi = wide >> @splat(31);
    const sum = lo +% hi;
    // Final reduction...
    return @truncate(sum);
}

pub fn reduceAccum(acc: Accum) Mersenne31 {
    return reduce64(acc);
}
```

## Multilinear with Generic SIMD

```zig
// multilinear.zig
const simd = @import("../simd.zig");

pub fn sumHalvesAndBind(comptime F: type, evals: []F, r: F) struct { sums: [2]F, bound: []F } {
    const half = evals.len / 2;
    const VEC_LEN = simd.register_bytes / @sizeOf(F);
    const Vec = @Vector(VEC_LEN, F.Int);
    const AccumVec = @Vector(VEC_LEN, F.Accum);

    var acc_a: AccumVec = @splat(0);
    var acc_b: AccumVec = @splat(0);
    const r_wide: AccumVec = @splat(@as(F.Accum, r.value));

    var i: usize = 0;
    while (i + VEC_LEN <= half) : (i += VEC_LEN) {
        // LOAD FIRST (aliasing safety)
        const a_vec: Vec = @bitCast(evals[i..][0..VEC_LEN].*);
        const b_vec: Vec = @bitCast(evals[half + i..][0..VEC_LEN].*);

        // Accumulate sums (generic widen + add)
        acc_a +%= F.widen(VEC_LEN, a_vec);
        acc_b +%= F.widen(VEC_LEN, b_vec);

        // Linear combine: a + r * (b - a)
        const diff = F.vecSub(VEC_LEN, a_vec, b_vec);  // field-specific subtraction
        const prod = F.widen(VEC_LEN, diff) *% r_wide; // generic wide multiply
        const result = F.widen(VEC_LEN, a_vec) +% prod; // generic wide add

        // STORE AFTER
        evals[i..][0..VEC_LEN].* = @bitCast(F.reduceVec(VEC_LEN, result));
    }

    // Scalar tail loop
    var sum_a: F.Accum = @reduce(.Add, acc_a);
    var sum_b: F.Accum = @reduce(.Add, acc_b);
    while (i < half) : (i += 1) {
        sum_a +%= evals[i].value;
        sum_b +%= evals[half + i].value;
        evals[i] = evals[i].add(r.mul(evals[half + i].sub(evals[i])));
    }

    return .{
        .sums = .{ F.reduceAccum(sum_a), F.reduceAccum(sum_b) },
        .bound = evals[0..half],
    };
}
```

## Why Keep Scalar Ops?

Even in a SIMD-native library, scalar ops are needed for:

| Use case | Why scalar |
|----------|------------|
| Verifier | Processes single elements, no batching |
| Transcript | absorb/squeeze individual field elements |
| Round poly eval | `eval_0 + c * (eval_1 - eval_0)` |
| Tail loops | Handle non-aligned remainders |
| Tests | Easier to write/debug |

Going "all SIMD" (scalars as length-1 vectors) adds overhead for no gain in these cases.

## Migration Steps

1. [ ] Add vector primitives to `mersenne31.zig`: `widen`, `vecSub`, `reduceVec`
2. [ ] Implement `sumHalvesAndBind` in `multilinear.zig` with generic SIMD
3. [ ] Verify correctness: fused result matches non-fused
4. [ ] Benchmark: compare against current field-based batch ops
5. [ ] Remove old batch ops from field: `sumSlices`, `linearCombineBatch`
6. [ ] Update `bind`, `sumHalves` to use same pattern (or deprecate if unused)

## Comparison

| Aspect | Current | Proposed |
|--------|---------|----------|
| SIMD location | Field | Multilinear |
| Field responsibility | Scalar + batch ops | Scalar + vector primitives |
| Multilinear responsibility | Thin coordinator | SIMD orchestration |
| Adding new field | Implement all batch ops | Implement vector primitives |
| Code clarity | Field does "too much" | Clear separation |

## Open Questions

1. Should `bind()` and `sumHalves()` remain as separate functions, or only keep fused version?

2. For `evaluate()` which uses repeated `bind()`: keep separate `bind` with SIMD, or use scalar loop (it's log(n) binds, small)?

3. Naming: `vecSub` vs `simdSub` vs `batchSub`?

## Files to Modify

- [ ] `src/fields/mersenne31.zig` — add vector primitives, remove batch ops
- [ ] `src/poly/multilinear.zig` — add generic SIMD loops
- [ ] `src/sumcheck.zig` — use new `sumHalvesAndBind`
- [ ] `bench/sumcheck.zig` — benchmark new implementation

## Relationship to TODO-FUSE.md

This refactor is a **prerequisite** for the fused sumcheck optimization. Once complete:
- `sumHalvesAndBind` will be the main operation
- Commit-then-challenge enables using it (challenges known upfront)
- Post-fusion optimizations (prefetching, NT stores) layer on top
