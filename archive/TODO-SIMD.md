# SIMD Implementation Guide

## Overview

This document captures learnings from SIMD optimization experiments on Mersenne31 field operations. Use this as a guide for reimplementation.

## Architecture

### File Structure

```
src/simd.zig           - Vector length configuration (comptime, generic)
src/fields/*.zig       - Field-specific SIMD operations (reduction logic)
src/poly/*.zig         - Algorithm-specific SIMD (NTT, etc.)
```

### simd.zig

Minimal configuration file. Only provides comptime vector lengths:

- `u32_len`: Use for 32-bit field elements (Mersenne31, BabyBear)
- `u64_len`: Use for 64-bit field elements (Goldilocks)

These values adapt at compile time based on target architecture:

- ARM NEON: 4 (128-bit)
- x86 SSE: 4 (128-bit)
- x86 AVX2: 8 (256-bit)
- AVX-512: 16 (512-bit)
- WASM SIMD: 4 (128-bit)

No runtime detection needed. No fallback paths needed - all modern targets have SIMD.

## Key Principles

### 1. Vector Length Based on Input/Output Type

Choose vector length based on your PRIMARY data type, not intermediate types.

For Mersenne31 (u32 elements):

- Use `u32_len` to determine how many elements per iteration
- Even though multiply widens to u64, we still process `u32_len` elements
- The u64 intermediate just uses 2x the registers (acceptable)

Wrong: Using `u64_len` for Mersenne31 (processes fewer elements, wastes register space)
Right: Using `u32_len` for Mersenne31 (maximizes throughput)

### 2. Why Widening to u64 is Necessary

Mersenne31 elements are 31 bits. Multiplication produces 62-bit results:

- (2^31 - 1) × (2^31 - 1) ≈ 2^62
- This exceeds u32 capacity (2^32)
- Must widen to u64 before multiply, then reduce back

The widening pattern:

1. Load u32 vectors
2. Widen to u64 vectors
3. Perform arithmetic (multiply, add)
4. Reduce back to u32 (field-specific reduction)
5. Store u32 vectors

### 3. Reduction is Field-Specific

Each field has unique reduction logic. For Mersenne31:

- Uses the identity: 2^31 ≡ 1 (mod p)
- Reduction: `(x & MODULUS) + (x >> 31)`
- This must be implemented per-field, not generically

### 4. No Scalar Fallbacks

All target platforms have SIMD. Simplify code by assuming SIMD is always available:

- Use `simd.u32_len orelse 4` if you must have a default
- But prefer compile error over silent fallback for unsupported targets

### 5. Comptime Everything

All vector configuration should be comptime:

- Vector lengths are comptime known
- Vector types are comptime constructed
- No runtime branching on SIMD availability

### 6. Packed Struct Layout for SIMD

Field element types must use `packed struct(u32)` for safe SIMD casting:

```zig
pub const Mersenne31 = packed struct(u32) {
    value: u32,
    // ... methods
};
```

This guarantees:

- `@sizeOf(Mersenne31) == 4`
- `@bitCast` works between `Mersenne31` and `u32`
- `@bitCast` works between `[N]Mersenne31` and `@Vector(N, u32)`

Without explicit layout, struct reinterpretation is undefined behavior.

### 7. Alignment (Caller Responsibility)

SIMD loads/stores perform best with aligned memory:

- AVX2 (256-bit): 32-byte alignment
- AVX-512 (512-bit): 64-byte alignment
- NEON/SSE (128-bit): 16-byte alignment

Callers are responsible for providing aligned memory:

```zig
// Heap allocation with alignment
const data = try allocator.alignedAlloc(Mersenne31, 32, count);
defer allocator.free(data);

// Stack allocation with alignment
var buf: [1024]Mersenne31 align(32) = undefined;
```

The batch operations assume aligned input. Unaligned access works but may be slower due to cache line splits.

For a generic alignment constant based on target:

```zig
// In simd.zig
pub const alignment: usize = if (u32_len) |len| len * 4 else 16;
```

### 8. Tail Handling

When input length isn't divisible by VEC_LEN, handle the remainder:

```zig
var i: usize = 0;
while (i + VEC_LEN <= len) : (i += VEC_LEN) {
    // SIMD loop body
}
// Tail: process remaining elements
while (i < len) : (i += 1) {
    // Same logic, single element
}
```

The tail loop uses the same algorithm but processes one element at a time.
For small tails (< VEC_LEN elements), this is negligible overhead.

Alternative: require aligned lengths at the API level (push complexity to caller).

## Implementation Guidelines

### For linearCombineBatch (highest priority)

This is the hot path for multilinear polynomial binding.

Operation: `dst[i] = a[i] + r * (b[i] - a[i])`

Optimizations that worked:

1. Extract inner loop to separate function (helps compiler)
2. Use explicit `@Vector(VEC_LEN, u32)` types
3. Compute diff in u32: `b + MODULUS - a` (no overflow, values < MODULUS)
4. Widen to u64 only for multiply
5. Inline vector reduction at end (no separate helper function)

Expected speedup: 1.3x - 1.5x over scalar

### For dotProduct

Operation: `sum(a[i] * b[i])`

**Critical: Accumulator Overflow**

Each product is ~62 bits. A u64 accumulator overflows after just 4 products:

- Max product: (2^31 - 2)^2 ≈ 2^62
- Overflow threshold: 2^64 / 2^62 = 4 products

Wrapping arithmetic does NOT work because 2^64 ≡ 4 (mod p), not 0.
Each overflow introduces an error of 4 in the final result.

**Solution: Single partial reduction step per product**

A full reduce64 has 3 steps. But we only need ONE step to keep values bounded:

```
Product:           < 2^62 (62 bits)
Partial reduction: lo = x & MODULUS     → < 2^31
                   hi = x >> 31         → < 2^31 (since x < 2^62)
                   partial = lo + hi    → < 2^32
```

One step brings 62 bits down to ~32 bits. Now we can accumulate:

```
u64 can hold: 2^64 / 2^32 = 2^32 partial values (4 billion products)
```

The key insight: `partial ≡ product (mod p)` because `2^31 ≡ 1 (mod p)`.
So `lo + hi ≡ lo + hi * 2^31 ≡ original product (mod p)`.

Pattern:

1. Use VEC_LEN parallel u64 accumulators
2. Load VEC_LEN u32 elements from a and b
3. Widen both to u64
4. Multiply element-wise (result is ~62 bits)
5. **Single partial reduction**: `acc += (prod & MODULUS) + (prod >> 31)`
6. Horizontal sum at end with `@reduce(.Add, acc)`
7. Single full reduce64 on final scalar

Inner loop is just: mask, shift, two adds. No second reduction, no branchless subtract.

### For sumSlices (core sumcheck operation)

Operation: Sum N slices in parallel, returning N results.

Primary use case is sumcheck round computation:

```zig
// Sumcheck: compute g(0) and g(1) by summing left/right halves
const half = evals.len / 2;
const sums = Mersenne31.sumSlices(2, .{evals[0..half], evals[half..]});
```

The slices are contiguous regions (adjacent halves of one buffer), so vector loads within each slice are efficient.

**Current implementation is suboptimal:**

```zig
// Current: scalar loads, N per iteration
for (0..len) |idx| {
    inline for (0..N) |s| {
        accs[s] += slices[s][idx];  // scalar
    }
}
```

**Optimal: vectorize within each slice:**

```zig
var acc: [N]@Vector(VEC_LEN, u64) = .{@splat(0)} ** N;

var i: usize = 0;
while (i + VEC_LEN <= len) : (i += VEC_LEN) {
    inline for (0..N) |s| {
        const chunk: @Vector(VEC_LEN, u32) = @bitCast(slices[s][i..][0..VEC_LEN].*);
        acc[s] += chunk;  // vector load + vector add
    }
}

// Horizontal sum + reduce
inline for (0..N) |s| {
    results[s] = reduce64(@reduce(.Add, acc[s]));
}
```

Each iteration: N contiguous vector loads, N vector adds, processes N × VEC_LEN elements.

**Overflow:** Each value < 2^31, overflow after 2^33 elements. For sumcheck n=24 (8M elements per half), this is safe without partial reduction.

### Removed: sumSlice, addBatch/subBatch/mulBatch/mulAddBatch

- `sumSlice`: Replaced by `sumSlices(1, .{slice})`. Remove function and update tests.
- Element-wise batch ops (`addBatch`, `subBatch`, `mulBatch`, `mulAddBatch`): Not used in sumcheck/GKR protocols. Remove to keep the implementation simple. Add back only if a specific protocol requires them.

## Code Structure Philosophy

**SIMD is not an optimization - it's the implementation.**

- Scalar ops (`add`, `mul`, `reduce64`): Single-element operations, keep simple
- Batch ops (`linearCombineBatch`, `dotProduct`, etc.): These ARE vector operations

Use `reduce64Vec` for vector reduction in batch operations. This keeps the reduction logic in one place (DRY) while the scalar `reduce64` remains for single-element operations. The `inline fn` ensures zero overhead. This keeps the code simple and makes the SIMD nature explicit.

The reduction logic (same algorithm as scalar, just on vectors):

1. First reduction: `lo = x & MODULUS`, `hi = x >> 31`, `sum = lo + hi`
2. Second reduction: same operation on sum
3. Final reduction: branchless subtract using `@select`
4. Truncate to u32 vector

All operations are parallel across lanes - no lane dependencies. Write this inline in each batch function.

## Performance Notes

### What Worked

- Explicit `@Vector` types with VEC_LEN from simd.zig
- Extracting inner loops to separate functions
- Processing VEC_LEN elements per iteration based on u32_len
- Inlining vector reduction logic directly in batch functions

### What Didn't Work

- Manual loop unrolling (made things slower)
- SIMD for simple reduction loops like sumSlice (LLVM does better)
- Using u64_len instead of u32_len (processed too few elements)

### Memory Bandwidth Considerations

At large sizes (n=24, 16M elements), operations become memory-bound.
SIMD helps but gains diminish as memory bandwidth saturates.
Expect ~1.3x speedup even for well-optimized SIMD code.

## Testing Strategy

1. Ensure existing tests pass (correctness)
2. Run benchmarks before/after for each function
3. Test on both ARM (local) and x86 (cloud) targets
4. Verify speedups scale appropriately with vector length

## Implementation Order

1. ✅ simd.zig - vector length config (`u32_len`, `u64_len`, `alignment`)
2. ✅ Remove `sumSlice` and unused batch ops, update tests to use `sumSlices`
3. ✅ Mersenne31 changed to `packed struct(u32)` for safe `@bitCast`
4. ✅ linearCombineBatch - highest impact, used by multilinear bind
5. ✅ sumSlices - core sumcheck operation, sum N slices in parallel
6. ✅ dotProduct - used in sumcheck and other protocols
