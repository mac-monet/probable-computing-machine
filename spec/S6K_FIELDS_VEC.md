# s6k Field Vector Operations

Fields provide a `vecOps` interface that s6k uses for SIMD-accelerated computation. This separates algorithm structure (s6k) from arithmetic strategy (field).

## Why Fields Own SIMD

Different fields have fundamentally different SIMD strategies:

| Field | Multiply Width | Reduction | Overflow Budget |
|-------|---------------|-----------|-----------------|
| Mersenne31 | 32→64 bit | `(x & p) + (x >> 31)` | ~2^32 partials |
| BabyBear | 32→64 bit | Montgomery | Different |
| Goldilocks | 64→128 bit | Different reduction | Needs u128 |
| Binary F₂ᵏ | Same width | XOR (no reduction) | Unlimited |

s6k can't have one implementation. Fields provide the arithmetic, s6k provides the structure.

---

## VecOps Interface

```zig
/// Fields must provide this interface for s6k SIMD support
pub const vecOps = struct {
    // ══════════════════════════════════════════════════════════════
    // Configuration
    // ══════════════════════════════════════════════════════════════

    /// Preferred SIMD width (elements per vector)
    /// Typically: 8 for 32-bit fields (AVX2), 4 for 64-bit fields
    pub const preferred_width: comptime_int;

    /// Accumulator element type (may be wider than field)
    /// Mersenne31: u64, Goldilocks: u128, Binary: same as field
    pub const Accumulator: type;

    /// How many accumulations before partialReduce is needed
    /// Mersenne31: ~2^32, Binary: unlimited (maxInt)
    pub const accumulation_budget: usize;

    // ══════════════════════════════════════════════════════════════
    // Types
    // ══════════════════════════════════════════════════════════════

    /// Vector of N field elements (packed representation)
    pub fn Vec(comptime N: comptime_int) type;

    /// Vector of N accumulators (for intermediate sums)
    pub fn AccumVec(comptime N: comptime_int) type;

    // ══════════════════════════════════════════════════════════════
    // Load/Store
    // ══════════════════════════════════════════════════════════════

    /// Load N elements from aligned pointer
    pub fn load(comptime N: comptime_int, ptr: *const [N]F) Vec(N);

    /// Store N elements to aligned pointer
    pub fn store(comptime N: comptime_int, ptr: *[N]F, v: Vec(N)) void;

    // ══════════════════════════════════════════════════════════════
    // Arithmetic
    // ══════════════════════════════════════════════════════════════

    /// Vector addition (may include partial reduction)
    pub fn addVec(comptime N: comptime_int, a: Vec(N), b: Vec(N)) Vec(N);

    /// Vector subtraction
    pub fn subVec(comptime N: comptime_int, a: Vec(N), b: Vec(N)) Vec(N);

    /// Vector multiply → accumulator (includes field-specific handling)
    pub fn mulVec(comptime N: comptime_int, a: Vec(N), b: Vec(N)) AccumVec(N);

    /// Scalar broadcast multiply: scalar * vec → accumulator
    pub fn scalarMulVec(comptime N: comptime_int, scalar: F, v: Vec(N)) AccumVec(N);

    // ══════════════════════════════════════════════════════════════
    // Accumulation
    // ══════════════════════════════════════════════════════════════

    /// Add to accumulator (for summing products)
    pub fn accumAdd(comptime N: comptime_int, acc: AccumVec(N), v: AccumVec(N)) AccumVec(N);

    /// Zero accumulator
    pub fn accumZero(comptime N: comptime_int) AccumVec(N);

    // ══════════════════════════════════════════════════════════════
    // Reduction (field-specific!)
    // ══════════════════════════════════════════════════════════════

    /// Partial reduction: keep accumulator in safe range
    /// Called every `accumulation_budget` iterations
    pub fn partialReduce(comptime N: comptime_int, acc: AccumVec(N)) AccumVec(N);

    /// Full reduction: accumulator → field elements
    pub fn reduce(comptime N: comptime_int, acc: AccumVec(N)) Vec(N);

    /// Horizontal sum: vector accumulator → single field element
    pub fn horizontalSum(comptime N: comptime_int, acc: AccumVec(N)) F;
};
```

---

## Mersenne31 Implementation

```zig
pub const Mersenne31 = packed struct(u32) {
    value: u32,

    pub const MODULUS: u32 = 0x7FFFFFFF;

    pub const vecOps = struct {
        pub const preferred_width = 8;  // AVX2: 8 × u32
        pub const Accumulator = u64;
        pub const accumulation_budget = 1 << 30;  // Safe with partial reduction

        pub fn Vec(comptime N: comptime_int) type {
            return @Vector(N, u32);
        }

        pub fn AccumVec(comptime N: comptime_int) type {
            return @Vector(N, u64);
        }

        pub fn load(comptime N: comptime_int, ptr: *const [N]Mersenne31) Vec(N) {
            return @bitCast(ptr.*);
        }

        pub fn store(comptime N: comptime_int, ptr: *[N]Mersenne31, v: Vec(N)) void {
            ptr.* = @bitCast(v);
        }

        pub fn mulVec(comptime N: comptime_int, a: Vec(N), b: Vec(N)) AccumVec(N) {
            const a_wide: AccumVec(N) = a;
            const b_wide: AccumVec(N) = b;
            const prod = a_wide *% b_wide;

            // Mersenne partial reduction: 62 bits → ~32 bits
            // Key: 2^31 ≡ 1 (mod p), so (x & p) + (x >> 31) ≡ x (mod p)
            const mask: AccumVec(N) = @splat(MODULUS);
            return (prod & mask) +% (prod >> @splat(31));
        }

        pub fn accumAdd(comptime N: comptime_int, acc: AccumVec(N), v: AccumVec(N)) AccumVec(N) {
            return acc +% v;
        }

        pub fn accumZero(comptime N: comptime_int) AccumVec(N) {
            return @splat(0);
        }

        pub fn partialReduce(comptime N: comptime_int, acc: AccumVec(N)) AccumVec(N) {
            const mask: AccumVec(N) = @splat(MODULUS);
            return (acc & mask) +% (acc >> @splat(31));
        }

        pub fn reduce(comptime N: comptime_int, acc: AccumVec(N)) Vec(N) {
            var r = partialReduce(N, acc);
            r = partialReduce(N, r);
            // Final: subtract modulus if >= modulus
            const mask: AccumVec(N) = @splat(MODULUS);
            const ge = r >= mask;
            r = @select(u64, ge, r -% mask, r);
            return @truncate(r);
        }

        pub fn horizontalSum(comptime N: comptime_int, acc: AccumVec(N)) Mersenne31 {
            const scalar: u64 = @reduce(.Add, acc);
            return Mersenne31.reduce64(scalar);
        }

        pub fn scalarMulVec(comptime N: comptime_int, scalar: Mersenne31, v: Vec(N)) AccumVec(N) {
            const s: AccumVec(N) = @splat(scalar.value);
            const v_wide: AccumVec(N) = v;
            const prod = s *% v_wide;
            const mask: AccumVec(N) = @splat(MODULUS);
            return (prod & mask) +% (prod >> @splat(31));
        }

        pub fn subVec(comptime N: comptime_int, a: Vec(N), b: Vec(N)) Vec(N) {
            // a - b = a + (p - b), then reduce
            const p: Vec(N) = @splat(MODULUS);
            const diff = a +% (p -% b);
            const ge = diff >= p;
            return @select(u32, ge, diff -% p, diff);
        }

        pub fn addVec(comptime N: comptime_int, a: Vec(N), b: Vec(N)) Vec(N) {
            var sum = a +% b;
            sum = (sum & @as(Vec(N), @splat(MODULUS))) +% (sum >> @splat(31));
            const ge = sum >= @as(Vec(N), @splat(MODULUS));
            return @select(u32, ge, sum -% @as(Vec(N), @splat(MODULUS)), sum);
        }
    };
};
```

---

## Binary Field Implementation (F₂⁶⁴)

```zig
pub const BinaryField64 = packed struct(u64) {
    value: u64,

    pub const vecOps = struct {
        pub const preferred_width = 4;  // 4 × u64
        pub const Accumulator = u64;    // No widening needed
        pub const accumulation_budget = std.math.maxInt(usize);  // XOR never overflows

        pub fn Vec(comptime N: comptime_int) type {
            return @Vector(N, u64);
        }

        pub fn AccumVec(comptime N: comptime_int) type {
            return @Vector(N, u64);  // Same as Vec for binary
        }

        pub fn mulVec(comptime N: comptime_int, a: Vec(N), b: Vec(N)) AccumVec(N) {
            // Carry-less multiply (CLMUL instruction or polynomial mul)
            return carrylessMul(a, b);
        }

        pub fn accumAdd(comptime N: comptime_int, acc: AccumVec(N), v: AccumVec(N)) AccumVec(N) {
            // Binary field: addition is XOR
            return acc ^ v;
        }

        pub fn partialReduce(comptime N: comptime_int, acc: AccumVec(N)) AccumVec(N) {
            // Binary field: no reduction needed during accumulation
            return acc;
        }

        pub fn reduce(comptime N: comptime_int, acc: AccumVec(N)) Vec(N) {
            // May need modular reduction by irreducible polynomial
            return moduloIrreducible(acc);
        }

        pub fn horizontalSum(comptime N: comptime_int, acc: AccumVec(N)) BinaryField64 {
            // XOR all lanes
            return .{ .value = @reduce(.Xor, acc) };
        }

        pub fn addVec(comptime N: comptime_int, a: Vec(N), b: Vec(N)) Vec(N) {
            return a ^ b;  // Binary addition is XOR
        }

        pub fn subVec(comptime N: comptime_int, a: Vec(N), b: Vec(N)) Vec(N) {
            return a ^ b;  // Binary subtraction is also XOR
        }
    };
};
```

---

## s6k Usage

s6k uses VecOps without knowing field-specific details:

```zig
pub fn s6k(comptime F: type) type {
    const V = F.vecOps;
    const N = V.preferred_width;

    return struct {
        pub fn computeRound(
            comptime degree: comptime_int,
            buffer: [*]const F,
            len: usize,
        ) [degree + 1]F {
            var coeffs: [degree + 1]V.AccumVec(N) = .{V.accumZero(N)} ** (degree + 1);

            const half = len / 2;
            var i: usize = 0;
            var accum_count: usize = 0;

            while (i + N <= half) : (i += N) {
                // Load lo/hi for each polynomial, compute products,
                // accumulate into coeffs using V.accumAdd and V.mulVec

                // ... polynomial product expansion ...

                accum_count += 1;
                if (accum_count >= V.accumulation_budget) {
                    // Field-specific partial reduction
                    inline for (&coeffs) |*c| {
                        c.* = V.partialReduce(N, c.*);
                    }
                    accum_count = 0;
                }
            }

            // Horizontal sum with full reduction
            var result: [degree + 1]F = undefined;
            inline for (coeffs, 0..) |c, k| {
                result[k] = V.horizontalSum(N, c);
            }

            // Tail loop...

            return result;
        }

        pub fn foldInPlace(
            comptime num_polys: comptime_int,
            buffer: [*]F,
            len: usize,
            challenge: F,
        ) usize {
            const half = len / 2;
            var i: usize = 0;

            while (i + N <= half) : (i += N) {
                inline for (0..num_polys) |p| {
                    const base = p * len;
                    const lo = V.load(N, @ptrCast(buffer + base + i));
                    const hi = V.load(N, @ptrCast(buffer + base + half + i));

                    // folded = lo + challenge * (hi - lo)
                    const diff = V.subVec(N, hi, lo);
                    const prod = V.scalarMulVec(N, challenge, diff);
                    const sum = V.accumAdd(N, @as(V.AccumVec(N), lo), prod);
                    const result = V.reduce(N, sum);

                    V.store(N, @ptrCast(buffer + base + i), result);
                }
            }

            // Tail loop, compaction...

            return half;
        }
    };
}
```

---

## Adding a New Field

To add s6k support for a new field:

1. Implement scalar arithmetic (`add`, `mul`, `sub`, `inv`)
2. Add `vecOps` struct with:
   - `preferred_width` based on element size and target arch
   - `Accumulator` type for safe intermediate sums
   - `accumulation_budget` before overflow risk
   - All vector operations with field-appropriate reduction

The field owns its SIMD strategy. s6k just uses it.

---

## References

- S6K_SYNC.md - s6k primitive specification
- S6K_ASYNC.md - s6k runtime specification
