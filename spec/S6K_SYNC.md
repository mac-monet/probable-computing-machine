# s6k: Sumcheck Runtime v2

**s6k** (sumcheck) is a hyperoptimized, data-oriented sumcheck primitive toolkit designed for composability, performance, and future async execution.

## Design Philosophy

### Core Principles

1. **Data-Oriented** - Explicit memory layouts, contiguous buffers, zero hidden allocations
2. **Toolkit, Not Framework** - Primitives that protocols compose, not abstractions that constrain
3. **DOD over OOP** - No objects with methods; data and transformations on that data
4. **Async-Ready** - Stateless primitives that map directly to job graph execution
5. **Algorithm-Agnostic** - Supports Algorithms 1-6 from literature without baking in choices

### What s6k Is

- A collection of SIMD-optimized primitive functions
- Explicit data layout contracts
- Building blocks for sumcheck-based protocols

### What s6k Is Not

- A sumcheck "runner" or "executor"
- An abstraction over protocols
- A framework with callbacks or virtual dispatch

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Protocol Layer (Basefold, Zerocheck, GKR, Lookups)             │
│    - Owns the sumcheck loop                                      │
│    - Chooses algorithm (1-6) based on constraints                │
│    - Manages transcript, commits, phase transitions              │
│    - Allocates buffers according to s6k layout contracts         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ calls primitives, passes buffers
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  s6k Toolkit                                                     │
│    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐        │
│    │ Round Compute │ │ Fold Ops      │ │ Eq Table Ops  │        │
│    └───────────────┘ └───────────────┘ └───────────────┘        │
│    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐        │
│    │ Accumulators  │ │ Tensor Ops    │ │ Lagrange Ops  │        │
│    └───────────────┘ └───────────────┘ └───────────────┘        │
│                                                                  │
│    - Pure functions: inputs → outputs                            │
│    - Zero allocations (caller provides all buffers)              │
│    - Explicit data layout requirements                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ SIMD / GPU implementations
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Backend (compile-time selected)                                 │
│    - Scalar (reference)                                          │
│    - SIMD (AVX2/AVX-512/NEON)                                    │
│    - GPU (Metal/CUDA) [future]                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Layouts

All s6k primitives operate on data with explicit layout contracts. Violating layouts is undefined behavior.

### PolyBuffer: Contiguous Polynomial Storage

```
Layout: [p₀[0..len], p₁[0..len], ..., p_{n-1}[0..len]]

Access: buffer[poly_idx * len + elem_idx]

Memory: Single contiguous allocation
        ┌─────────────────────────────────────────────────────┐
        │ p₀                │ p₁                │ p₂          │
        │ [lo₀|hi₀]         │ [lo₁|hi₁]         │ [lo₂|hi₂]   │
        └─────────────────────────────────────────────────────┘
        ↑                                                     ↑
        buffer.ptr                               buffer.ptr + num_polys * len
```

Each polynomial uses **standard multilinear layout**:
```
[lo₀, lo₁, ..., lo_{n/2-1}, hi₀, hi₁, ..., hi_{n/2-1}]

Where: f(0, x') = lo[x']
       f(1, x') = hi[x']
```

**Rationale**:
- Sequential access triggers hardware prefetching
- SIMD can process W consecutive elements
- Fold operation is simple: `folded[i] = lo[i] + c·(hi[i] - lo[i])`
- Natural output of NTT and other operations

### AccumulatorTable: Precomputed Sums for Algorithm 4 (No Eq)

For sumchecks WITHOUT eq polynomial (pure small-value optimization):

```
Formula: A_i(v, u) = Σ_{x'∈{0,1}^{ℓ-i}} ∏_{k=1}^d p_k(v, u, x')

         where v ∈ U_d^{i-1} = {0,1,...,d}^{i-1} indexes Lagrange
         evaluation points, NOT sum indices.

         The product p_k(v, u, x') evaluates p_k at point
         (v₁, v₂, ..., v_{i-1}, u, x').

Layout: For each round i ∈ [1, ℓ₀], for each u ∈ Û_d:
        A_i(v, u) for v ∈ U_d^{i-1}

        Optimized for inner product: t_i(u) = ⟨R_i, A_i(·, u)⟩

Memory: [round_1_data | round_2_data | ... | round_ℓ₀_data]

        Where round_i_data = [A_i(·,0) | A_i(·,1) | ... | A_i(·,d-1)]
        And   A_i(·,u) = [A_i(0,u), A_i(1,u), ..., A_i(num_v-1,u)]

Field:  BASE FIELD - all products are ss (small × small)
```

**Size**: `Σᵢ (d+1)^{i-1} × d` field elements

For d=2, ℓ₀=5: `(1 + 3 + 9 + 27 + 81) × 2 = 242` elements

### EqAccumulatorTable: Precomputed Sums for Algorithm 6 (With Eq)

For sumchecks WITH eq polynomial (combines SVO with Gruen):

```
Formula: A_i(v, u) = Σ_{x_R∈{0,1}^{ℓ/2-i}} ẽq(w_R, x_R) ·
                     Σ_{x_L∈{0,1}^{ℓ/2}} ẽq(w_L, x_L) ·
                     ∏_{k=1}^d p_k(v, u, x_L, x_R)

         where v ∈ U_d^{i-1} indexes Lagrange evaluation points.

         Note: x_L and x_R lengths are SWAPPED compared to Algorithm 5.
         This allows reusing inner sums across rounds.

         w_L = w_{[ℓ/2+1, ℓ]}  (second half of verifier challenges)
         w_R = w_{[i+1, ℓ/2]}  (first half, excluding processed rounds)

Layout: Same as AccumulatorTable

Field:  EXTENSION FIELD - eq terms are in Ext, so weighted sums are too

Mult costs:
  - ∏_k p_k(v, u, x_L, x_R): ss (base × base)
  - ẽq(w_L, x_L) · product: sl (ext × base) via mulBase
  - ẽq(w_R, x_R) · inner_sum: ll (ext × ext)
```

### ChallengeTensor: Kronecker Product of Lagrange Bases

```
Layout: R_i = ⊗_{j=1}^{i-1} (L_{U_d,k}(r_j))_{k=0}^d

        R_1 = [1]                           (length 1)
        R_2 = [L_0(r_1), L_1(r_1), L_2(r_1)] (length d+1)
        R_3 = R_2 ⊗ [L_0(r_2), ...]         (length (d+1)²)
        ...

Memory: Fixed buffer of size (d+1)^{ℓ₀-1}
        ┌──────────────────────────────────────────┐
        │ R_i elements (len grows each round)  |pad│
        └──────────────────────────────────────────┘

Field:  EXTENSION FIELD - challenges r_j are in Ext
```

For d=2, ℓ₀=5: max 81 elements (stack-allocatable)

### LinearFactorState: Gruen Optimization State (Algorithms 5/6)

For eq-weighted sumchecks, we factor: s_i(X) = l_i(X) · t_i(X)

```
l_i(X) = ẽq(w_{[1,i-1]}, r_{[1,i-1]}) · ẽq(w_i, X)
       = eq_prefix · ((1 - w_i)(1 - X) + w_i · X)

t_i(X) = "reduced" polynomial without the linear eq factor

State:  eq_prefix: Ext  -- accumulated ẽq(w_{[1,i-1]}, r_{[1,i-1]})
        w_i: Ext        -- current round's verifier challenge

Field:  EXTENSION FIELD - all verifier challenges w are in Ext
```

The prover computes t_i(u) for u ∈ Û_d = {0, 2, ..., d}, then recovers:
```
t_i(1) = l_i(1)^{-1} · (C_{i-1} - l_i(0) · t_i(0))
```
where C_{i-1} is the claimed sum from the previous round.

### EqTable: Equality Polynomial Evaluations

```
Standard: eq_table[x] = ẽq(w, x) for x ∈ {0,1}^n

Split (for Gruen optimization):
        eq_L[x_L] = ẽq(w_L, x_L) for x_L ∈ {0,1}^{ℓ/2}
        eq_R[x_R] = ẽq(w_R, x_R) for x_R ∈ {0,1}^{ℓ/2}

Memory: Contiguous arrays, accessed sequentially in inner loops
```

---

## s6k Primitive Interface

### Core Types

```zig
pub fn s6k(comptime F: type) type {
    return struct {
        // ══════════════════════════════════════════════════════════
        // Type Definitions
        // ══════════════════════════════════════════════════════════

        /// Round polynomial coefficients: round(t) = Σₖ cₖ·tᵏ
        pub fn Coeffs(comptime degree: comptime_int) type {
            return [degree + 1]F;
        }

        /// Polynomial buffer view (non-owning)
        pub const PolyBuffer = struct {
            data: [*]F,
            len: usize,
            num_polys: usize,

            pub fn poly(self: PolyBuffer, idx: usize) []F {
                return self.data[idx * self.len ..][0..self.len];
            }

            pub fn slice(self: PolyBuffer) []F {
                return self.data[0 .. self.num_polys * self.len];
            }
        };

        // ══════════════════════════════════════════════════════════
        // Round Computation (Algorithm 1 style)
        // ══════════════════════════════════════════════════════════

        /// Compute round polynomial: Σᵢ ∏ⱼ (loⱼ[i] + t·δⱼ[i])
        /// Returns coefficients [c₀, c₁, ..., c_d]
        ///
        /// Layout: buffer contains `degree` polynomials of length `len`
        ///         buffer[p * len + i] = p-th polynomial at index i
        pub fn computeRound(
            comptime degree: comptime_int,
            buffer: [*]const F,
            len: usize,
        ) Coeffs(degree) {
            // SIMD-optimized implementation
        }

        /// Compute round with eq weighting: Σᵢ eq[i] · ∏ⱼ (loⱼ[i] + t·δⱼ[i])
        /// For sumchecks involving ẽq(w, x) polynomial
        pub fn computeRoundWithEq(
            comptime degree: comptime_int,
            buffer: [*]const F,
            len: usize,
            eq: [*]const F,
        ) Coeffs(degree) {
            // degree of output is degree + 1 due to eq factor
        }

        /// Compute t_i(u) with split eq (Algorithm 5):
        ///
        /// t_i(u) = Σ_{x_R∈{0,1}^{ℓ/2}} ẽq(w_R, x_R) ·
        ///          Σ_{x_L∈{0,1}^{ℓ/2-i}} ẽq(w_L, x_L) ·
        ///          ∏_k (P_k[0,x_L,x_R] + u·(P_k[1,x_L,x_R] - P_k[0,x_L,x_R]))
        ///
        /// Returns t_i(u) for u ∈ Û_d = {0, 2, ..., d}.
        /// Does NOT return t_i(1) - use recoverT1() to get it.
        /// Does NOT return s_i - use applyLinearFactor() to convert.
        ///
        /// Note: This computes t_i, NOT s_i directly!
        pub fn computeRoundSplitEq(
            comptime degree: comptime_int,
            buffer: [*]const F,      // Polynomial evaluations
            len: usize,              // Current polynomial length (2^{ℓ-i+1})
            eq_L: [*]const F,        // ẽq(w_L, ·) for x_L ∈ {0,1}^{ℓ/2-i}
            eq_L_len: usize,         // 2^{ℓ/2-i}
            eq_R: [*]const F,        // ẽq(w_R, ·) for x_R ∈ {0,1}^{ℓ/2}
            eq_R_len: usize,         // 2^{ℓ/2}
        ) [degree]F {                // Note: returns d values, not d+1
            // Two-level summation for cache efficiency:
            // Outer loop: x_R (strided access)
            // Inner loop: x_L (contiguous access)
        }

        // ══════════════════════════════════════════════════════════
        // Fold Operations
        // ══════════════════════════════════════════════════════════

        /// Fold polynomials in-place: p[i] = lo[i] + c·(hi[i] - lo[i])
        /// Returns new length (len / 2)
        ///
        /// After fold, buffer layout becomes:
        /// [p₀_folded | p₁_folded | ... | (garbage)]
        /// Caller should update len tracking
        pub fn foldInPlace(
            comptime num_polys: comptime_int,
            buffer: [*]F,
            len: usize,
            challenge: F,
        ) usize {
            const half = len / 2;
            // Fold each polynomial, then compact
            inline for (0..num_polys) |p| {
                const base = p * len;
                var i: usize = 0;
                while (i < half) : (i += 1) {
                    const lo = buffer[base + i];
                    const hi = buffer[base + half + i];
                    buffer[base + i] = lo.add(challenge.mul(hi.sub(lo)));
                }
            }
            // Compact: move folded results to be contiguous
            inline for (1..num_polys) |p| {
                const src = p * len;
                const dst = p * half;
                @memcpy(buffer[dst..][0..half], buffer[src..][0..half]);
            }
            return half;
        }

        /// Fold base field polynomials to extension field (out-of-place)
        /// Required for first fold when challenge is in extension field
        pub fn foldToExt(
            comptime ExtF: type,
            comptime num_polys: comptime_int,
            base_buffer: [*]const F,
            len: usize,
            challenge: ExtF,
            ext_buffer: [*]ExtF,
        ) usize {
            const half = len / 2;
            inline for (0..num_polys) |p| {
                const base_offset = p * len;
                const ext_offset = p * half;
                var i: usize = 0;
                while (i < half) : (i += 1) {
                    const lo = ExtF.fromBase(base_buffer[base_offset + i]);
                    const hi = ExtF.fromBase(base_buffer[base_offset + half + i]);
                    ext_buffer[ext_offset + i] = lo.add(challenge.mul(hi.sub(lo)));
                }
            }
            return half;
        }

        // ══════════════════════════════════════════════════════════
        // Eq Table Operations
        // ══════════════════════════════════════════════════════════

        /// Build eq table: eq[x] = ẽq(challenges[0..num_vars], x)
        /// Output buffer must have length 2^num_vars
        pub fn buildEqTable(
            challenges: [*]const F,
            num_vars: usize,
            out: [*]F,
        ) void {
            // Incremental construction:
            // eq[0] = 1
            // For each challenge r_i:
            //   eq[x ++ 0] = eq[x] · (1 - r_i)
            //   eq[x ++ 1] = eq[x] · r_i
        }

        /// Fold eq table in-place after receiving challenge
        /// eq'[x'] = eq[0,x']·(1-r) + eq[1,x']·r
        pub fn foldEqTable(
            eq: [*]F,
            len: usize,
            challenge: F,
        ) usize {
            const half = len / 2;
            for (0..half) |i| {
                const lo = eq[i];
                const hi = eq[half + i];
                eq[i] = lo.add(challenge.mul(hi.sub(lo)));
            }
            return half;
        }

        // ══════════════════════════════════════════════════════════
        // Accumulator Operations - Algorithm 4 (No Eq)
        // ══════════════════════════════════════════════════════════

        /// Precompute accumulator table for small-value optimization (Algorithm 4)
        ///
        /// A_i(v, u) = Σ_{x'∈{0,1}^{ℓ-i}} ∏_{k=1}^d p_k(v, u, x')
        ///
        /// where v ∈ U_d^{i-1} indexes Lagrange evaluation points.
        ///
        /// All multiplications are ss (base × base).
        /// Output is in BASE field.
        ///
        /// Output layout: see AccumulatorTable documentation
        pub fn precomputeAccumulators(
            comptime degree: comptime_int,
            buffer: [*]const F,       // Original polynomial evals
            len: usize,               // 2^ℓ
            num_rounds: usize,        // ℓ₀
            out: [*]F,                // Output accumulator table
            out_offsets: [*]usize,    // Filled with round offsets
        ) void {
            // For each x' ∈ {0,1}^{ℓ-ℓ₀}:
            //   For each v ∈ U_d^{ℓ₀-1}:
            //     For each u ∈ Û_d:
            //       product = ∏_k p_k(v, u, x')   // ss mults
            //       A_i(v_prefix, u) += product   // for appropriate round i
        }

        // ══════════════════════════════════════════════════════════
        // Accumulator Operations - Algorithm 6 (With Eq)
        // ══════════════════════════════════════════════════════════

        /// Precompute eq-weighted accumulator table (Algorithm 6)
        ///
        /// A_i(v, u) = Σ_{x_R} ẽq(w_R, x_R) · Σ_{x_L} ẽq(w_L, x_L) · ∏_k p_k(v, u, x_L, x_R)
        ///
        /// where v ∈ U_d^{i-1} indexes Lagrange evaluation points.
        ///
        /// Multiplication costs:
        ///   - ∏_k p_k(...): ss (base × base)
        ///   - ẽq(w_L, x_L) · product: sl via Ext.mulBase
        ///   - ẽq(w_R, x_R) · inner_sum: ll (ext × ext)
        ///
        /// Output is in EXTENSION field.
        pub fn precomputeEqAccumulators(
            comptime ExtF: type,
            comptime degree: comptime_int,
            buffer: [*]const F,       // Original polynomial evals (base field)
            len: usize,
            num_rounds: usize,        // ℓ₀
            eq_L: [*]const ExtF,      // ẽq(w_L, ·), length 2^{ℓ/2}
            eq_L_len: usize,
            eq_R: [*]const ExtF,      // ẽq(w_R, ·), length 2^{ℓ/2}
            eq_R_len: usize,
            out: [*]ExtF,             // Output in extension field
            out_offsets: [*]usize,
        ) void {
            // Two-level summation with eq weighting
            // Inner sum over x_L is reused across rounds
        }

        // ══════════════════════════════════════════════════════════
        // Round from Accumulators
        // ══════════════════════════════════════════════════════════

        /// Compute t_i(u) from base field accumulators (Algorithm 4)
        /// t_i(u) = ⟨R_i, A_i(·, u)⟩
        ///
        /// This is a MIXED dot product: Ext · Base → Ext
        /// Uses sl multiplications via ExtF.mulBase
        ///
        /// Returns t_i(u) for u ∈ Û_d. Does NOT include t_i(1).
        /// Caller must use recoverT1() for eq-weighted sumchecks.
        pub fn computeRoundFromAccumulators(
            comptime ExtF: type,
            comptime degree: comptime_int,
            challenge_tensor: [*]const ExtF,  // R_i, length (d+1)^{i-1}
            tensor_len: usize,
            accumulators: [*]const F,         // A_i(·, u) in BASE field
            num_v: usize,                     // (d+1)^{i-1}
        ) [degree]ExtF {
            var coeffs: [degree]ExtF = undefined;
            inline for (0..degree) |u| {
                const accum_u = accumulators[u * num_v ..][0..num_v];
                // Mixed dot product: Σ_v R_i[v] * A_i(v, u)
                // Each term is sl: ExtF.mulBase(tensor[v], accum_u[v])
                var sum = ExtF.zero;
                for (0..tensor_len) |v| {
                    sum = sum.add(challenge_tensor[v].mulBase(accum_u[v]));
                }
                coeffs[u] = sum;
            }
            return coeffs;
        }

        /// Compute t_i(u) from extension field accumulators (Algorithm 6)
        /// t_i(u) = ⟨R_i, A_i(·, u)⟩
        ///
        /// This is Ext · Ext → Ext (ll multiplications)
        /// because eq-weighted accumulators are already in Ext.
        pub fn computeRoundFromEqAccumulators(
            comptime degree: comptime_int,
            challenge_tensor: [*]const F,     // R_i in Ext
            tensor_len: usize,
            accumulators: [*]const F,         // A_i(·, u) in Ext
            num_v: usize,
        ) [degree]F {
            var coeffs: [degree]F = undefined;
            inline for (0..degree) |u| {
                const accum_u = accumulators[u * num_v ..][0..num_v];
                // Ext · Ext dot product (ll)
                coeffs[u] = F.dotProduct(
                    challenge_tensor[0..tensor_len],
                    accum_u,
                );
            }
            return coeffs;
        }

        // ══════════════════════════════════════════════════════════
        // Tensor / Lagrange Operations
        // ══════════════════════════════════════════════════════════

        /// Compute Lagrange basis at point: {L_{U_d,k}(r)}_{k=0}^d
        /// Where U_d = {0, 1, ..., d} is the evaluation domain
        pub fn lagrangeBasis(
            comptime degree: comptime_int,
            point: F,
        ) [degree + 1]F {
            // L_k(x) = ∏_{j≠k} (x - j) / (k - j)
        }

        /// Extend challenge tensor: R_{i+1} = R_i ⊗ basis
        /// Writes to out buffer, returns new length
        ///
        /// in and out may alias if out has sufficient capacity
        pub fn extendTensor(
            comptime degree: comptime_int,
            in: [*]const F,
            in_len: usize,
            basis: [degree + 1]F,
            out: [*]F,
        ) usize {
            const new_len = in_len * (degree + 1);
            // Work backwards to allow in-place extension
            var i: usize = in_len;
            while (i > 0) {
                i -= 1;
                inline for (0..degree + 1) |k| {
                    out[i * (degree + 1) + k] = in[i].mul(basis[k]);
                }
            }
            return new_len;
        }

        // ══════════════════════════════════════════════════════════
        // Gruen Linear Factor Recovery (Algorithms 5/6)
        // ══════════════════════════════════════════════════════════

        /// Linear factor state for eq-weighted sumchecks.
        /// Tracks: s_i(X) = l_i(X) · t_i(X)
        ///
        /// All values are in EXTENSION field.
        pub const LinearFactorState = struct {
            /// ẽq(w_{[1,i-1]}, r_{[1,i-1]}) - accumulated product
            eq_prefix: F,

            /// Current round's verifier challenge w_i
            w_i: F,

            pub fn init(w_1: F) LinearFactorState {
                return .{ .eq_prefix = F.one, .w_i = w_1 };
            }

            /// Compute l_i(X) = eq_prefix · ẽq(w_i, X)
            /// Returns [l_i(0), l_i(1)]
            pub fn linearFactor(self: LinearFactorState) [2]F {
                // ẽq(w_i, 0) = 1 - w_i
                // ẽq(w_i, 1) = w_i
                return .{
                    self.eq_prefix.mul(F.one.sub(self.w_i)),  // l_i(0)
                    self.eq_prefix.mul(self.w_i),              // l_i(1)
                };
            }

            /// Update state after round i with challenge r_i
            pub fn update(self: *LinearFactorState, next_w: F, r_i: F) void {
                // eq_prefix *= ẽq(w_i, r_i) = (1-w_i)(1-r_i) + w_i·r_i
                const one_minus_w = F.one.sub(self.w_i);
                const one_minus_r = F.one.sub(r_i);
                const eq_eval = one_minus_w.mul(one_minus_r).add(self.w_i.mul(r_i));
                self.eq_prefix = self.eq_prefix.mul(eq_eval);
                self.w_i = next_w;
            }
        };

        /// Recover t_i(1) from the sum property.
        ///
        /// Given: t_i(u) for u ∈ Û_d = {0, 2, 3, ..., d}
        /// Need:  t_i(1)
        ///
        /// Using: s_i(0) + s_i(1) = C_{i-1}
        ///        l_i(0)·t_i(0) + l_i(1)·t_i(1) = C_{i-1}
        ///
        /// Therefore: t_i(1) = l_i(1)^{-1} · (C_{i-1} - l_i(0)·t_i(0))
        pub fn recoverT1(
            t_at_0: F,
            linear_factor: [2]F,  // [l_i(0), l_i(1)]
            claimed_sum: F,       // C_{i-1}
        ) F {
            const l0 = linear_factor[0];
            const l1 = linear_factor[1];
            return claimed_sum.sub(l0.mul(t_at_0)).mul(l1.inv());
        }

        /// Convert t_i coefficients to s_i coefficients.
        /// s_i(X) = l_i(X) · t_i(X)
        ///        = (l0 + (l1-l0)·X) · (t0 + t1·X + t2·X² + ...)
        ///
        /// Output degree is input degree + 1.
        pub fn applyLinearFactor(
            comptime degree: comptime_int,
            t_coeffs: [degree + 1]F,
            linear_factor: [2]F,
        ) [degree + 2]F {
            const l0 = linear_factor[0];
            const l1_minus_l0 = linear_factor[1].sub(linear_factor[0]);

            var s_coeffs: [degree + 2]F = undefined;
            // s_k = l0·t_k + (l1-l0)·t_{k-1}
            s_coeffs[0] = l0.mul(t_coeffs[0]);
            inline for (1..degree + 1) |k| {
                s_coeffs[k] = l0.mul(t_coeffs[k]).add(l1_minus_l0.mul(t_coeffs[k - 1]));
            }
            s_coeffs[degree + 1] = l1_minus_l0.mul(t_coeffs[degree]);
            return s_coeffs;
        }

        /// Build full t_i coefficients from partial evaluations.
        ///
        /// Input: t_i(u) for u ∈ Û_d = {0, 2, 3, ..., d} (d values)
        /// Plus:  t_i(1) from recoverT1()
        ///
        /// Output: coefficients [t_0, t_1, ..., t_d] via interpolation
        pub fn interpolateT(
            comptime degree: comptime_int,
            t_evals: [degree]F,    // t_i(0), t_i(2), t_i(3), ..., t_i(d)
            t_at_1: F,             // t_i(1) from recovery
        ) [degree + 1]F {
            // Lagrange interpolation over U_d = {0, 1, 2, ..., d}
            // Reorder evaluations: [t(0), t(1), t(2), ..., t(d)]
            var evals: [degree + 1]F = undefined;
            evals[0] = t_evals[0];
            evals[1] = t_at_1;
            inline for (2..degree + 1) |k| {
                evals[k] = t_evals[k - 1];
            }
            // Now interpolate to get coefficients
            // ... (standard Lagrange interpolation)
        }

        // ══════════════════════════════════════════════════════════
        // Verification Helpers
        // ══════════════════════════════════════════════════════════

        /// Check sum property: round(0) + round(1) == claimed
        pub fn checkSum(
            comptime degree: comptime_int,
            coeffs: Coeffs(degree),
            claimed: F,
        ) bool {
            const at_0 = coeffs[0];
            const at_1 = evalAt(degree, coeffs, F.one);
            return at_0.add(at_1).eq(claimed);
        }

        /// Evaluate round polynomial at point using Horner's method
        pub fn evalAt(
            comptime degree: comptime_int,
            coeffs: Coeffs(degree),
            point: F,
        ) F {
            var result = coeffs[degree];
            comptime var i = degree;
            inline while (i > 0) {
                i -= 1;
                result = result.mul(point).add(coeffs[i]);
            }
            return result;
        }

        // ══════════════════════════════════════════════════════════
        // Parallel Chunk Operations (for async runtime)
        // ══════════════════════════════════════════════════════════

        /// Compute partial round coefficients for a chunk
        /// Used by async runtime to parallelize large polynomials
        pub fn computeRoundChunk(
            comptime degree: comptime_int,
            buffer: [*]const F,
            total_len: usize,
            chunk_offset: usize,
            chunk_len: usize,
        ) Coeffs(degree) {
            // Same as computeRound but only processes [offset, offset+chunk_len)
        }

        /// Reduce partial coefficients by summing
        pub fn reduceCoeffs(
            comptime degree: comptime_int,
            partials: []const Coeffs(degree),
        ) Coeffs(degree) {
            var result = std.mem.zeroes(Coeffs(degree));
            for (partials) |partial| {
                inline for (0..degree + 1) |k| {
                    result[k] = result[k].add(partial[k]);
                }
            }
            return result;
        }
    };
}
```

---

## Protocol Integration

### Strategy: Type Bundle (No Methods)

Strategy bundles types for a specific sumcheck configuration.
Clarifies which field is used where.

```zig
pub fn Strategy(
    comptime BaseF: type,
    comptime ExtF: type,
    comptime degree: comptime_int,
    comptime num_polys: comptime_int,
) type {
    comptime {
        // Verify ExtF is an extension of BaseF
        if (@hasDecl(ExtF, "BaseField")) {
            if (ExtF.BaseField != BaseF) {
                @compileError("ExtF.BaseField must equal BaseF");
            }
        }
    }

    return struct {
        // Field types
        pub const Base = BaseF;
        pub const Ext = ExtF;

        // Parameters
        pub const Degree = degree;
        pub const NumPolys = num_polys;

        // s6k instantiations for each field
        pub const BaseOps = s6k(BaseF);
        pub const ExtOps = s6k(ExtF);

        // Coefficient types (degree+1 for standard, degree+2 after Gruen)
        pub const BaseCoeffs = [degree + 1]BaseF;
        pub const ExtCoeffs = [degree + 1]ExtF;
        pub const FullCoeffs = [degree + 2]ExtF;  // After Gruen linear factor

        // Accumulator types
        pub const Accumulators = BaseOps.AccumulatorTable;      // Alg 4: Base field
        pub const EqAccumulators = ExtOps.EqAccumulatorTable;   // Alg 6: Ext field

        // Buffer sizes
        pub fn polyBufferSize(len: usize) usize {
            return num_polys * len;
        }

        pub fn extBufferSize(base_len: usize) usize {
            return num_polys * (base_len / 2);
        }

        /// Cost model for algorithm selection
        pub const Cost = struct {
            ss: usize,  // Base × Base
            sl: usize,  // Ext × Base (via mulBase)
            ll: usize,  // Ext × Ext
        };
    };
}
```

### Example: Basefold Prover

```zig
const S = Strategy(M31, M31Ext3, 2, 2);

pub const BasefoldProver = struct {
    // Buffers (caller allocated, contiguous)
    base_buffer: []S.Base,   // 2 polys × 2^n elements
    ext_buffer: []S.Ext,     // 2 polys × 2^{n-1} elements

    // State
    len: usize,
    round: usize,
    transcript: *Transcript,

    pub fn prove(self: *BasefoldProver) !Proof {
        const num_vars = std.math.log2(self.len);

        // Round 0: BaseF
        {
            // Protocol-specific: commit before round
            const commitment = try self.commitMerkle(self.basePolySlice(0));
            self.transcript.absorb(commitment);

            // s6k: compute round polynomial
            const coeffs = S.BaseS6k.computeRound(
                S.Degree,
                self.base_buffer.ptr,
                self.len,
            );

            // Transcript
            self.transcript.absorbCoeffs(S.Base, &coeffs);
            const challenge = self.transcript.squeeze(S.Ext);

            // s6k: fold base → ext
            self.len = S.BaseS6k.foldToExt(
                S.Ext,
                S.NumPolys,
                self.base_buffer.ptr,
                self.len,
                challenge,
                self.ext_buffer.ptr,
            );
            self.round = 1;
        }

        // Rounds 1..n: ExtF
        while (self.round < num_vars) : (self.round += 1) {
            // Protocol-specific: commit
            const commitment = try self.commitMerkle(self.extPolySlice(0));
            self.transcript.absorb(commitment);

            // s6k: compute round polynomial
            const coeffs = S.ExtS6k.computeRound(
                S.Degree,
                self.ext_buffer.ptr,
                self.len,
            );

            // Transcript
            self.transcript.absorbCoeffs(S.Ext, &coeffs);
            const challenge = self.transcript.squeeze(S.Ext);

            // s6k: fold in-place
            self.len = S.ExtS6k.foldInPlace(
                S.NumPolys,
                self.ext_buffer.ptr,
                self.len,
                challenge,
            );
        }

        return self.buildProof();
    }

    fn basePolySlice(self: *BasefoldProver, idx: usize) []S.Base {
        return self.base_buffer[idx * self.len ..][0..self.len];
    }

    fn extPolySlice(self: *BasefoldProver, idx: usize) []S.Ext {
        return self.ext_buffer[idx * self.len ..][0..self.len];
    }
};
```

### Example: Zerocheck with Algorithm 6

```zig
const S = Strategy(M31, M31Ext3, 2, 2);
const L0 = 5;  // Rounds using SVO before switching to standard

pub const ZerocheckProver = struct {
    // Polynomial buffer (base field initially)
    poly_buffer: []S.Base,
    ext_buffer: []S.Ext,
    len: usize,

    // Eq-weighted accumulator table (EXTENSION field for Alg 6)
    accum_table: []S.Ext,
    accum_offsets: [L0 + 1]usize,

    // Challenge tensor R_i (extension field)
    challenge_tensor: ChallengeTensor(S.Ext, S.Degree, L0),

    // Split eq tables (extension field)
    eq_L: []S.Ext,
    eq_R: []S.Ext,
    eq_L_len: usize,
    eq_R_len: usize,

    // Verifier challenges w (extension field)
    w: []S.Ext,

    // Gruen linear factor state
    linear_state: S.ExtOps.LinearFactorState,

    // Current claimed sum
    claimed_sum: S.Ext,

    transcript: *Transcript,

    pub fn prove(self: *ZerocheckProver) !Proof {
        const num_vars = std.math.log2(self.len);

        // Phase 0: Precompute eq-weighted accumulators (Algorithm 6)
        // Note: Using precomputeEqAccumulators, not precomputeAccumulators
        S.BaseOps.precomputeEqAccumulators(
            S.Ext,
            S.Degree,
            self.poly_buffer.ptr,
            self.len,
            L0,
            self.eq_L.ptr,
            self.eq_L_len,
            self.eq_R.ptr,
            self.eq_R_len,
            self.accum_table.ptr,
            &self.accum_offsets,
        );

        // Initialize Gruen linear factor state
        self.linear_state = S.ExtOps.LinearFactorState.init(self.w[0]);

        // Phase 1: Accumulator-based rounds (1..L0)
        self.challenge_tensor.init();

        for (1..L0 + 1) |round| {
            const accum_offset = self.accum_offsets[round];
            const num_v = std.math.pow(usize, S.Degree + 1, round - 1);

            // s6k: compute t_i(u) from eq-weighted accumulators (ll dot product)
            const t_partial = S.ExtOps.computeRoundFromEqAccumulators(
                S.Degree,
                self.challenge_tensor.ptr(),
                self.challenge_tensor.len,
                self.accum_table.ptr + accum_offset,
                num_v,
            );

            // Gruen: recover t_i(1) from sum property
            const linear_factor = self.linear_state.linearFactor();
            const t_at_1 = S.ExtOps.recoverT1(t_partial[0], linear_factor, self.claimed_sum);

            // Build full t_i coefficients via interpolation
            const t_coeffs = S.ExtOps.interpolateT(S.Degree, t_partial, t_at_1);

            // Convert t_i to s_i: s_i(X) = l_i(X) · t_i(X)
            const s_coeffs = S.ExtOps.applyLinearFactor(S.Degree, t_coeffs, linear_factor);

            // Transcript
            self.transcript.absorbCoeffs(S.Ext, &s_coeffs);
            const challenge = self.transcript.squeeze(S.Ext);

            // Update claimed sum: C_i = s_i(r_i)
            self.claimed_sum = S.ExtOps.evalAt(S.Degree + 1, s_coeffs, challenge);

            // Update linear factor state for next round
            if (round < num_vars) {
                self.linear_state.update(self.w[round], challenge);
            }

            // Extend tensor: R_{i+1} = R_i ⊗ (L_k(r_i))_k
            const basis = S.ExtOps.lagrangeBasis(S.Degree, challenge);
            self.challenge_tensor.len = S.ExtOps.extendTensor(
                S.Degree,
                self.challenge_tensor.ptr(),
                self.challenge_tensor.len,
                basis,
                self.challenge_tensor.ptr(),
            );
        }

        // Phase 2: Materialize folded state from tensor
        self.materializeFromTensor();

        // Phase 3: Standard rounds with split eq (L0+1..num_vars)
        for (L0 + 1..num_vars) |round| {
            // s6k: compute t_i with split eq (returns d values, not d+1)
            const t_partial = S.ExtOps.computeRoundSplitEq(
                S.Degree,
                self.ext_buffer.ptr,
                self.len,
                self.eq_L.ptr,
                self.eq_L_len,
                self.eq_R.ptr,
                self.eq_R_len,
            );

            // Gruen: recover t_i(1) and convert to s_i
            const linear_factor = self.linear_state.linearFactor();
            const t_at_1 = S.ExtOps.recoverT1(t_partial[0], linear_factor, self.claimed_sum);
            const t_coeffs = S.ExtOps.interpolateT(S.Degree, t_partial, t_at_1);
            const s_coeffs = S.ExtOps.applyLinearFactor(S.Degree, t_coeffs, linear_factor);

            self.transcript.absorbCoeffs(S.Ext, &s_coeffs);
            const challenge = self.transcript.squeeze(S.Ext);

            self.claimed_sum = S.ExtOps.evalAt(S.Degree + 1, s_coeffs, challenge);

            if (round < num_vars) {
                self.linear_state.update(self.w[round], challenge);
            }

            // s6k: fold polynomials
            self.len = S.ExtOps.foldInPlace(
                S.NumPolys,
                self.ext_buffer.ptr,
                self.len,
                challenge,
            );

            // s6k: fold eq_L table (eq_R stays fixed after L0)
            if (round < num_vars / 2) {
                self.eq_L_len = S.ExtOps.foldEqTable(self.eq_L.ptr, self.eq_L_len, challenge);
            }
        }

        return self.buildProof();
    }
};

fn ChallengeTensor(comptime F: type, comptime degree: comptime_int, comptime max_rounds: comptime_int) type {
    const max_len = std.math.pow(usize, degree + 1, max_rounds - 1);

    return struct {
        data: [max_len]F = undefined,
        len: usize = 0,

        pub fn init(self: *@This()) void {
            self.data[0] = F.one;
            self.len = 1;
        }

        pub fn ptr(self: *@This()) [*]F {
            return &self.data;
        }
    };
}
```

---

## Algorithm Support Matrix

| Algorithm | Use Case | s6k Primitives Used | Accum Field |
|-----------|----------|---------------------|-------------|
| **Alg 1** | Standard, memory available | `computeRound`, `foldInPlace`, `foldToExt` | N/A |
| **Alg 2** | Memory-constrained | `computeRoundWithEq`, `buildEqTable`, delayed fold | N/A |
| **Alg 3** | (Superseded by Alg 4) | Not implemented | - |
| **Alg 4** | Small values, Toom-Cook (no eq) | `precomputeAccumulators`, `computeRoundFromAccumulators`, `lagrangeBasis`, `extendTensor` | **Base** |
| **Alg 5** | With eq poly (Gruen) | `computeRoundSplitEq`, `recoverT1`, `applyLinearFactor`, `LinearFactorState` | N/A |
| **Alg 6** | Alg 4 + Alg 5 combined | `precomputeEqAccumulators`, `computeRoundFromEqAccumulators`, + all Gruen primitives | **Ext** |

**Key insight**: Algorithm 4 accumulators are in BASE field (ss products), Algorithm 6 accumulators are in EXTENSION field (eq-weighted).

### Choosing an Algorithm

```
                    ┌─────────────────────┐
                    │ Does sumcheck have  │
                    │ ẽq(w,x) factor?     │
                    └─────────┬───────────┘
                              │
              ┌───────────────┴───────────────┐
              │ NO                            │ YES
              ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐
    │ Memory-         │             │ Use Gruen opt   │
    │ constrained?    │             │ (factor out eq) │
    └────────┬────────┘             └────────┬────────┘
             │                               │
    ┌────────┴────────┐             ┌────────┴────────┐
    │ NO      │ YES   │             │                 │
    ▼         ▼       │             ▼                 │
  Alg 1    Alg 2      │    ┌─────────────────┐        │
                      │    │ Small values?   │        │
                      │    │ (base field)    │        │
                      │    └────────┬────────┘        │
                      │             │                 │
                      │    ┌────────┴────────┐        │
                      │    │ NO      │ YES   │        │
                      │    ▼         ▼       │        │
                      │  Alg 5    Alg 6      │        │
                      │                      │        │
                      └──────────────────────┘        │
```

---

## Async Runtime Compatibility

s6k primitives map directly to async jobs:

### Job Types

```zig
pub const Job = union(enum) {
    // Standard operations
    compute_round: ComputeRoundJob,
    fold_in_place: FoldInPlaceJob,
    fold_to_ext: FoldToExtJob,

    // Eq operations
    build_eq_table: BuildEqTableJob,
    fold_eq_table: FoldEqTableJob,
    compute_round_split_eq: ComputeRoundSplitEqJob,

    // Accumulator operations
    precompute_accumulators: PrecomputeAccumulatorsJob,
    compute_round_from_accumulators: ComputeRoundFromAccumulatorsJob,

    // Tensor operations
    extend_tensor: ExtendTensorJob,

    // Parallel variants
    compute_round_chunk: ComputeRoundChunkJob,
    reduce_coeffs: ReduceCoeffsJob,

    // Transcript / IO
    transcript_absorb: TranscriptAbsorbJob,
    transcript_squeeze: TranscriptSqueezeJob,
    merkle_commit: MerkleCommitJob,
};

pub const ComputeRoundJob = struct {
    buffer: BufferHandle,
    len: u32,
    degree: u8,
    output: PromiseHandle,
};
```

### Job Executor

```zig
fn executeJob(self: *Runtime, job: Job) void {
    switch (job) {
        .compute_round => |j| {
            const buf = self.buffers.getSlice(j.buffer);
            const coeffs = switch (j.degree) {
                1 => s6k(F).computeRound(1, buf.ptr, j.len),
                2 => s6k(F).computeRound(2, buf.ptr, j.len),
                3 => s6k(F).computeRound(3, buf.ptr, j.len),
                else => unreachable,
            };
            self.promises.fulfill(j.output, coeffs);
        },
        // ... other jobs
    }
}
```

### Parallelism Mapping

| Sync Pattern | Async Pattern |
|--------------|---------------|
| `computeRound(buf, len)` | Single `compute_round` job |
| `computeRound` on large poly | Multiple `compute_round_chunk` + `reduce_coeffs` |
| `precomputeAccumulators` | Parallel chunks + reduce |
| Multiple independent sumchecks | Parallel job graphs |
| Compute + commit | Pipeline parallelism (overlap) |

### Buffer Handles

```zig
pub const BufferHandle = enum(u32) { _ };

pub const BufferRegistry = struct {
    // SoA layout for buffer metadata
    ptrs: []?[*]u8,
    lens: []u32,
    elem_sizes: []u8,
    layouts: []BufferLayout,

    pub const BufferLayout = enum {
        poly_buffer,
        accumulator_table,
        eq_table,
        challenge_tensor,
    };

    pub fn alloc(self: *BufferRegistry, comptime T: type, len: usize) !BufferHandle;
    pub fn getSlice(self: *BufferRegistry, handle: BufferHandle) Slice;
    pub fn free(self: *BufferRegistry, handle: BufferHandle) void;
};
```

---

## Memory Budget Analysis

For a sumcheck with `n = 2^20` elements, `d = 2`, `ℓ₀ = 5`:

| Component | Size (M31) | Size (M31Ext3) | Notes |
|-----------|------------|----------------|-------|
| Polynomial buffer | 8 MB | 24 MB | 2 polys × 2^20 × 4B/12B |
| Accumulator table | - | 3 KB | 242 × 12B |
| Challenge tensor | - | 1 KB | 81 × 12B (stack) |
| eq_L table | - | 12 KB | 2^10 × 12B |
| eq_R table | - | 12 KB | 2^10 × 12B |
| **Total** | **8 MB** | **~24 MB** | Dominated by polys |

After first fold (base → ext):
- Base buffer: no longer needed (can free)
- Ext buffer: 2 polys × 2^19 × 12B = 12 MB

---

## Implementation Plan

### Phase 1: Core Primitives
1. `computeRound` (scalar, then SIMD)
2. `foldInPlace`, `foldToExt`
3. `evalAt`, `checkSum`
4. Basic tests against naive implementation

### Phase 2: Eq Operations
1. `buildEqTable`, `foldEqTable`
2. `computeRoundWithEq`
3. `computeRoundSplitEq` (Gruen optimization)

### Phase 3: Accumulator Operations (Algorithms 4/6)
1. `precomputeAccumulators`
2. `computeRoundFromAccumulators`
3. `lagrangeBasis`, `extendTensor`

### Phase 4: Protocol Integration
1. Basefold prover using s6k
2. Zerocheck prover with Algorithm 5/6
3. Benchmarks comparing algorithms

### Phase 5: Async Runtime
1. Buffer registry with handles
2. Job types and executor
3. Chunk operations for parallelism
4. Job graph builder

### Phase 6: Backends
1. SIMD optimization (AVX2/AVX-512/NEON)
2. Thread pool for chunk parallelism
3. GPU backend (Metal/CUDA) [future]

---

## Module Structure

```
src/
├── s6k/
│   ├── core.zig           # s6k(F) - main primitive interface
│   ├── round.zig          # computeRound implementations
│   ├── fold.zig           # fold operations
│   ├── eq.zig             # eq table operations
│   ├── accumulator.zig    # Algorithm 4 accumulators (Base field)
│   ├── eq_accumulator.zig # Algorithm 6 accumulators (Ext field)
│   ├── tensor.zig         # Lagrange/tensor operations
│   ├── gruen.zig          # LinearFactorState, recoverT1, applyLinearFactor
│   ├── verify.zig         # checkSum, evalAt
│   ├── simd/
│   │   ├── avx2.zig
│   │   ├── avx512.zig
│   │   └── neon.zig
│   └── strategy.zig       # Strategy type bundle
├── runtime/               # [Future] Async runtime
│   ├── job.zig
│   ├── buffer.zig
│   ├── executor.zig
│   └── graph.zig
└── protocols/
    ├── basefold.zig       # Uses s6k (Alg 1)
    ├── zerocheck.zig      # Uses s6k (Alg 5/6)
    └── gkr.zig            # Uses s6k
```

---

## References

- [Speeding Up Sum-Check Proving](https://eprint.iacr.org/2025/1117.pdf) - Algorithms 1-6
- [Optimizing the Sumcheck Protocol](https://hackmd.io/@tcoratger/SJjJBfWWlg) - Implementation notes
- S6K_ASYNC.md - Async runtime evolution

---

## Appendix: Field Operation Costs

Understanding multiplication costs drives algorithm selection.

### Field Interface Mapping (src/fields/field.zig)

| Cost Type | Symbol | Field Interface | Example |
|-----------|--------|-----------------|---------|
| ss | Base × Base | `Base.mul(Base, Base)` | M31 × M31 |
| sl | Ext × Base | `Ext.mulBase(Ext, Base)` | M31Ext3 × M31 |
| ll | Ext × Ext | `Ext.mul(Ext, Ext)` | M31Ext3 × M31Ext3 |

### Relative Costs

| Operation | Symbol | Example | Relative Cost |
|-----------|--------|---------|---------------|
| Base × Base | ss | M31 × M31 | 1× |
| Base × Ext | sl | M31 × M31Ext3 | ~3× |
| Ext × Ext | ll | M31Ext3 × M31Ext3 | ~9× |

### Batch Operations

| Operation | Field Interface | Use Case |
|-----------|-----------------|----------|
| Base dot product | `Base.dotProduct([]Base, []Base)` | Accumulator sums (ss) |
| Mixed dot product | `Ext.Batch.dotProductMixed([]Base)` | Tensor × accumulators (sl) |
| Ext dot product | `Ext.dotProduct([]Ext, []Ext)` | Eq-weighted accumulators (ll) |

Algorithms 3-6 optimize by:
1. Keeping data in base field longer (fewer ll)
2. Precomputing expensive products (amortize ll)
3. Using structure (eq factorization) to reduce degree
4. Using `mulBase` for sl operations instead of full ll multiply
