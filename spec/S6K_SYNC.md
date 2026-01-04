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

### AccumulatorTable: Precomputed Sums for Algorithms 4/6

```
Layout: For each round i ∈ [1, ℓ₀], for each u ∈ Û_d:
        A_i(v, u) for v ∈ [0, (d+1)^{i-1})

        Optimized for inner product: ⟨R_i, A_i(·, u)⟩

Memory: [round_1_data | round_2_data | ... | round_ℓ₀_data]

        Where round_i_data = [A_i(·,0) | A_i(·,1) | ... | A_i(·,d-1)]
        And   A_i(·,u) = [A_i(0,u), A_i(1,u), ..., A_i(num_v-1,u)]
```

**Size**: `Σᵢ (d+1)^{i-1} × d` field elements

For d=2, ℓ₀=5: `(1 + 3 + 9 + 27 + 81) × 2 = 242` elements

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
```

For d=2, ℓ₀=5: max 81 elements (stack-allocatable)

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

        /// Compute round with split eq (Gruen/Algorithm 5):
        /// Σ_{x_R} eq_R[x_R] · Σ_{x_L} eq_L[x_L] · ∏ⱼ (loⱼ + t·δⱼ)
        pub fn computeRoundSplitEq(
            comptime degree: comptime_int,
            buffer: [*]const F,
            len: usize,
            eq_L: [*]const F,
            eq_L_len: usize,
            eq_R: [*]const F,
            eq_R_len: usize,
        ) Coeffs(degree) {
            // Two-level summation for cache efficiency
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
        // Accumulator Operations (Algorithms 4/6)
        // ══════════════════════════════════════════════════════════

        /// Precompute accumulator table for small-value optimization
        ///
        /// A_i(v, u) = Σ_{x'} [Σ_{y': Σy'_k = v} ∏_k p_k(y'_k, u, x')]
        ///
        /// Output layout: see AccumulatorTable documentation
        pub fn precomputeAccumulators(
            comptime degree: comptime_int,
            buffer: [*]const F,
            len: usize,
            num_rounds: usize,
            out: [*]F,
            out_offsets: [*]usize,  // Filled with round offsets
        ) void {
            // See Algorithm 4 in literature
        }

        /// Compute round polynomial from precomputed accumulators
        /// round(u) = ⟨R_i, A_i(·, u)⟩
        pub fn computeRoundFromAccumulators(
            comptime degree: comptime_int,
            challenge_tensor: [*]const F,
            tensor_len: usize,
            accumulators: [*]const F,  // A_i(·, u) for all u
            num_v: usize,              // (d+1)^{i-1}
        ) Coeffs(degree) {
            var coeffs: Coeffs(degree) = undefined;
            inline for (0..degree) |u| {
                const accum_u = accumulators[u * num_v ..][0..num_v];
                coeffs[u] = dotProduct(challenge_tensor[0..tensor_len], accum_u);
            }
            // Derive coeffs[degree] from sum property if needed
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

Strategy provides type aliases for a specific sumcheck configuration:

```zig
pub fn Strategy(
    comptime BaseF: type,
    comptime ExtF: type,
    comptime degree: comptime_int,
    comptime num_polys: comptime_int,
) type {
    return struct {
        // Field types
        pub const Base = BaseF;
        pub const Ext = ExtF;

        // Parameters
        pub const Degree = degree;
        pub const NumPolys = num_polys;

        // s6k instantiations
        pub const BaseS6k = s6k(BaseF);
        pub const ExtS6k = s6k(ExtF);

        // Coefficient types
        pub const BaseCoeffs = BaseS6k.Coeffs(degree);
        pub const ExtCoeffs = ExtS6k.Coeffs(degree);

        // Buffer sizes
        pub fn polyBufferSize(len: usize) usize {
            return num_polys * len;
        }

        pub fn extBufferSize(base_len: usize) usize {
            return num_polys * (base_len / 2);
        }
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
    // Polynomial buffer
    poly_buffer: []S.Base,
    ext_buffer: []S.Ext,
    len: usize,

    // Accumulator table (precomputed)
    accum_table: []S.Ext,
    accum_offsets: [L0 + 1]usize,

    // Challenge tensor (stack allocated)
    challenge_tensor: ChallengeTensor(S.Ext, S.Degree, L0),

    // Split eq tables
    eq_L: []S.Ext,
    eq_R: []S.Ext,

    transcript: *Transcript,

    pub fn prove(self: *ZerocheckProver) !Proof {
        const num_vars = std.math.log2(self.len);

        // Phase 0: Precompute accumulators
        S.BaseS6k.precomputeAccumulators(
            S.Degree,
            self.poly_buffer.ptr,
            self.len,
            L0,
            self.accum_table.ptr,
            &self.accum_offsets,
        );

        // Build eq tables
        S.ExtS6k.buildEqTable(self.w_L.ptr, num_vars / 2, self.eq_L.ptr);
        S.ExtS6k.buildEqTable(self.w_R.ptr, num_vars / 2, self.eq_R.ptr);

        // Phase 1: Accumulator-based rounds (1..L0)
        self.challenge_tensor.init();

        for (1..L0 + 1) |round| {
            const accum_offset = self.accum_offsets[round];
            const num_v = std.math.pow(usize, S.Degree + 1, round - 1);

            // s6k: compute from accumulators
            const t_coeffs = S.ExtS6k.computeRoundFromAccumulators(
                S.Degree,
                self.challenge_tensor.ptr(),
                self.challenge_tensor.len,
                self.accum_table.ptr + accum_offset,
                num_v,
            );

            // Gruen: reconstruct s_i from t_i and linear factor
            const coeffs = self.reconstructWithLinearFactor(t_coeffs, round);

            // Transcript
            self.transcript.absorbCoeffs(S.Ext, &coeffs);
            const challenge = self.transcript.squeeze(S.Ext);

            // Extend tensor
            const basis = S.ExtS6k.lagrangeBasis(S.Degree, challenge);
            self.challenge_tensor.len = S.ExtS6k.extendTensor(
                S.Degree,
                self.challenge_tensor.ptr(),
                self.challenge_tensor.len,
                basis,
                self.challenge_tensor.ptr(),
            );
        }

        // Phase 2: Materialize folded state from accumulators
        self.materializeFromAccumulators();

        // Phase 3: Standard rounds with split eq (L0+1..num_vars)
        for (L0 + 1..num_vars) |round| {
            // s6k: compute with split eq
            const t_coeffs = S.ExtS6k.computeRoundSplitEq(
                S.Degree,
                self.ext_buffer.ptr,
                self.len,
                self.eq_L.ptr,
                self.eq_L_len,
                self.eq_R.ptr,
                self.eq_R_len,
            );

            const coeffs = self.reconstructWithLinearFactor(t_coeffs, round);

            self.transcript.absorbCoeffs(S.Ext, &coeffs);
            const challenge = self.transcript.squeeze(S.Ext);

            // s6k: fold
            self.len = S.ExtS6k.foldInPlace(
                S.NumPolys,
                self.ext_buffer.ptr,
                self.len,
                challenge,
            );

            // s6k: fold eq tables
            self.eq_L_len = S.ExtS6k.foldEqTable(self.eq_L.ptr, self.eq_L_len, challenge);
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

| Algorithm | Use Case | s6k Primitives Used |
|-----------|----------|---------------------|
| **Alg 1** | Standard, memory available | `computeRound`, `foldInPlace` |
| **Alg 2** | Memory-constrained | `computeRound`, `buildEqTable`, delayed fold |
| **Alg 3** | Small values, fewer ll mults | `precomputeAccumulators` (eq-based) |
| **Alg 4** | Small values, Toom-Cook | `precomputeAccumulators`, `lagrangeBasis` |
| **Alg 5** | With eq poly (Gruen) | `computeRoundSplitEq`, split eq tables |
| **Alg 6** | Alg 4 + Alg 5 combined | All accumulator + split eq primitives |

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
│   ├── accumulator.zig    # accumulator operations (Alg 4/6)
│   ├── tensor.zig         # Lagrange/tensor operations
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
    ├── basefold.zig       # Uses s6k
    ├── zerocheck.zig      # Uses s6k
    └── gkr.zig            # Uses s6k
```

---

## References

- [Speeding Up Sum-Check Proving](https://eprint.iacr.org/2025/1117.pdf) - Algorithms 1-6
- [Optimizing the Sumcheck Protocol](https://hackmd.io/@tcoratger/SJjJBfWWlg) - Implementation notes
- S6K_ASYNC.md - Async runtime evolution

---

## Appendix: Field Operation Costs

Understanding multiplication costs drives algorithm selection:

| Operation | Symbol | Example | Relative Cost |
|-----------|--------|---------|---------------|
| Base × Base | ss | M31 × M31 | 1× |
| Base × Ext | sl | M31 × M31Ext3 | ~3× |
| Ext × Ext | ll | M31Ext3 × M31Ext3 | ~9× |

Algorithms 3-6 optimize by:
1. Keeping data in base field longer (fewer ll)
2. Precomputing expensive products (amortize ll)
3. Using structure (eq factorization) to reduce degree
