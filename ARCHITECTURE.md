# libzero Architecture

A zero-knowledge proving library built on sumcheck protocols, designed for data-oriented efficiency.

## Vision

libzero aims to be a competitive client-side prover by leveraging:

- **Sumcheck-based IOPs** over NTT-heavy approaches for cache-friendly memory access
- **Data-oriented design** principles throughout the stack
- **Zig's low-level control** for explicit memory layout and SIMD optimization
- **Streaming architecture** inspired by SCRIBE for O(log N) memory usage

## Design Principles

### 1. Sequential Memory Access

Random I/O is dramatically slower than sequential I/O. Every algorithm choice prioritizes linear memory traversal:

- Sumcheck naturally halves data each round (sequential reads)
- Avoid NTT butterfly patterns (strided access)
- Process data in cache-line-sized blocks

### 2. Delayed Reduction

Field arithmetic accumulates in wider integers, reducing only when necessary:

- Sum millions of 31-bit elements in a 64-bit accumulator
- Single reduction at the end instead of per-operation
- Enables better instruction pipelining

### 3. Batch Over Scalar

Operations work on slices, not individual elements:

- `sumSlice`, `foldSlice` instead of element-by-element loops
- Enables SIMD without changing call sites
- Reduces function call overhead

### 4. Compile-Time Generics

Monomorphization over runtime polymorphism:

- `fn Sumcheck(comptime F: type)` generates specialized code per field
- Zero vtable overhead
- Enables field-specific optimizations

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           libzero                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │   Slot A    │    │   Slot B    │    │   Slot C    │             │
│  │   Field     │    │    PCS      │    │    IOP      │             │
│  │             │    │             │    │             │             │
│  │ Mersenne31  │    │    FRI      │    │  Sumcheck   │             │
│  │ Goldilocks  │    │  Brakedown  │    │  Zerocheck  │             │
│  │ BinaryTower │    │    KZG      │    │  GKR/LogUp  │             │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘             │
│         │                  │                  │                     │
│         └──────────────────┼──────────────────┘                     │
│                            │                                        │
│                   ┌────────▼────────┐                               │
│                   │   Polynomial    │                               │
│                   │  Abstraction    │                               │
│                   │                 │                               │
│                   │  Multilinear    │                               │
│                   │  Virtual Poly   │                               │
│                   └────────┬────────┘                               │
│                            │                                        │
│                   ┌────────▼────────┐                               │
│                   │   Transcript    │                               │
│                   │  (Fiat-Shamir)  │                               │
│                   └─────────────────┘                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Slot A: Field Arithmetic

The atomic unit of computation. All higher layers are generic over the field.

### Interface

```zig
pub fn FieldInterface(comptime F: type) type {
    // Required:
    F.MODULUS          // The prime modulus
    F.ENCODED_SIZE     // Bytes when serialized
    F.zero, F.one      // Additive/multiplicative identities
    
    F.add(a, b) → F    // Addition
    F.sub(a, b) → F    // Subtraction  
    F.mul(a, b) → F    // Multiplication
    F.neg(a) → F       // Negation
    F.inv(a) → F       // Multiplicative inverse
    
    F.eql(a, b) → bool // Equality
    F.isZero(a) → bool // Zero check
    
    F.toBytes(a) → [N]u8      // Serialization
    F.fromBytes(b) → ?F       // Deserialization
    F.fromU64(x) → F          // Construction
    
    // Batch operations (DOD):
    F.sumSlice([]F) → F                    // Sum with delayed reduction
    F.sumHalves([]F) → {F, F}              // Sum first/second half
    F.foldSlice([]F, r) → []F              // Linear combination fold
}
```

### Implementations

| Field | Size | Modulus | Reduction | Use Case |
|-------|------|---------|-----------|----------|
| Mersenne31 | 31-bit | 2³¹ - 1 | Shift + mask | Primary (cache efficient) |
| Goldilocks | 64-bit | 2⁶⁴ - 2³² + 1 | EPSILON trick | Plonky3 compatibility |
| BinaryTower | 1-128 bit | Extension field | XOR-based | Binius (future) |

### Memory Layout

```
Mersenne31 slice (cache-optimized):
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ e0 │ e1 │ e2 │ e3 │ e4 │ e5 │ e6 │ e7 │  ← 32 bytes = 1/2 cache line
└────┴────┴────┴────┴────┴────┴────┴────┘
  4B   4B   4B   4B   4B   4B   4B   4B

Goldilocks slice:
┌────────┬────────┬────────┬────────┐
│   e0   │   e1   │   e2   │   e3   │  ← 32 bytes = 1/2 cache line
└────────┴────────┴────────┴────────┘
    8B       8B       8B       8B
```

---

## Slot B: Polynomial Commitment Scheme (PCS)

Compresses polynomial evaluations into short commitments with opening proofs.

### Interface

```zig
pub fn PCS(comptime F: type) type {
    return struct {
        const Commitment: type;
        const Opening: type;
        const Proof: type;
        
        // Commit to polynomial evaluations
        fn commit(evals: []const F) Commitment;
        
        // Open at a point, produce proof
        fn open(commitment: Commitment, point: []const F, eval: F) Proof;
        
        // Verify opening
        fn verify(commitment: Commitment, point: []const F, eval: F, proof: Proof) bool;
    };
}
```

### Implementations (Future)

| PCS | Commit | Proof Size | Verify | Setup |
|-----|--------|------------|--------|-------|
| FRI | O(n log n) | O(log² n) | O(log² n) | None |
| Brakedown | O(n) | O(√n) | O(√n) | None |
| KZG | O(n) | O(1) | O(1) | Trusted |

### Integration Point

The sumcheck protocol produces a final evaluation claim `(point, value)`. The PCS proves this claim is consistent with the committed polynomial:

```
Sumcheck output: "P(r₁, r₂, ..., rₙ) = v"
                          ↓
PCS.open(commitment, [r₁, r₂, ..., rₙ], v) → proof
```

---

## Slot C: Interactive Oracle Proof (IOP)

The core proving logic. Sumcheck is the foundation.

### Sumcheck Protocol

Proves: `∑_{x ∈ {0,1}ⁿ} P(x) = H`

```
Round 1:  P(X, x₂, ..., xₙ)     → s₁(X)    [sum over x₂...xₙ]
          Verifier sends r₁
          
Round 2:  P(r₁, X, x₃, ..., xₙ) → s₂(X)    [sum over x₃...xₙ]  
          Verifier sends r₂
          
  ...
  
Round n:  P(r₁, ..., rₙ₋₁, X)   → sₙ(X)    [no sum]
          Verifier sends rₙ
          
Final:    Check P(r₁, ..., rₙ) = sₙ(rₙ) via PCS
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Sumcheck Prover                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: evals[2ⁿ]     Multilinear polynomial evaluations        │
│                                                                 │
│  Round 1:                                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ sum₀ = Σ evals[0..half]     (first half)                │    │
│  │ sum₁ = Σ evals[half..n]     (second half)               │    │
│  │ s₁(X) = sum₀ + X·(sum₁ - sum₀)                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ↓                                     │
│  Receive challenge r₁ from transcript                           │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Fold: evals[i] = evals[i] + r₁·(evals[i+half] - evals[i])│   │
│  │ New size: 2ⁿ⁻¹                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ↓                                     │
│  Round 2: (repeat with smaller evals)                           │
│                          ...                                    │
│                           ↓                                     │
│  Output: [s₁, s₂, ..., sₙ], final_eval                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Usage

```
Round 1: Process 2ⁿ elements   → output 2ⁿ⁻¹
Round 2: Process 2ⁿ⁻¹ elements → output 2ⁿ⁻²
...
Round n: Process 2 elements    → output 1

Total memory: O(2ⁿ) - single allocation, reused each round
Peak memory:  O(2ⁿ) - no intermediate buffers needed
```

### Extensions

| Protocol | Purpose | Builds On |
|----------|---------|-----------|
| Zerocheck | Prove P(x) = 0 for all x ∈ {0,1}ⁿ | Sumcheck |
| GKR | Verify layered circuit computation | Sumcheck |
| LogUp | Lookup arguments | Sumcheck + grand products |
| Lasso | Lookup via sparse polynomials | Sumcheck |

---

## Polynomial Abstraction Layer

Bridges the gap between raw evaluations and protocol requirements.

### Multilinear Polynomial

```zig
pub fn MultilinearPoly(comptime F: type) type {
    return struct {
        evals: []F,        // Evaluations over {0,1}ⁿ
        num_vars: usize,   // n
        
        // Evaluation at arbitrary point
        fn evaluate(self, point: []const F) F;
        
        // Partial evaluation (bind first variable)
        fn bind(self, r: F) void;  // In-place, halves size
        
        // Batch operations for sumcheck
        fn sum(self) F;
        fn sumHalves(self) struct { F, F };
    };
}
```

### Virtual Polynomial (Future)

For constraint systems, we don't materialize `P(x) = A(x)·B(x) - C(x)`. Instead:

```zig
pub fn VirtualPoly(comptime F: type) type {
    return struct {
        // References to underlying polynomials
        components: []const *MultilinearPoly(F),
        
        // Evaluation function (computes on-the-fly)
        eval_fn: fn([]const F) F,
        
        // Sumcheck interface
        fn sumHalves(self) struct { F, F };
        fn bind(self, r: F) void;
    };
}
```

This enables:

- Memory efficiency (don't materialize product polynomials)
- Lazy evaluation (compute only what's needed)
- Composition (build complex constraints from simple parts)

---

## Transcript (Fiat-Shamir)

Transforms interactive protocols into non-interactive proofs.

```zig
pub fn Transcript(comptime F: type) type {
    return struct {
        state: Blake3,
        
        // Add data to transcript
        fn absorb(self, elem: F) void;
        fn absorbBytes(self, bytes: []const u8) void;
        
        // Extract challenge
        fn squeeze(self) F;
    };
}
```

### Security

- Domain separation via initial label
- All prover messages absorbed before challenges
- Deterministic: same inputs → same challenges

---

## Directory Structure

```
libzero/
├── build.zig
├── README.md
├── ARCHITECTURE.md
│
├── src/
│   ├── main.zig                 # Library entry point
│   │
│   ├── field/
│   │   ├── interface.zig        # Field trait definition
│   │   ├── mersenne31.zig       # Primary field
│   │   ├── goldilocks.zig       # Alternative field
│   │
│   ├── poly/
│   │   ├── multilinear.zig      # MLE representation
│   │   ├── virtual.zig          # Virtual polynomials
│   │   └── eq.zig               # Equality polynomial
│   │
│   ├── sumcheck/
│   │   ├── prover.zig           # Sumcheck prover
│   │   ├── verifier.zig         # Sumcheck verifier
│   │   └── protocol.zig         # Full Fiat-Shamir protocol
│   │
│   ├── transcript/
│   │   └── transcript.zig       # Fiat-Shamir transcript
│   │
│   └── pcs/                     # Future: Polynomial commitments
│       ├── interface.zig
│       ├── fri.zig
│       └── brakedown.zig
│
├── bench/
│   ├── field_bench.zig          # Field microbenchmarks
│   ├── sumcheck_bench.zig       # Sumcheck benchmarks
│   └── e2e_bench.zig            # End-to-end benchmarks
│
└── tests/
    ├── field_tests.zig
    ├── poly_tests.zig
    └── sumcheck_tests.zig
```

---

## Data-Oriented Patterns

### Pattern 1: Structure of Arrays

```zig
// Anti-pattern: Array of Structures
const BadPoly = struct {
    points: []struct { x: F, y: F, z: F },
};

// DOD: Structure of Arrays
const GoodPoly = struct {
    evals: []F,  // All evaluations contiguous
    // Metadata stored separately
};
```

### Pattern 2: Batch APIs

```zig
// Anti-pattern: Process one at a time
for (elements) |e| {
    result = result.add(e);
}

// DOD: Batch operation
result = F.sumSlices(elements);
```

### Pattern 3: In-Place Mutation

```zig
// Anti-pattern: Allocate new buffer each round
fn bind(self) NewPoly {
    var new = allocate(self.len / 2);
    // ...
    return new;
}

// DOD: Mutate in place
fn bind(self, r: F) void {
    // Fold into first half, update slice bounds
    self.evals = self.evals[0..self.evals.len / 2];
}
```

### Pattern 4: Delayed Reduction

```zig
// Anti-pattern: Reduce after every operation
var sum = F.zero;
for (slice) |e| {
    sum = sum.add(e);  // Reduces each iteration
}

// DOD: Accumulate wide, reduce once
var acc: u64 = 0;
for (slice) |e| {
    acc += e.value;    // No reduction
}
return F.reduce64(acc);  // Single reduction
```

---

## Future Optimizations

### Batched Sumcheck

When verifying multiple polynomial claims simultaneously, combine them into a single sumcheck:

**Problem:** k separate claims require k × n rounds

```
claim_1: Σ f_1(x) = c_1
claim_2: Σ f_2(x) = c_2
...
claim_k: Σ f_k(x) = c_k
```

**Solution:** Random linear combination into single claim

```
Verifier sends random α
Combined: Σ (f_1(x) + α·f_2(x) + α²·f_3(x) + ...) = c_1 + α·c_2 + α²·c_3 + ...
Single sumcheck: n rounds instead of k × n
```

**Optimal Data Layout:**

```zig
// Interleaved k-tuples for cache efficiency
// evals[i] = [f_0(x_i), f_1(x_i), ..., f_{k-1}(x_i)]
evals: [][k]Mersenne31   // 2^n groups of k elements

// Each iteration accesses all k values at same hypercube index
// k values contiguous → single cache line (for small k)
// Combination Σ αⁱ·f_i(x) is a dot product
```

**When to use:**

- GKR protocol (multiple layer claims)
- Batch verification (N proofs at once)
- Multi-instance proofs (N executions of same circuit)

**Trade-off with k:**

- Small k (2-8): k-tuple fits in cache line, interleaved wins
- Large k (100+): Tuple spans cache lines, separate buffers may be better

### Multi-threading

Sumcheck operations are embarrassingly parallel within each round:

```
┌─────────────────────────────────────────────────────────┐
│                    sumHalves (parallel)                  │
├─────────────────────────────────────────────────────────┤
│  Thread 0        Thread 1        Thread 2        Thread 3│
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐│
│  │chunk 0  │    │chunk 1  │    │chunk 2  │    │chunk 3  ││
│  │partial  │    │partial  │    │partial  │    │partial  ││
│  │sum      │    │sum      │    │sum      │    │sum      ││
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘│
│       └──────────────┴──────────────┴──────────────┘     │
│                          ↓                               │
│                   Combine partials                       │
│                   (single reduce)                        │
└─────────────────────────────────────────────────────────┘
```

**Parallelizable operations:**

- `sumSlices`: Each thread sums a chunk, combine at end
- `linearCombineBatch` (bind): Each thread processes a range
- `dotProduct`: Same pattern as sum

**Implementation approach:**

```zig
// Static thread pool (avoid allocation per call)
const ThreadPool = struct {
    workers: [NUM_THREADS]Worker,

    fn parallelSum(slices: []const Mersenne31) Mersenne31 {
        // Divide work, each thread returns partial sum
        // Main thread combines partials
    }
};
```

**Expected speedup:**

- n=24 (16M elements): ~4x with 8 cores (memory-bound)
- n=20 (1M elements): ~3x (starts fitting in L3)
- n=16 (64K elements): ~2x (L3 cache, less parallelism benefit)

**Memory considerations:**

- Each thread needs its own accumulator (no contention)
- Chunk boundaries should align to cache lines (64 bytes)
- For batched sumcheck: partition by hypercube index, not by polynomial

### SIMD Observations

From benchmarking on Apple Silicon:

| Operation | Explicit SIMD | LLVM Auto-vectorize | Winner |
|-----------|---------------|---------------------|--------|
| sumSlices | Slower (4x) | Faster | LLVM |
| linearCombineBatch | 1.3-1.5x faster | Baseline | Explicit |
| dotProduct | TBD | TBD | TBD |

**Key insight:** Simple reductions (sum) are memory-bound and LLVM auto-vectorizes well. Complex arithmetic (multiply + reduce) benefits from explicit SIMD because compute hides memory latency.

**Guidelines:**

- Trust LLVM for simple loops (sum, copy)
- Explicit SIMD for: multiply, field reduction, multi-operation kernels
- Always benchmark both approaches

---

## Performance Targets

### Field Operations (Mersenne31, single core)

| Operation | Target | Notes |
|-----------|--------|-------|
| add | > 500 M ops/sec | Should be ~1 cycle |
| mul | > 200 M ops/sec | 64-bit multiply + reduce |
| sumSlice (2²⁴) | < 20 ms | Memory-bound |
| foldSlice (2²⁴) | < 50 ms | Compute-bound |

### Sumcheck Prover (Mersenne31, single core)

| Size | Target | Jolt Reference |
|------|--------|----------------|
| 2¹⁶ | < 1 ms | ~0.5 ms |
| 2²⁰ | < 15 ms | ~8 ms |
| 2²⁴ | < 250 ms | ~150 ms |

### Memory

| Size | Footprint |
|------|-----------|
| 2²⁰ elements | 4 MB (Mersenne31) |
| 2²⁴ elements | 64 MB (Mersenne31) |
| 2²⁸ elements | 1 GB (Mersenne31) |

---

## Future Directions

### Phase 1: MVP (Current)

- [x] Field arithmetic (Mersenne31)
- [ ] Multilinear polynomial
- [ ] Sumcheck prover/verifier
- [ ] Benchmarks vs Jolt

### Phase 2: PCS Integration

<!-- - [ ] FRI commitment scheme -->
- [ ] Merkle tree (Blake3)
- [ ] End-to-end proof generation

### Phase 3: Advanced IOPs

- [ ] Zerocheck
- [ ] GKR protocol
- [ ] LogUp/Lasso lookups

### Phase 4: Optimizations

**SIMD Field Operations:**

- [x] `linearCombineBatch` - explicit SIMD (1.3-1.5x speedup)
- [x] `sumSlices` - trust LLVM auto-vectorization (explicit SIMD was 4x slower)
- [ ] `dotProduct` - fix overflow bug with partial reduction, then SIMD
- [ ] Benchmark on x86 (AVX2/AVX-512) to validate cross-platform

**Batched Sumcheck:**

- [ ] `BatchedMultilinear(k)` struct with interleaved k-tuple layout
- [ ] Batched `sumHalves` returning `[2][k]F`
- [ ] Batched `bind` operating on all k polynomials
- [ ] Horner's method for combining with α powers

**Multi-threading:**

- [ ] Static thread pool (avoid per-call allocation)
- [ ] Parallel `sumSlices` with chunk partitioning
- [ ] Parallel `linearCombineBatch` (bind)
- [ ] Cache-line aligned chunk boundaries (64 bytes)
- [ ] Benchmark scaling: 2, 4, 8 cores

**GPU Offload (Future):**

- [ ] Evaluate compute shaders vs CUDA
- [ ] Identify crossover point (n=? where GPU wins)
- [ ] Memory transfer overhead analysis

### Phase 5: Applications

- [ ] Simple VM circuit
- [ ] zkVM integration
- [ ] DA layer piggybacking (Celestia)

---

## References

- [Sumcheck Protocol](https://people.cs.georgetown.edu/jthaler/sumcheck.pdf) - Thaler
- [Lasso/Jolt](https://eprint.iacr.org/2023/1217) - a]6z research
- [SCRIBE](https://eprint.iacr.org/2024/1blockchain) - Streaming proofs
- [Plonky3](https://github.com/Plonky3/Plonky3) - Reference implementation
- [Binius](https://eprint.iacr.org/2023/1784) - Binary field approach

---

## Contributing

The hot paths are:

1. `field/mersenne31.zig` - Field arithmetic
2. `poly/multilinear.zig` - Polynomial operations
3. `sumcheck/prover.zig` - Sumcheck rounds

Profile before optimizing. The bottleneck is usually memory bandwidth, not compute.
