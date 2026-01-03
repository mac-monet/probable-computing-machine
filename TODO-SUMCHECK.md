# Generic Sumcheck Refactor

## Goal
Replace `sumcheck.zig` and `product_sumcheck.zig` with a single generic `Sumcheck(F, Config)` that supports all variants via comptime configuration.

## Core Interface

```zig
pub fn Sumcheck(comptime F: type, comptime Config: type) type {
    return struct {
        pub fn prove(state: *Config.State, transcript: *Transcript(F), rounds: []Config.RoundPoly) Config.FinalEval;
        pub fn verify(claimed: F, rounds: []const Config.RoundPoly, transcript: *Transcript(F), challenges: []F) !F;
    };
}
```

## Config Requirements

```zig
Config {
    const State;                                      // Polynomial data
    const RoundPoly;                                  // Round polynomial type
    const FinalEval;                                  // What prove() returns

    fn computeRound(*State) RoundPoly;                // Build round poly
    fn absorb(*Transcript, *const RoundPoly) void;    // Fiat-Shamir
    fn bind(*State, challenge: F) void;               // Fold polynomial
    fn finalEval(*State) FinalEval;                   // Extract final value

    // Optional fused operation (bandwidth optimization)
    const bindAndComputeNext: ?fn(*State, F) RoundPoly = null;
}
```

## Configs to Implement

| Config | State | Round Degree | Use Case |
|--------|-------|--------------|----------|
| `LinearConfig` | 1 poly | 1 (2 evals) | Simple sum |
| `ProductConfig` | 2 polys (f, g) | 2 (3 evals) | f·eq sumcheck |
| `ZerocheckConfig` | 2 polys (C, eq) | 2 (3 evals) | Constraint proving |
| `VirtualProductConfig` | N column refs | N (N+1 evals) | Memory-efficient compositions |

## Fused Bind+Compute Optimization

First round: `computeRound` only (no prior challenge)
Subsequent rounds: `bindAndComputeNext` combines:
  1. Fold with challenge
  2. Accumulate next round sums while writing

Saves ~50% memory bandwidth (one pass instead of two per round).

```zig
fn bindAndComputeNext(state: *State, c: F) RoundPoly {
    const half = state.evals.len / 2;
    const quarter = half / 2;
    var sum_0, sum_1 = F.zero;

    for (0..half) |i| {
        const folded = state.evals[i].add(c.mul(state.evals[i + half].sub(state.evals[i])));
        state.evals[i] = folded;
        if (i < quarter) sum_0 = sum_0.add(folded) else sum_1 = sum_1.add(folded);
    }
    state.evals = state.evals[0..half];
    return .{ .eval_0 = sum_0, .eval_1 = sum_1 };
}
```

## Implementation Order

1. Define `Config` interface in `src/sumcheck/config.zig`
2. Implement `LinearConfig` - simplest case
3. Implement generic `Sumcheck(F, Config)` in `src/sumcheck/core.zig`
4. Implement `ProductConfig` - replaces `product_sumcheck.zig`
5. Add fused optimization to both configs
6. Implement `VirtualProductConfig` for lazy evaluation
7. Update `basefold.zig` to use new generic
8. Update `vm/prover.zig` to use new generic
9. Delete old `sumcheck.zig` and `product_sumcheck.zig`

## Future: Chunked Interface (for async engine)

```zig
Config {
    // Chunk-parallel operations
    fn computeRoundChunk(*State, start: usize, end: usize) PartialRound;
    fn mergePartials([]PartialRound) RoundPoly;
    fn bindChunk(*State, challenge: F, start: usize, end: usize) void;
}
```

## Extension Field Requirements

Small base fields (M31, BabyBear, Goldilocks) require extension field challenges for security.

**Why:** Grinding attacks. Prover can try ~|F| different commitments to find one where challenges "work out." With |F| = 2^31, this is feasible in hours.

**What needs extension field:**

| Component | Extension needed? | Reason |
|-----------|-------------------|--------|
| Sumcheck challenges | Yes | Prevent grinding on challenge points |
| FRI/Basefold folding | Yes | Prevent grinding on fold challenges |
| RLC batching | Yes | Prevent cancellation attacks |
| Trace/witness data | No | Just storage, no security role |
| Constraint evaluation | No | Data computation, not challenges |

**The contamination problem:**

Once you bind with an ExtF challenge, values become ExtF:
```
Round 0: data in BaseF, challenge c ∈ ExtF
         bind(baseF_data, c) → ExtF values
Round 1+: all computation in ExtF
```

**Revised interface:**

```zig
pub fn Sumcheck(comptime BaseF: type, comptime Config: type) type {
    const ExtF = BaseF.Extension;

    return struct {
        // State starts in BaseF, transitions to ExtF after first bind
        // Transcript produces ExtF challenges
        // Round polys are ExtF after round 0
    };
}
```

**Config changes:**

```zig
Config {
    const BaseState;     // Initial data in BaseF
    const ExtState;      // After first bind, data in ExtF
    const RoundPoly;     // In ExtF (evals are ExtF after round 0)

    fn computeRoundBase(*BaseState) RoundPoly;           // Round 0 only
    fn computeRoundExt(*ExtState) RoundPoly;             // Rounds 1+
    fn bindBaseToExt(*BaseState, ExtF) ExtState;         // First bind: BaseF → ExtF
    fn bindExt(*ExtState, ExtF) void;                    // Subsequent binds: ExtF → ExtF
}
```

**Extension field to implement:**

```zig
// M31 quadratic extension: F[x]/(x² - 5)
pub const M31Ext2 = struct {
    c0: M31,
    c1: M31,

    // For 62-bit security
};

// M31 quartic extension: Ext2[y]/(y² - nonresidue)
pub const M31Ext4 = struct {
    c0: M31Ext2,
    c1: M31Ext2,

    // For 124-bit security
};
```

**Implementation order (revised):**

1. Implement `M31Ext2` and `M31Ext4` in `src/field/`
2. Add `Extension` type to field interface
3. Update `Transcript` to work with extension fields
4. Update `Sumcheck` to handle BaseF → ExtF transition
5. Update `Basefold` to use ExtF for folding challenges
