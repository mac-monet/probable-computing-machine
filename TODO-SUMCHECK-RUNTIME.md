# Sumcheck Runtime Architecture

This document describes the architecture for a composable, hyperoptimized sumcheck system that separates protocol orchestration from computation.

## Design Goals

1. **Composable** - Protocols (Basefold, GKR, zerocheck, lookups) control orchestration
2. **Hyperoptimized** - Single kernel codebase with SIMD, cache-aware implementations
3. **Runtime-swappable** - Same protocol code can target CPU, GPU, or distributed backends
4. **Zero-cost abstraction** - Strategy layer is compile-time, no runtime dispatch overhead

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│  Protocol Layer (Basefold, GKR, Zerocheck, Lookups)          │
│    - Orchestration (when to commit, absorb, etc.)            │
│    - Transcript ownership (Fiat-Shamir)                      │
│    - Protocol-specific state management                      │
└──────────────────────────────────────────────────────────────┘
                              │
                              │ prepares data, calls
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  Sumcheck Strategy (compile-time adapter)                    │
│    - Translates protocol state → kernel format               │
│    - Defines: degree, num_polys, data layout                 │
│    - Zero runtime cost (monomorphized)                       │
└──────────────────────────────────────────────────────────────┘
                              │
                              │ dispatches to
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  Sumcheck Kernel (the hyperoptimized runtime)                │
│    - Pure computation, no transcript, no orchestration       │
│    - SIMD, cache-aware, potentially GPU/async                │
│    - Fixed set of optimized implementations                  │
└──────────────────────────────────────────────────────────────┘
```

## Layer Responsibilities

### Protocol Layer

Protocols **own orchestration** - they decide:
- When to commit (Basefold commits between rounds)
- When to absorb into transcript
- How to chain sumchecks (GKR)
- Extension field transitions

Protocols **do not** implement sumcheck math.

### Strategy Layer

Strategies are **compile-time adapters** that:
- Define how protocol state maps to kernel inputs
- Specify degree and polynomial count
- Handle protocol-specific fold logic

Strategies are monomorphized - zero runtime overhead.

### Kernel Layer

The kernel is **dumb but fast**:
- Pure computation on raw slices
- No knowledge of transcripts, protocols, or meaning
- Single place for SIMD/GPU optimization
- Stateless and data-parallel

## Kernel Interface

```zig
// src/sumcheck/kernel.zig

pub fn Kernel(comptime F: type) type {
    return struct {
        // ============ Round Polynomial Computation ============ //

        /// Degree-1: Σ evals[i] for each half
        /// Returns [sum_lo, sum_hi]
        pub fn roundDegree1(evals: []const F) [2]F;

        /// Degree-2: Σ f[i]*g[i] evaluated at 0, 1, 2
        /// Returns [g(0), g(1), g(2)]
        pub fn roundDegree2(f: []const F, g: []const F) [3]F;

        /// Degree-N: General case (future)
        pub fn roundDegreeN(comptime N: usize, polys: [N][]const F) [N + 1]F;

        // ============ Folding ============ //

        /// Fold in-place: evals[i] = lo + challenge * (hi - lo)
        /// Returns smaller slice view
        pub fn fold(evals: []F, challenge: F) []F;

        /// Fold with extension field challenge (BaseF data, ExtF challenge)
        pub fn foldExt(comptime ExtF: type, evals: []F, challenge: ExtF) []ExtF;

        // ============ Verification Helpers ============ //

        /// Check round polynomial sum: round[0] + round[1] == claimed
        pub fn checkRoundSum(comptime degree: usize, round: [degree + 1]F, claimed: F) bool;

        /// Evaluate round polynomial at challenge point
        pub fn evalRoundPoly(comptime degree: usize, round: [degree + 1]F, challenge: F) F;
    };
}
```

## Strategy Interface

```zig
// src/sumcheck/strategy.zig

pub const StrategySpec = struct {
    /// Protocol's state type
    State: type,

    /// Tuple of evaluation slice types
    /// e.g., [1][]const F for linear, [2][]const F for product
    EvalSlices: type,

    /// Round polynomial degree (1 = linear, 2 = product)
    degree: comptime_int,

    /// Extract evaluation slices from protocol state
    getEvals: fn (*State) EvalSlices,

    /// Fold state with challenge (protocol may have custom logic)
    fold: fn (*State, F) void,
};

pub fn Strategy(comptime F: type, comptime spec: StrategySpec) type {
    const K = Kernel(F);

    return struct {
        pub const RoundPoly = [spec.degree + 1]F;

        /// Compute round polynomial using appropriate kernel
        pub fn computeRound(state: *spec.State) RoundPoly {
            const slices = spec.getEvals(state);
            return switch (spec.degree) {
                1 => K.roundDegree1(slices[0]),
                2 => K.roundDegree2(slices[0], slices[1]),
                else => K.roundDegreeN(spec.degree, slices),
            };
        }

        /// Fold protocol state
        pub fn fold(state: *spec.State, challenge: F) void {
            spec.fold(state, challenge);
        }

        /// Verify round sum
        pub fn checkRound(round: RoundPoly, claimed: F) bool {
            return K.checkRoundSum(spec.degree, round, claimed);
        }

        /// Evaluate round poly at challenge
        pub fn evalRound(round: RoundPoly, challenge: F) F {
            return K.evalRoundPoly(spec.degree, round, challenge);
        }
    };
}
```

## Protocol Usage Examples

### Basefold (interleaves commits)

```zig
const BasefoldState = struct {
    f_evals: []F,
    eq_evals: []F,
};

const BasefoldStrategy = Strategy(F, .{
    .State = BasefoldState,
    .EvalSlices = struct { f: []const F, eq: []const F },
    .degree = 2,

    .getEvals = struct {
        fn get(s: *BasefoldState) @TypeOf(.{}).EvalSlices {
            return .{ .f = s.f_evals, .eq = s.eq_evals };
        }
    }.get,

    .fold = struct {
        fn fold(s: *BasefoldState, c: F) void {
            s.f_evals = Kernel(F).fold(s.f_evals, c);
            s.eq_evals = Kernel(F).fold(s.eq_evals, c);
        }
    }.fold,
});

fn proveBasefold(state: *BasefoldState, transcript: *Transcript) void {
    for (0..num_vars) |i| {
        // Protocol-specific: commit BEFORE sumcheck round
        commitments[i] = Merkle.commit(state.f_evals);
        transcript.absorbBytes(&commitments[i]);

        // Delegate to strategy/kernel
        rounds[i] = BasefoldStrategy.computeRound(state);

        // Protocol-specific: absorb and squeeze
        for (rounds[i]) |eval| transcript.absorb(eval);
        const challenge = transcript.squeeze();

        // Delegate fold to strategy
        BasefoldStrategy.fold(state, challenge);
    }
}
```

### GKR (chains sumchecks)

```zig
const GKRStrategy = Strategy(F, .{
    .State = GKRLayerState,
    .EvalSlices = [3][]const F,  // add, mul_left, mul_right
    .degree = 3,
    // ...
});

fn proveGKRLayer(layer: *GKRLayerState, transcript: *Transcript) void {
    // First sumcheck for this layer
    for (0..num_vars) |i| {
        rounds[i] = GKRStrategy.computeRound(layer);
        // ... absorb, squeeze, fold ...
    }

    // Chain to next layer (GKR-specific logic)
    const next_claim = deriveNextLayerClaim(layer, challenges);
    // ... continue with next layer ...
}
```

### Virtual Columns (lazy evaluation)

```zig
const VirtualState = struct {
    columns: [][]const F,      // raw column data
    composition: CompositionFn, // how to combine columns

    // Lazy: don't materialize full product, compute on-the-fly
    fn evalAt(self: *VirtualState, i: usize) F {
        return self.composition(self.columns, i);
    }
};

const VirtualStrategy = Strategy(F, .{
    .State = VirtualState,
    .EvalSlices = VirtualEvalProvider,  // lazy iterator
    .degree = 2,
    // ...
});
```

## Runtime Backend Options

For large polynomials or hardware acceleration:

```zig
pub const RuntimeBackend = union(enum) {
    /// Inline CPU - current default
    inline_cpu,

    /// Thread pool for parallel sumcheck
    thread_pool: *ThreadPool,

    /// GPU acceleration (CUDA/Metal/Vulkan)
    gpu: *GpuContext,

    /// Separate process (sandboxing, distributed)
    remote: *RemoteRuntime,
};

pub fn Kernel(comptime F: type, comptime backend: RuntimeBackend) type {
    return struct {
        pub fn roundDegree2(f: []const F, g: []const F) [3]F {
            return switch (backend) {
                .inline_cpu => roundDegree2Cpu(f, g),
                .thread_pool => |pool| roundDegree2Parallel(f, g, pool),
                .gpu => |ctx| roundDegree2Gpu(f, g, ctx),
                .remote => |rt| rt.call(.roundDegree2, .{ f, g }),
            };
        }
    };
}
```

## Async/Pipelined Execution

For overlapping computation with I/O:

```zig
pub fn KernelAsync(comptime F: type) type {
    return struct {
        pub fn roundDegree2(f: []const F, g: []const F) Future([3]F);
        pub fn fold(evals: []F, challenge: F) Future([]F);
    };
}

// Protocol can overlap computation
fn proveBasefoldPipelined(state: *State, transcript: *Transcript) void {
    var prev_round_future: ?Future([3]F) = null;

    for (0..num_vars) |i| {
        // Start next round computation
        const round_future = KernelAsync.roundDegree2(state.f, state.eq);

        // While computing, do I/O for previous round
        if (prev_round_future) |fut| {
            const prev_round = fut.await();
            for (prev_round) |eval| transcript.absorb(eval);
            const c = transcript.squeeze();
            KernelAsync.fold(state, c).await();
        }

        prev_round_future = round_future;
    }
}
```

## Extension Field Handling

The kernel supports BaseF → ExtF transitions:

```zig
pub fn Kernel(comptime F: type) type {
    const ExtF = F.Extension;  // e.g., M31 → M31Ext2

    return struct {
        /// First round: BaseF data, ExtF challenge → ExtF result
        pub fn foldBaseToExt(evals: []const F, challenge: ExtF) []ExtF;

        /// Subsequent rounds: ExtF data, ExtF challenge → ExtF result
        pub fn foldExt(evals: []ExtF, challenge: ExtF) []ExtF;

        /// Round computation in extension field
        pub fn roundDegree2Ext(f: []const ExtF, g: []const ExtF) [3]ExtF;
    };
}
```

Strategy handles the transition:

```zig
const ExtensionAwareStrategy = struct {
    state: union(enum) {
        base: BaseState,
        ext: ExtState,
    },

    fn fold(self: *@This(), challenge: ExtF) void {
        switch (self.state) {
            .base => |*s| {
                // Transition to extension field
                self.state = .{ .ext = Kernel.foldBaseToExt(s.evals, challenge) };
            },
            .ext => |*s| {
                s.evals = Kernel.foldExt(s.evals, challenge);
            },
        }
    }
};
```

## Module Structure

```
src/
├── sumcheck/
│   ├── kernel.zig          # Hyperoptimized primitives
│   ├── kernel_simd.zig     # SIMD implementations
│   ├── kernel_gpu.zig      # GPU implementations (future)
│   ├── strategy.zig        # Compile-time adapters
│   └── verify.zig          # Verification math
├── iop/
│   ├── zerocheck.zig       # Uses sumcheck strategy
│   ├── gkr.zig             # Uses sumcheck strategy
│   └── lookup.zig          # Uses sumcheck strategy
├── pcs/
│   └── basefold.zig        # Uses sumcheck strategy
└── protocol/
    └── ...
```

## Implementation Plan

### Phase 1: Extract Kernel
1. Create `src/sumcheck/kernel.zig` with current SIMD-optimized code
2. Move `roundDegree1`, `roundDegree2`, `fold` from current impl
3. Add `checkRoundSum`, `evalRoundPoly` helpers

### Phase 2: Strategy Layer
1. Create `src/sumcheck/strategy.zig` with `StrategySpec` and `Strategy`
2. Define `LinearStrategy`, `ProductStrategy` built-in strategies
3. Verify zero overhead vs current implementation

### Phase 3: Migrate Protocols
1. Update Basefold to use `BasefoldStrategy`
2. Verify tests still pass
3. Remove `prove`/`verify` from old sumcheck (protocols orchestrate)

### Phase 4: Extension Fields
1. Implement `M31Ext2`, `M31Ext4` in `src/fields/`
2. Add `foldBaseToExt`, `foldExt` to kernel
3. Update strategies to handle BaseF → ExtF transition

### Phase 5: Runtime Backends
1. Add `RuntimeBackend` enum
2. Implement thread pool backend for large polynomials
3. Add async/Future interface for pipelined execution

### Phase 6: GPU (Future)
1. Metal/CUDA kernel implementations
2. Memory transfer optimization
3. Batched operations for amortizing transfer cost

## Testing Strategy

```zig
test "kernel matches naive implementation" {
    // Verify SIMD kernel produces same results as scalar
}

test "strategy has zero overhead" {
    // Benchmark strategy vs direct kernel call
}

test "protocol orchestration is correct" {
    // End-to-end Basefold test with strategy
}

test "extension field transition" {
    // Verify BaseF → ExtF produces correct results
}
```

## Open Questions

1. **Chunked interface**: For polynomials larger than memory, need streaming API
2. **Batched sumcheck**: Multiple sumcheck instances with shared randomness
3. **Witness generation**: How does kernel interact with witness/trace generation?
4. **Memory pools**: Should kernel manage its own memory, or use caller's allocator?
