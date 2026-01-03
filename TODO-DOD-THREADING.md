# DOD Threading Design for Scratch Buffers

## Data Transformation Analysis

Map what happens to data in each phase:

```
Phase 1 (Commit):
  INPUT:  poly_evals[2^n]     (read-only during commit)
  WORK:   hash pairs bottom-up
  OUTPUT: merkle_nodes[2^(n+1)-1] → arena (persists in proof)

Phase 2 (IOP - Sumcheck):
  INPUT:  poly_evals[2^n]     (mutated via folding)
  WORK:   compute round poly (parallel sum), fold in-place
  OUTPUT: round_polys[n] → arena
          poly shrinks: 2^n → 2^(n-1) → ... → 1

Phase 3 (Open - Basefold):
  INPUT:  already-folded poly from Phase 2, OR fresh poly
  WORK:   more folding + merkle openings
  OUTPUT: opening_proofs → arena
```

Key insight: **polynomial data is the hot path**. Everything else (merkle nodes, round polys, proofs) goes to arena and is "cold" after creation.

## The Multithreading Problem

Within a sumcheck round, two operations are parallelizable:

```zig
// 1. Compute round polynomial: Σ over half the domain
//    Embarrassingly parallel - each thread sums a range
fn computeRound(poly: []const F, eq: []const F) RoundPoly {
    const half = poly.len / 2;
    // Thread 0: sum indices [0, half/num_threads)
    // Thread 1: sum indices [half/num_threads, 2*half/num_threads)
    // ...
    // Reduce partial sums at end
}

// 2. Fold: each element independent
//    Embarrassingly parallel - each thread folds a range
fn fold(poly: []F, challenge: F) []F {
    const half = poly.len / 2;
    // Thread 0: fold indices [0, half/num_threads)
    // Thread 1: fold indices [half/num_threads, 2*half/num_threads)
    // No reduction needed - writes go directly to poly[0..half]
}
```

Problem: **threads need scratch space for partial sums** in `computeRound`.

## Solution: Hierarchical Scratch

```zig
pub const ScratchConfig = struct {
    max_vars: usize,           // max polynomial size = 2^max_vars
    num_threads: usize,        // thread pool size
};

pub fn ProverContext(comptime F: type, comptime config: ScratchConfig) type {
    const max_size = 1 << config.max_vars;
    const num_threads = config.num_threads;

    return struct {
        const Self = @This();

        // ═══════════════════════════════════════════════════════════
        // SHARED BUFFERS (polynomial data lives here)
        // ═══════════════════════════════════════════════════════════

        /// Primary polynomial buffer - folded in place
        /// After round i: valid data in [0, 2^(n-i))
        poly: []F,              // size: max_size

        /// Secondary polynomial buffer (eq, second poly in product)
        poly_aux: []F,          // size: max_size

        /// Tertiary buffer for operations needing 3 polys (GKR, cubic sumcheck)
        poly_third: []F,        // size: max_size (optional, can be null)

        // ═══════════════════════════════════════════════════════════
        // PER-THREAD SCRATCH (small, cache-local)
        // ═══════════════════════════════════════════════════════════

        /// Per-thread scratch for partial sums, temporary values
        /// Sized for one cache line per "slot" to avoid false sharing
        thread_scratch: [num_threads]ThreadScratch,

        // ═══════════════════════════════════════════════════════════
        // PROOF DATA (append-only, freed all at once)
        // ═══════════════════════════════════════════════════════════

        arena: std.heap.ArenaAllocator,

        // ═══════════════════════════════════════════════════════════
        // SHARED STATE
        // ═══════════════════════════════════════════════════════════

        transcript: Transcript,

        /// Current valid size of poly buffer (shrinks during folding)
        current_size: usize,

        /// Thread pool handle
        thread_pool: *ThreadPool,
    };
}

const ThreadScratch = struct {
    /// Partial sums for round poly computation (degree+1 values)
    /// Aligned to cache line to prevent false sharing
    partial_sums: [8]F align(64),   // supports up to degree-7 sumcheck

    /// Small scratch for thread-local work
    local: [64]F align(64),         // one cache line of field elements
};
```

## Buffer Reuse Across Phases

```
TIME →

┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: COMMIT                                                 │
│                                                                 │
│ poly[2^n]: ████████████████████████████████  (read-only)        │
│                                                                 │
│ Merkle build uses poly as input, writes to arena                │
│ poly_aux: [unused, available for merkle scratch]                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: IOP (multiple IOPs run sequentially)                   │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Zerocheck on constraint poly                                │ │
│ │                                                             │ │
│ │ Round 0: poly[2^n], poly_aux[2^n] (eq)                      │ │
│ │          ████████████████████████████████                   │ │
│ │          ████████████████████████████████                   │ │
│ │                                                             │ │
│ │ Round 1: poly[2^(n-1)], poly_aux[2^(n-1)]                   │ │
│ │          ████████████████                                   │ │
│ │          ████████████████                                   │ │
│ │          [freed]────────→ could reuse for something         │ │
│ │                                                             │ │
│ │ Round n: poly[1], poly_aux[1]                               │ │
│ │          █                                                  │ │
│ │          █                                                  │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Lookup IOP (if enabled)                                     │ │
│ │                                                             │ │
│ │ RESET: poly_aux rebuilt with lookup data                    │ │
│ │ poly: ████████████████████████████████ (query poly)         │ │
│ │ poly_aux: ████████████████████████████████ (table poly)     │ │
│ │                                                             │ │
│ │ Folds same as above                                         │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: OPEN                                                   │
│                                                                 │
│ For each opening:                                               │
│   RESET: Load polynomial into poly buffer                       │
│   Run Basefold (sumcheck + merkle proofs)                       │
│   poly shrinks via folding, same pattern as Phase 2             │
└─────────────────────────────────────────────────────────────────┘
```

## Parallel Sumcheck Round Implementation

```zig
fn computeRoundParallel(
    ctx: *ProverContext,
    comptime degree: usize,
) [degree + 1]F {
    const poly = ctx.poly[0..ctx.current_size];
    const eq = ctx.poly_aux[0..ctx.current_size];
    const half = poly.len / 2;
    const num_threads = ctx.thread_pool.num_threads;
    const chunk_size = half / num_threads;

    // 1. Parallel: each thread computes partial sums for its chunk
    ctx.thread_pool.parallel(num_threads, struct {
        fn work(thread_id: usize, ctx_ptr: *ProverContext) void {
            const start = thread_id * chunk_size;
            const end = if (thread_id == num_threads - 1) half else start + chunk_size;

            // Accumulate into thread-local scratch (no contention)
            var partial: [degree + 1]F = .{F.zero} ** (degree + 1);

            for (start..end) |i| {
                const lo_f = poly[i];
                const hi_f = poly[i + half];
                const lo_eq = eq[i];
                const hi_eq = eq[i + half];

                // Evaluate at 0, 1, ..., degree
                inline for (0..degree + 1) |eval_point| {
                    const f_interp = interpolate(lo_f, hi_f, eval_point);
                    const eq_interp = interpolate(lo_eq, hi_eq, eval_point);
                    partial[eval_point] = partial[eval_point].add(f_interp.mul(eq_interp));
                }
            }

            // Write to thread scratch (cache-line aligned, no false sharing)
            @memcpy(&ctx_ptr.thread_scratch[thread_id].partial_sums, &partial);
        }
    }.work);

    // 2. Sequential: reduce partial sums (fast, only num_threads additions)
    var result: [degree + 1]F = .{F.zero} ** (degree + 1);
    for (ctx.thread_scratch[0..num_threads]) |ts| {
        inline for (0..degree + 1) |i| {
            result[i] = result[i].add(ts.partial_sums[i]);
        }
    }

    return result;
}

fn foldParallel(ctx: *ProverContext, challenge: F) void {
    const poly = ctx.poly[0..ctx.current_size];
    const eq = ctx.poly_aux[0..ctx.current_size];
    const half = poly.len / 2;
    const num_threads = ctx.thread_pool.num_threads;
    const chunk_size = half / num_threads;

    // Parallel: each thread folds its chunk (writes directly to poly[0..half])
    // No reduction needed - each thread writes to disjoint indices
    ctx.thread_pool.parallel(num_threads, struct {
        fn work(thread_id: usize, ctx_ptr: *ProverContext) void {
            const start = thread_id * chunk_size;
            const end = if (thread_id == num_threads - 1) half else start + chunk_size;

            for (start..end) |i| {
                // poly[i] = lo + challenge * (hi - lo)
                const lo = poly[i];
                const hi = poly[i + half];
                poly[i] = lo.add(challenge.mul(hi.sub(lo)));

                // same for eq
                const lo_eq = eq[i];
                const hi_eq = eq[i + half];
                eq[i] = lo_eq.add(challenge.mul(hi_eq.sub(lo_eq)));
            }
        }
    }.work);

    ctx.current_size = half;
}
```

## Phase Transitions

```zig
pub const Phase = enum { commit, iop, open };

pub fn transitionTo(ctx: *ProverContext, phase: Phase, poly_data: ?[]const F) void {
    switch (phase) {
        .commit => {
            // poly buffer has original polynomial (read-only during commit)
            // poly_aux is free for merkle scratch
            ctx.current_size = poly_data.?.len;
            @memcpy(ctx.poly[0..ctx.current_size], poly_data.?);
        },
        .iop => {
            // Entering IOP phase
            // If poly_data provided, load it (otherwise reuse from commit)
            if (poly_data) |data| {
                ctx.current_size = data.len;
                @memcpy(ctx.poly[0..ctx.current_size], data);
            }
            // poly_aux will be populated with eq or second poly
        },
        .open => {
            // Reset for opening phase
            // May need to reload original polynomial if it was folded
            if (poly_data) |data| {
                ctx.current_size = data.len;
                @memcpy(ctx.poly[0..ctx.current_size], data);
            }
        },
    }
}

// Between IOPs within Phase 2
pub fn resetForNextIOP(ctx: *ProverContext, new_poly: ?[]const F, new_aux: ?[]const F) void {
    if (new_poly) |p| {
        ctx.current_size = p.len;
        @memcpy(ctx.poly[0..p.len], p);
    }
    if (new_aux) |a| {
        @memcpy(ctx.poly_aux[0..a.len], a);
    }
}
```

## Handling Different IOP Scratch Sizes

Buffers sized for maximum, tracked for actual use:

```zig
const ProverConfig = struct {
    max_constraint_vars: usize,  // e.g., 20 → 2^20 constraints
    max_lookup_table: usize,     // e.g., 16 → 2^16 table entries
    max_gkr_layer: usize,        // e.g., 18 → 2^18 max layer size

    pub fn maxPolySize(self: ProverConfig) usize {
        return @max(
            @as(usize, 1) << self.max_constraint_vars,
            @as(usize, 1) << self.max_lookup_table,
            @as(usize, 1) << self.max_gkr_layer,
        );
    }
};

// Buffers sized to max, but current_size tracks actual usage
ctx.poly = allocator.alloc(F, config.maxPolySize());
ctx.current_size = actual_constraint_count;  // might be smaller
```

## Memory Layout Visualization

```
ProverContext Memory Layout
═══════════════════════════════════════════════════════════════════

SHARED (allocated once, reused across all phases)
┌─────────────────────────────────────────────────────────────────┐
│ poly[2^max_vars]                                                │
│ ┌───────────────────────────────────────────────────────────┐   │
│ │ Round 0: [████████████████████████████████████████████]   │   │
│ │ Round 1: [████████████████████]                           │   │
│ │ Round 2: [██████████]                                     │   │
│ │ ...                                                       │   │
│ │ Final:   [█]                                              │   │
│ └───────────────────────────────────────────────────────────┘   │
│ current_size tracks valid region                                │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│ poly_aux[2^max_vars]  (same pattern)                            │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│ poly_third[2^max_vars]  (optional, for GKR/cubic)               │
└─────────────────────────────────────────────────────────────────┘

PER-THREAD (small, cache-aligned)
┌─────────────────────────────────────────────────────────────────┐
│ thread_scratch[num_threads]                                     │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Thread 0: [partial_sums: 64B aligned][local: 64B aligned]   │ │
│ │ Thread 1: [partial_sums: 64B aligned][local: 64B aligned]   │ │
│ │ Thread 2: [partial_sums: 64B aligned][local: 64B aligned]   │ │
│ │ ...                                                         │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ Each thread writes only to its own scratch → no false sharing   │
└─────────────────────────────────────────────────────────────────┘

ARENA (grows during proving, freed all at once)
┌─────────────────────────────────────────────────────────────────┐
│ arena: ArenaAllocator                                           │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ [merkle_nodes][round_polys][commitments][query_proofs]...   │ │
│ │ ──────────────────────────────────────────────────────────→ │ │
│ │                                          grows rightward    │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ All freed on arena.reset() or arena.deinit()                    │
└─────────────────────────────────────────────────────────────────┘
```

## Threading Model

```
                    ┌─────────────────────┐
                    │   Main Thread       │
                    │   (orchestrator)    │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
    ┌───────────┐        ┌───────────┐        ┌───────────┐
    │ Worker 0  │        │ Worker 1  │        │ Worker 2  │
    │           │        │           │        │           │
    │ Reads:    │        │ Reads:    │        │ Reads:    │
    │ poly[0:k] │        │ poly[k:2k]│        │ poly[2k:n]│
    │           │        │           │        │           │
    │ Writes:   │        │ Writes:   │        │ Writes:   │
    │ scratch[0]│        │ scratch[1]│        │ scratch[2]│
    │ poly[0:k/2]        │ poly[k/2:k]│       │ poly[k:n/2]│
    └───────────┘        └───────────┘        └───────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                               ▼ barrier
                    ┌─────────────────────┐
                    │   Main Thread       │
                    │   reduces scratch   │
                    │   absorbs to        │
                    │   transcript        │
                    └─────────────────────┘

Within a round:
1. FORK: workers compute partial sums in parallel
2. JOIN: main thread reduces, absorbs, squeezes challenge
3. FORK: workers fold their ranges in parallel
4. JOIN: barrier before next round

Transcript operations are always single-threaded (sequential absorb/squeeze).
```

## DOD Principles Applied

| Principle | Application |
|-----------|-------------|
| **Single owner** | `ProverContext` owns all buffers, IOPs borrow via slices |
| **Reuse over allocate** | Same `poly`/`poly_aux` buffers across all phases and IOPs |
| **Hot/cold split** | Polynomial data (hot, mutated) separate from proof data (cold, arena) |
| **Cache-line alignment** | Thread scratch aligned to 64 bytes, no false sharing |
| **Partition over lock** | Threads get index ranges, not locks on shared state |
| **Sequential phases, parallel within** | Barrier between phases; fork-join within rounds |
| **Size for worst-case, track actual** | Buffers sized to max, `current_size` tracks valid data |

## Open Questions

1. **When does parallelization help?**
   - Small polynomials: overhead > benefit
   - Threshold likely around 2^12 - 2^14 elements
   - Should have single-threaded fallback

2. **SIMD within threads?**
   - Field ops (M31) are good candidates for SIMD
   - Each thread could use AVX2/AVX512 on its chunk
   - Orthogonal to threading design

3. **Memory bandwidth bottleneck?**
   - Large polynomials may be memory-bound, not compute-bound
   - Prefetching might help: `@prefetch(poly[i + prefetch_distance])`
   - Profile before optimizing

4. **Thread pool design?**
   - Static pool (created once) vs dynamic spawning
   - Work-stealing vs fixed partitioning
   - Zig's `std.Thread.Pool` or custom?
