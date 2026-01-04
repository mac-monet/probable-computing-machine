# s6k Runtime

The s6k runtime is the async execution engine for sumcheck computations. It wraps s6k primitives (see S6K_SYNC.md) as jobs and handles scheduling, parallelism, and hardware dispatch.

## Evolution from Primitives

```
s6k Primitives (v1)              s6k Runtime (v2)
─────────────────────            ─────────────────────
Protocol calls directly    →     Protocol submits job graph
Protocol owns execution    →     Runtime owns execution
Synchronous               →     Asynchronous
Single-threaded           →     Parallel (CPU pool, GPU, distributed)
```

The primitives remain unchanged - the runtime wraps them:

```zig
// Primitive (from SUMCHECK_RUNTIME_V2.md)
pub fn computeRound(comptime degree, buffer: [*]const F, len: usize) Coeffs(degree);

// Runtime job executor calls the same primitive
fn executeJob(self: *Runtime, job: Job) void {
    switch (job) {
        .compute_round => |j| {
            const buf = self.buffers.getSlice(j.buffer);
            const coeffs = s6k(F).computeRound(j.degree, buf.ptr, j.len);
            self.promises.fulfill(j.output, coeffs);
        },
        // ...
    }
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Protocol Layer (Basefold, Zerocheck, GKR, Lookups)             │
│    - Builds job graph describing computation                     │
│    - Submits graph to runtime                                    │
│    - Awaits results                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ submits job graph
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  s6k Runtime                                                     │
│    - Analyzes job dependencies                                   │
│    - Schedules ready jobs to backends                            │
│    - Manages buffer lifecycle                                    │
│    - Fulfills promises on completion                             │
└─────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
   ┌────────────┐      ┌────────────┐      ┌────────────┐
   │ CPU Pool   │      │ GPU Queue  │      │ IO Pool    │
   │            │      │            │      │            │
   └──────┬─────┘      └──────┬─────┘      └──────┬─────┘
          │                   │                   │
          ▼                   ▼                   ▼
   ┌────────────────────────────────────────────────────┐
   │  s6k Primitives (same code, different backends)    │
   │    - computeRound, foldInPlace, etc.               │
   │    - SIMD (CPU), shader (GPU)                      │
   └────────────────────────────────────────────────────┘
```

---

## Motivation

The synchronous primitive model leaves performance on the table:

| Limitation | Runtime Solution |
|------------|------------------|
| No parallelism across sumchecks | Batched job graphs execute concurrently |
| No compute/IO overlap | Pipeline parallelism (commit while computing) |
| No within-round parallelism | Chunk jobs split large polynomials |
| No GPU offload | Backend dispatch based on size threshold |
| No distributed execution | Handles serialize for network transfer |

---

## Parallelism Opportunities

### 1. Independent Sumchecks (Batched Proving)

Multiple sumcheck instances with independent transcripts:

```
Time →

Sumcheck A: [round_0] [round_1] [round_2] ...
Sumcheck B: [round_0] [round_1] [round_2] ...
Sumcheck C: [round_0] [round_1] [round_2] ...
            ↑________↑_________↑
            All run in parallel
```

### 2. Pipeline Parallelism (Compute + IO)

Overlap Merkle commits with computation:

```
Time →

Compute:  [round_0        ] [round_1        ] [round_2        ]
Commit:            [commit_0       ] [commit_1       ] [commit_2
                   ↑_______________↑
                   Overlapped with round_1 compute
```

### 3. Within-Round Parallelism (Chunking)

Large polynomials split across workers:

```
Polynomial (2^20 elements):

┌───────────────┬───────────────┬───────────────┬───────────────┐
│ chunk_0       │ chunk_1       │ chunk_2       │ chunk_3       │
│ (0..2^18)     │ (2^18..2^19)  │ (2^19..3·2^18)│ (3·2^18..2^20)│
└───────┬───────┴───────┬───────┴───────┬───────┴───────┬───────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
   partial_0       partial_1       partial_2       partial_3
        │               │               │               │
        └───────────────┴───────┬───────┴───────────────┘
                                ▼
                         reduce_coeffs
                                │
                                ▼
                         final coefficients
```

### 4. Cross-Layer Parallelism (GKR)

GKR layers can pipeline:

```
Layer N:   [sumcheck] → claim_{N-1}
Layer N-1:             [sumcheck] → claim_{N-2}
Layer N-2:                         [sumcheck] → ...
                       ↑
                       Starts when claim ready
```

### 5. Precomputation Parallelism (SVO)

Accumulator precomputation is embarrassingly parallel:

```
precompute_accumulators job → splits into:

┌─────────────────┬─────────────────┬─────────────────┐
│ chunk (x'=0..K) │ chunk (x'=K..2K)│ chunk (x'=2K..) │
└────────┬────────┴────────┬────────┴────────┬────────┘
         │                 │                 │
         ▼                 ▼                 ▼
    partial_accum     partial_accum     partial_accum
         │                 │                 │
         └─────────────────┼─────────────────┘
                           ▼
                  reduce_accumulators
```

---

## Job Types

Jobs map 1:1 to s6k primitives. The runtime executes the primitive when the job runs.

```zig
pub const Job = union(enum) {
    // ══════════════════════════════════════════════════════════════
    // Round Computation
    // Primitive: s6k.computeRound
    // ══════════════════════════════════════════════════════════════
    compute_round: struct {
        buffer: BufferHandle,
        len: u32,
        degree: u8,
        output: PromiseHandle,
    },

    // Primitive: s6k.computeRoundWithEq
    compute_round_with_eq: struct {
        buffer: BufferHandle,
        len: u32,
        eq_table: BufferHandle,
        degree: u8,
        output: PromiseHandle,
    },

    // Primitive: s6k.computeRoundSplitEq (Gruen optimization)
    compute_round_split_eq: struct {
        buffer: BufferHandle,
        len: u32,
        eq_L: BufferHandle,
        eq_L_len: u32,
        eq_R: BufferHandle,
        eq_R_len: u32,
        degree: u8,
        output: PromiseHandle,
    },

    // Primitive: s6k.computeRoundFromAccumulators (SVO)
    compute_round_from_accumulators: struct {
        challenge_tensor: BufferHandle,
        tensor_len: u32,
        accumulators: BufferHandle,
        num_v: u32,
        degree: u8,
        output: PromiseHandle,
    },

    // ══════════════════════════════════════════════════════════════
    // Folding
    // Primitive: s6k.foldInPlace
    // ══════════════════════════════════════════════════════════════
    fold_in_place: struct {
        buffer: BufferHandle,
        len: u32,
        num_polys: u8,
        challenge: ValueHandle,
        output_len: PromiseHandle,  // Returns new length
    },

    // Primitive: s6k.foldToExt
    fold_to_ext: struct {
        base_buffer: BufferHandle,
        len: u32,
        num_polys: u8,
        challenge: ValueHandle,
        ext_buffer: BufferHandle,
        output_len: PromiseHandle,
    },

    // ══════════════════════════════════════════════════════════════
    // Eq Table Operations
    // Primitive: s6k.buildEqTable
    // ══════════════════════════════════════════════════════════════
    build_eq_table: struct {
        challenges: BufferHandle,
        num_vars: u32,
        output: BufferHandle,
    },

    // Primitive: s6k.foldEqTable
    fold_eq_table: struct {
        eq_table: BufferHandle,
        len: u32,
        challenge: ValueHandle,
        output_len: PromiseHandle,
    },

    // ══════════════════════════════════════════════════════════════
    // Accumulator Operations (SVO - Algorithms 4/6)
    // Primitive: s6k.precomputeAccumulators
    // ══════════════════════════════════════════════════════════════
    precompute_accumulators: struct {
        buffer: BufferHandle,
        len: u32,
        num_rounds: u8,
        degree: u8,
        output: BufferHandle,
        offsets_output: BufferHandle,
    },

    // Primitive: s6k.lagrangeBasis
    lagrange_basis: struct {
        point: ValueHandle,
        degree: u8,
        output: PromiseHandle,
    },

    // Primitive: s6k.extendTensor
    extend_tensor: struct {
        tensor: BufferHandle,
        tensor_len: u32,
        basis: ValueHandle,
        degree: u8,
        output_len: PromiseHandle,
    },

    // ══════════════════════════════════════════════════════════════
    // Parallel Chunk Operations
    // For splitting large computations across workers
    // ══════════════════════════════════════════════════════════════
    compute_round_chunk: struct {
        buffer: BufferHandle,
        total_len: u32,
        chunk_offset: u32,
        chunk_len: u32,
        degree: u8,
        output: PromiseHandle,  // Partial coefficients
    },

    reduce_coeffs: struct {
        partials: []const PromiseHandle,
        degree: u8,
        output: PromiseHandle,
    },

    precompute_accumulators_chunk: struct {
        buffer: BufferHandle,
        len: u32,
        chunk_offset: u32,
        chunk_len: u32,
        num_rounds: u8,
        degree: u8,
        output: BufferHandle,  // Partial accumulators
    },

    reduce_accumulators: struct {
        partials: []const BufferHandle,
        output: BufferHandle,
    },

    // ══════════════════════════════════════════════════════════════
    // Verification
    // Primitive: s6k.checkSum, s6k.evalAt
    // ══════════════════════════════════════════════════════════════
    check_sum: struct {
        coeffs: ValueHandle,
        claimed: ValueHandle,
        degree: u8,
        output: PromiseHandle,  // bool
    },

    eval_at: struct {
        coeffs: ValueHandle,
        point: ValueHandle,
        degree: u8,
        output: PromiseHandle,
    },

    // ══════════════════════════════════════════════════════════════
    // Transcript Operations
    // ══════════════════════════════════════════════════════════════
    transcript_absorb: struct {
        transcript: TranscriptHandle,
        data: ValueHandle,
    },

    transcript_absorb_buffer: struct {
        transcript: TranscriptHandle,
        buffer: BufferHandle,
        len: u32,
    },

    transcript_squeeze: struct {
        transcript: TranscriptHandle,
        output: PromiseHandle,
    },

    // ══════════════════════════════════════════════════════════════
    // IO Operations
    // ══════════════════════════════════════════════════════════════
    merkle_commit: struct {
        data: BufferHandle,
        len: u32,
        output: PromiseHandle,
    },

    merkle_open: struct {
        commitment: ValueHandle,
        index: u32,
        output: PromiseHandle,
    },
};
```

---

## Handles

Handles are stable references that survive across async boundaries. They enable serialization for distributed execution and lifecycle tracking.

```zig
/// Reference to a buffer (polynomial data, accumulator tables, etc.)
pub const BufferHandle = enum(u32) { _ };

/// Reference to a single value (challenge, commitment, coefficients)
pub const ValueHandle = enum(u32) { _ };

/// Reference to a future value (not yet computed)
pub const PromiseHandle = enum(u32) { _ };

/// Reference to transcript state
pub const TranscriptHandle = enum(u32) { _ };
```

---

## Buffer Registry

The runtime manages buffer allocation and lifecycle. Buffers have explicit layouts matching s6k requirements.

```zig
pub const BufferRegistry = struct {
    // SoA layout for metadata (DOD-aligned)
    ptrs: []?[*]u8,
    lens: []u32,
    capacities: []u32,
    elem_sizes: []u8,
    layouts: []BufferLayout,
    locations: []BufferLocation,

    pub const BufferLayout = enum {
        /// [p₀[0..len], p₁[0..len], ...] - contiguous polynomials
        poly_buffer,
        /// [round₁_data | round₂_data | ...] - accumulator table
        accumulator_table,
        /// [eq[0], eq[1], ...] - eq polynomial evaluations
        eq_table,
        /// Kronecker product of Lagrange bases
        challenge_tensor,
        /// Generic byte buffer
        raw,
    };

    pub const BufferLocation = enum {
        cpu,
        gpu,
        remote,
        evicted,  // Spilled to disk under memory pressure
    };

    // ══════════════════════════════════════════════════════════════
    // Allocation
    // ══════════════════════════════════════════════════════════════

    /// Allocate polynomial buffer with explicit layout
    pub fn allocPolyBuffer(
        self: *BufferRegistry,
        comptime F: type,
        num_polys: usize,
        len: usize,
    ) !BufferHandle;

    /// Allocate accumulator table for SVO
    pub fn allocAccumulatorTable(
        self: *BufferRegistry,
        comptime F: type,
        degree: usize,
        num_rounds: usize,
    ) !BufferHandle;

    /// Allocate eq table
    pub fn allocEqTable(
        self: *BufferRegistry,
        comptime F: type,
        num_vars: usize,
    ) !BufferHandle;

    /// Allocate challenge tensor (typically small, could be stack)
    pub fn allocChallengeTensor(
        self: *BufferRegistry,
        comptime F: type,
        degree: usize,
        max_rounds: usize,
    ) !BufferHandle;

    // ══════════════════════════════════════════════════════════════
    // Access
    // ══════════════════════════════════════════════════════════════

    /// Get pointer for CPU access (asserts location == .cpu)
    pub fn getPtr(self: *BufferRegistry, comptime T: type, handle: BufferHandle) [*]T;

    /// Get slice for CPU access
    pub fn getSlice(self: *BufferRegistry, comptime T: type, handle: BufferHandle) []T;

    /// Get buffer metadata
    pub fn getLayout(self: *BufferRegistry, handle: BufferHandle) BufferLayout;
    pub fn getLen(self: *BufferRegistry, handle: BufferHandle) u32;

    // ══════════════════════════════════════════════════════════════
    // Lifecycle
    // ══════════════════════════════════════════════════════════════

    /// Release buffer
    pub fn free(self: *BufferRegistry, handle: BufferHandle) void;

    /// Update length after fold (buffer shrinks logically)
    pub fn setLen(self: *BufferRegistry, handle: BufferHandle, new_len: u32) void;

    // ══════════════════════════════════════════════════════════════
    // Transfer
    // ══════════════════════════════════════════════════════════════

    /// Transfer to GPU memory (async)
    pub fn toGpu(self: *BufferRegistry, handle: BufferHandle) !TransferJob;

    /// Transfer from GPU memory (async)
    pub fn fromGpu(self: *BufferRegistry, handle: BufferHandle) !TransferJob;

    /// Upload data from caller memory
    pub fn upload(self: *BufferRegistry, handle: BufferHandle, data: []const u8) !void;

    /// Download data to caller memory
    pub fn download(self: *BufferRegistry, handle: BufferHandle, out: []u8) !void;
};
```

---

## Job Graph

Protocols build job graphs that declare dependencies explicitly.

```zig
pub const JobId = enum(u32) { _ };

pub const JobGraph = struct {
    jobs: []const Job,
    /// dependencies[i] = job indices that job i depends on
    dependencies: []const []const JobId,

    pub fn getReadyJobs(self: *const JobGraph, completed: []const bool) []JobId {
        // Return jobs whose dependencies are all completed
    }
};

pub const JobGraphBuilder = struct {
    jobs: std.ArrayList(Job),
    deps: std.ArrayList(std.ArrayList(JobId)),

    pub fn init(allocator: Allocator) JobGraphBuilder;

    /// Add a job, return its ID
    pub fn add(self: *JobGraphBuilder, job: Job) JobId;

    /// Declare dependency: `dependent` waits for `dependency`
    pub fn dependsOn(self: *JobGraphBuilder, dependent: JobId, dependency: JobId) void;

    /// Create a new promise handle
    pub fn promise(self: *JobGraphBuilder) PromiseHandle;

    /// Build immutable graph
    pub fn build(self: *JobGraphBuilder) JobGraph;
};
```

---

## Runtime

The core execution engine.

```zig
pub const Runtime = struct {
    // Backends
    cpu_pool: *ThreadPool,
    gpu_queue: ?*GpuQueue,
    io_pool: *IoPool,

    // State
    allocator: Allocator,
    buffers: BufferRegistry,
    promises: PromiseRegistry,
    values: ValueRegistry,
    transcripts: TranscriptRegistry,

    // Thresholds
    const GPU_THRESHOLD = 1 << 16;      // 64K elements → GPU
    const CHUNK_THRESHOLD = 1 << 14;    // 16K elements per chunk

    // ══════════════════════════════════════════════════════════════
    // Execution
    // ══════════════════════════════════════════════════════════════

    /// Execute job graph to completion
    pub fn execute(self: *Runtime, graph: JobGraph) !void {
        var completed = try self.allocator.alloc(bool, graph.jobs.len);
        defer self.allocator.free(completed);
        @memset(completed, false);

        while (!allComplete(completed)) {
            const ready = graph.getReadyJobs(completed);

            for (ready) |job_id| {
                self.dispatch(graph.jobs[@intFromEnum(job_id)], job_id);
            }

            const event = try self.waitForCompletion();
            completed[@intFromEnum(event.job_id)] = true;
        }
    }

    /// Execute and return specific promise value
    pub fn executeAndAwait(
        self: *Runtime,
        comptime T: type,
        graph: JobGraph,
        result: PromiseHandle,
    ) !T {
        try self.execute(graph);
        return self.promises.get(T, result);
    }

    // ══════════════════════════════════════════════════════════════
    // Dispatch
    // ══════════════════════════════════════════════════════════════

    fn dispatch(self: *Runtime, job: Job, job_id: JobId) void {
        switch (job) {
            // Heavy compute - consider GPU or chunking
            .compute_round,
            .compute_round_with_eq,
            .compute_round_split_eq,
            .fold_in_place,
            .fold_to_ext,
            => |j| {
                const size = self.buffers.getLen(j.buffer);

                if (self.gpu_queue != null and size > GPU_THRESHOLD) {
                    self.gpu_queue.?.submit(job, job_id);
                } else if (size > CHUNK_THRESHOLD * 2) {
                    self.splitAndSubmit(job, job_id);
                } else {
                    self.cpu_pool.submit(job, job_id);
                }
            },

            // Accumulator precomputation - always parallelize
            .precompute_accumulators => {
                self.splitAndSubmit(job, job_id);
            },

            // Small/fast - run inline
            .lagrange_basis,
            .extend_tensor,
            .check_sum,
            .eval_at,
            .transcript_absorb,
            .transcript_squeeze,
            => {
                self.executeInline(job, job_id);
            },

            // IO-bound
            .merkle_commit, .merkle_open => {
                self.io_pool.submit(job, job_id);
            },

            // Chunk operations - direct to CPU pool
            .compute_round_chunk,
            .reduce_coeffs,
            .precompute_accumulators_chunk,
            .reduce_accumulators,
            => {
                self.cpu_pool.submit(job, job_id);
            },

            else => self.cpu_pool.submit(job, job_id),
        }
    }

    fn splitAndSubmit(self: *Runtime, job: Job, job_id: JobId) void {
        // Create chunk jobs + reduce job
        // Submit chunks to CPU pool
        // Reduce job depends on all chunks
    }

    fn executeInline(self: *Runtime, job: Job, job_id: JobId) void {
        // Execute immediately on coordinator thread
        self.executeJob(job);
        self.markComplete(job_id);
    }

    // ══════════════════════════════════════════════════════════════
    // Job Execution (calls s6k primitives)
    // ══════════════════════════════════════════════════════════════

    fn executeJob(self: *Runtime, job: Job) void {
        switch (job) {
            .compute_round => |j| {
                const buf = self.buffers.getPtr(F, j.buffer);
                const coeffs = s6k(F).computeRound(j.degree, buf, j.len);
                self.promises.fulfill(j.output, coeffs);
            },

            .compute_round_split_eq => |j| {
                const buf = self.buffers.getPtr(F, j.buffer);
                const eq_L = self.buffers.getPtr(F, j.eq_L);
                const eq_R = self.buffers.getPtr(F, j.eq_R);
                const coeffs = s6k(F).computeRoundSplitEq(
                    j.degree, buf, j.len,
                    eq_L, j.eq_L_len, eq_R, j.eq_R_len,
                );
                self.promises.fulfill(j.output, coeffs);
            },

            .fold_in_place => |j| {
                const buf = self.buffers.getPtr(F, j.buffer);
                const challenge = self.values.get(F, j.challenge);
                const new_len = s6k(F).foldInPlace(j.num_polys, buf, j.len, challenge);
                self.buffers.setLen(j.buffer, new_len);
                self.promises.fulfill(j.output_len, new_len);
            },

            // ... other jobs call corresponding s6k primitives
        }
    }
};
```

---

## Protocol Usage

### Example: Zerocheck with Algorithm 6

```zig
const S = s6k.Strategy(M31, M31Ext3, 2, 2);
const L0 = 5;  // SVO rounds

pub fn buildZerocheckGraph(
    runtime: *Runtime,
    num_vars: usize,
    poly_buffer: BufferHandle,
    w_challenges: BufferHandle,
) !JobGraph {
    var b = JobGraphBuilder.init(runtime.allocator);

    // Allocate runtime-managed buffers
    const accum_table = try runtime.buffers.allocAccumulatorTable(S.Ext, S.Degree, L0);
    const tensor = try runtime.buffers.allocChallengeTensor(S.Ext, S.Degree, L0);
    const eq_L = try runtime.buffers.allocEqTable(S.Ext, num_vars / 2);
    const eq_R = try runtime.buffers.allocEqTable(S.Ext, num_vars / 2);
    const ext_buffer = try runtime.buffers.allocPolyBuffer(S.Ext, S.NumPolys, 1 << (num_vars - 1));

    // ══════════════════════════════════════════════════════════════
    // Phase 0: Precomputation (parallel)
    // ══════════════════════════════════════════════════════════════

    const precompute = b.add(.{ .precompute_accumulators = .{
        .buffer = poly_buffer,
        .len = 1 << num_vars,
        .num_rounds = L0,
        .degree = S.Degree,
        .output = accum_table,
        .offsets_output = ...,
    }});

    const build_eq_L = b.add(.{ .build_eq_table = .{
        .challenges = w_challenges,
        .num_vars = num_vars / 2,
        .output = eq_L,
    }});

    const build_eq_R = b.add(.{ .build_eq_table = .{
        .challenges = w_challenges,  // offset for right half
        .num_vars = num_vars / 2,
        .output = eq_R,
    }});
    // precompute, build_eq_L, build_eq_R run in parallel!

    // ══════════════════════════════════════════════════════════════
    // Phase 1: SVO Rounds (1..L0)
    // ══════════════════════════════════════════════════════════════

    var prev_job = precompute;
    var tensor_len: u32 = 1;

    for (1..L0 + 1) |round| {
        const num_v = std.math.pow(u32, S.Degree + 1, round - 1);

        const compute = b.add(.{ .compute_round_from_accumulators = .{
            .challenge_tensor = tensor,
            .tensor_len = tensor_len,
            .accumulators = accum_table,
            .num_v = num_v,
            .degree = S.Degree,
            .output = b.promise(),
        }});
        b.dependsOn(compute, prev_job);

        const absorb = b.add(.{ .transcript_absorb = .{
            .transcript = transcript,
            .data = compute.output,
        }});
        b.dependsOn(absorb, compute);

        const squeeze = b.add(.{ .transcript_squeeze = .{
            .transcript = transcript,
            .output = b.promise(),
        }});
        b.dependsOn(squeeze, absorb);

        const basis = b.add(.{ .lagrange_basis = .{
            .point = squeeze.output,
            .degree = S.Degree,
            .output = b.promise(),
        }});
        b.dependsOn(basis, squeeze);

        const extend = b.add(.{ .extend_tensor = .{
            .tensor = tensor,
            .tensor_len = tensor_len,
            .basis = basis.output,
            .degree = S.Degree,
            .output_len = b.promise(),
        }});
        b.dependsOn(extend, basis);

        tensor_len *= (S.Degree + 1);
        prev_job = extend;
    }

    // ══════════════════════════════════════════════════════════════
    // Phase 2: Transition (materialize folded state)
    // ══════════════════════════════════════════════════════════════

    const materialize = b.add(.{ .materialize_from_accumulators = .{
        .accum_table = accum_table,
        .tensor = tensor,
        .output = ext_buffer,
    }});
    b.dependsOn(materialize, prev_job);
    prev_job = materialize;

    // ══════════════════════════════════════════════════════════════
    // Phase 3: Standard Rounds with Split Eq (L0+1..num_vars)
    // ══════════════════════════════════════════════════════════════

    var current_len: u32 = 1 << (num_vars - L0);
    var eq_L_len: u32 = 1 << (num_vars / 2 - L0);
    var eq_R_len: u32 = 1 << (num_vars / 2);

    for (L0 + 1..num_vars) |round| {
        const compute = b.add(.{ .compute_round_split_eq = .{
            .buffer = ext_buffer,
            .len = current_len,
            .eq_L = eq_L,
            .eq_L_len = eq_L_len,
            .eq_R = eq_R,
            .eq_R_len = eq_R_len,
            .degree = S.Degree,
            .output = b.promise(),
        }});
        b.dependsOn(compute, prev_job);
        b.dependsOn(compute, build_eq_L);
        b.dependsOn(compute, build_eq_R);

        const absorb = b.add(.{ .transcript_absorb = .{
            .transcript = transcript,
            .data = compute.output,
        }});
        b.dependsOn(absorb, compute);

        const squeeze = b.add(.{ .transcript_squeeze = .{
            .transcript = transcript,
            .output = b.promise(),
        }});
        b.dependsOn(squeeze, absorb);

        const fold = b.add(.{ .fold_in_place = .{
            .buffer = ext_buffer,
            .len = current_len,
            .num_polys = S.NumPolys,
            .challenge = squeeze.output,
            .output_len = b.promise(),
        }});
        b.dependsOn(fold, squeeze);

        const fold_eq_L = b.add(.{ .fold_eq_table = .{
            .eq_table = eq_L,
            .len = eq_L_len,
            .challenge = squeeze.output,
            .output_len = b.promise(),
        }});
        b.dependsOn(fold_eq_L, squeeze);

        current_len /= 2;
        eq_L_len /= 2;
        prev_job = fold;
    }

    return b.build();
}

// Protocol usage
pub fn proveZerocheck(self: *ZerocheckProver, runtime: *Runtime) !Proof {
    const graph = try buildZerocheckGraph(
        runtime,
        self.num_vars,
        self.poly_buffer,
        self.w_challenges,
    );

    try runtime.execute(graph);

    return self.extractProof(runtime);
}
```

### Example: Basefold with Pipeline Parallelism

```zig
pub fn buildBasefoldGraph(
    runtime: *Runtime,
    num_vars: usize,
    f_buffer: BufferHandle,
    eq_buffer: BufferHandle,
    transcript: TranscriptHandle,
) !JobGraph {
    var b = JobGraphBuilder.init(runtime.allocator);

    const ext_buffer = try runtime.buffers.allocPolyBuffer(S.Ext, 2, 1 << (num_vars - 1));

    // ══════════════════════════════════════════════════════════════
    // Round 0 (BaseF) - Commit and compute in parallel
    // ══════════════════════════════════════════════════════════════

    const commit_0 = b.add(.{ .merkle_commit = .{
        .data = f_buffer,
        .len = 1 << num_vars,
        .output = b.promise(),
    }});

    const round_0 = b.add(.{ .compute_round = .{
        .buffer = f_buffer,  // Both f and eq in same buffer layout
        .len = 1 << num_vars,
        .degree = 2,
        .output = b.promise(),
    }});
    // commit_0 and round_0 run in parallel!

    const absorb_commit_0 = b.add(.{ .transcript_absorb = .{
        .transcript = transcript,
        .data = commit_0.output,
    }});
    b.dependsOn(absorb_commit_0, commit_0);

    const absorb_round_0 = b.add(.{ .transcript_absorb = .{
        .transcript = transcript,
        .data = round_0.output,
    }});
    b.dependsOn(absorb_round_0, round_0);
    b.dependsOn(absorb_round_0, absorb_commit_0);  // Order matters for transcript

    const squeeze_0 = b.add(.{ .transcript_squeeze = .{
        .transcript = transcript,
        .output = b.promise(),
    }});
    b.dependsOn(squeeze_0, absorb_round_0);

    const fold_0 = b.add(.{ .fold_to_ext = .{
        .base_buffer = f_buffer,
        .len = 1 << num_vars,
        .num_polys = 2,
        .challenge = squeeze_0.output,
        .ext_buffer = ext_buffer,
        .output_len = b.promise(),
    }});
    b.dependsOn(fold_0, squeeze_0);

    // ══════════════════════════════════════════════════════════════
    // Rounds 1..n (ExtF) - Pipeline commit with next round
    // ══════════════════════════════════════════════════════════════

    var prev_fold = fold_0;
    var current_len: u32 = 1 << (num_vars - 1);

    for (1..num_vars) |round| {
        // Commit and compute in parallel
        const commit_i = b.add(.{ .merkle_commit = .{
            .data = ext_buffer,
            .len = current_len,
            .output = b.promise(),
        }});
        b.dependsOn(commit_i, prev_fold);

        const round_i = b.add(.{ .compute_round = .{
            .buffer = ext_buffer,
            .len = current_len,
            .degree = 2,
            .output = b.promise(),
        }});
        b.dependsOn(round_i, prev_fold);
        // commit_i and round_i run in parallel!

        const absorb_commit = b.add(.{ .transcript_absorb = .{
            .transcript = transcript,
            .data = commit_i.output,
        }});
        b.dependsOn(absorb_commit, commit_i);

        const absorb_round = b.add(.{ .transcript_absorb = .{
            .transcript = transcript,
            .data = round_i.output,
        }});
        b.dependsOn(absorb_round, round_i);
        b.dependsOn(absorb_round, absorb_commit);

        const squeeze_i = b.add(.{ .transcript_squeeze = .{
            .transcript = transcript,
            .output = b.promise(),
        }});
        b.dependsOn(squeeze_i, absorb_round);

        const fold_i = b.add(.{ .fold_in_place = .{
            .buffer = ext_buffer,
            .len = current_len,
            .num_polys = 2,
            .challenge = squeeze_i.output,
            .output_len = b.promise(),
        }});
        b.dependsOn(fold_i, squeeze_i);

        current_len /= 2;
        prev_fold = fold_i;
    }

    return b.build();
}
```

---

## Migration Path

### Phase 1: Primitives Only (S6K_SYNC)

Protocol calls s6k primitives directly:

```zig
const coeffs = s6k(F).computeRound(2, buffer.ptr, len);
transcript.absorb(coeffs);
const challenge = transcript.squeeze();
len = s6k(F).foldInPlace(2, buffer.ptr, len, challenge);
```

### Phase 2: Runtime Wrapper (Sequential)

Introduce runtime, execute jobs synchronously:

```zig
const graph = buildSumcheckGraph(...);
try runtime.execute(graph);  // Runs jobs one at a time
```

Same semantics, validates the API.

### Phase 3: Thread Pool

Runtime dispatches to thread pool:
- Multiple sumchecks run in parallel
- Large polynomials chunked across threads
- Still appears synchronous to protocol

### Phase 4: Async IO

Overlap compute and IO:
- Merkle commits run on IO pool
- Compute continues while commit in flight

### Phase 5: GPU Backend

Offload to GPU:
- Runtime decides based on size threshold
- Buffer transfer managed automatically
- Transparent to protocol

### Phase 6: Distributed

Spread across machines:
- Handles serialize for network transfer
- Runtime coordinates via RPC
- Same job graph, different execution

---

## Design Constraints

These constraints ensure the model works:

### 1. Primitives Must Be Stateless

s6k primitives take all inputs as parameters, return all outputs. No hidden state.

### 2. Buffers Must Be Handle-Based

Protocol uses handles, not raw pointers. Runtime manages lifecycle.

### 3. Dependencies Must Be Explicit

Job graph declares what depends on what. No implicit ordering.

### 4. Side Effects Must Be Jobs

Transcript operations, IO are jobs with explicit dependencies.

---

## Open Questions

1. **Error Propagation**: How do job failures cascade? Cancel dependent jobs?

2. **Backpressure**: What if jobs produce faster than consumers? Queue limits?

3. **Cancellation**: Can we cancel an in-flight job graph?

4. **Checkpointing**: Can we save/restore job graph execution state?

5. **Profiling**: How do we trace job execution for optimization?

6. **Memory Pressure**: How does runtime handle OOM? Spill to disk?

7. **GPU Memory**: When to transfer between CPU/GPU? Heuristics?

8. **Distributed Scheduling**: How to partition job graph across machines?

---

## References

- S6K_SYNC.md - s6k primitive specification
- [Speeding Up Sum-Check Proving](https://eprint.iacr.org/2025/1117.pdf) - Algorithms 1-6
