# Unified Prover Configuration

## Goal

Single comptime config flows through entire proving stack: context, PCS, merkle.

## Current State (Option A)

Separate comptime params at each layer:

```zig
// merkle/tree.zig
MerkleTree(F, Hasher, max_depth: u8)

// pcs/basefold.zig
Basefold(F, Config{ .max_vars, .num_queries })

// core/context.zig
ProverContext(F, max_vars: usize, Options{ .tracing })
```

User must ensure `basefold.max_vars == context.max_vars == merkle.max_depth`.

## Target State (Option B)

Unified config in `core/context.zig`:

```zig
pub const ProverConfig = struct {
    max_vars: comptime_int = 20,    // 2^20 = 1M max evals
    num_queries: comptime_int = 32, // soundness parameter
    tracing: bool = false,
};

pub fn ProverContext(comptime F: type, comptime config: ProverConfig) type {
    // Derive PCS from same config
    pub const PCS = Basefold(F, .{
        .max_vars = config.max_vars,
        .num_queries = config.num_queries,
    });

    // Scratch buffers sized to max
    const max_size = 1 << config.max_vars;
    scratch: [max_size]F,

    // Transcript, arena, tracing...
}
```

## Benefits

1. **Single source of truth** - one config, no mismatch bugs
2. **Context owns resources** - scratch, arena, transcript already there
3. **Type safety** - incompatible configs caught at comptime
4. **Simpler API** - user instantiates one type, gets everything

## Migration Path

1. [x] Add `max_vars` to basefold config
2. [x] Add `max_depth` param to merkle tree
3. [ ] Move `ProverConfig` to context.zig with all fields
4. [ ] Have context instantiate PCS internally
5. [ ] Basefold uses context's scratch/transcript instead of creating own
6. [ ] Update prove/verify to take context reference

## Size Calculations

With `max_vars = 20`, `num_queries = 32`:

| Component | Size |
|-----------|------|
| Merkle.Proof | 20 * 32 = 640 bytes |
| LayerProof | ~1.3 KB |
| QueryProof | 20 * 1.3KB = ~26 KB |
| OpeningProof | 32 * 26KB = ~832 KB |

With `max_vars = 16`, `num_queries = 32`:

| Component | Size |
|-----------|------|
| Merkle.Proof | 16 * 32 = 512 bytes |
| LayerProof | ~1 KB |
| QueryProof | 16 * 1KB = ~16 KB |
| OpeningProof | 32 * 16KB = ~512 KB |
