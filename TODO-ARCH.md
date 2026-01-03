# Architecture Migration Plan

This document outlines the migration from the current architecture to a composable, DOD-aligned proof system.

## Current State

```
src/
├── sumcheck.zig           # Linear sumcheck (degree-1)
├── product_sumcheck.zig   # Product sumcheck (degree-2), separate impl
├── protocol.zig           # Sumcheck + Merkle wrapper (unused by VM)
├── pcs/
│   └── basefold.zig       # PCS with embedded sumcheck logic
└── vm/
    ├── prover.zig         # Calls Basefold.prove() directly
    └── verifier.zig       # Calls Basefold.verify() directly
```

**Problems:**
1. Sumcheck reimplemented for each degree (linear vs product)
2. Basefold embeds sumcheck rather than composing with it
3. No clean way to swap PCS (Basefold → FRI → Brakedown)
4. No clean way to swap IOP (sumcheck → GKR → lookups)
5. Memory ownership scattered across layers

## Target Architecture

```
src/
├── core/
│   ├── context.zig        # ProverContext, VerifierContext (memory ownership)
│   └── transcript.zig     # Fiat-Shamir (moved here)
├── iop/
│   ├── sumcheck.zig       # Generic sumcheck via comptime config
│   ├── gkr.zig            # GKR protocol (uses sumcheck)
│   └── lookup.zig         # Logup/lookup arguments (uses sumcheck)
├── pcs/
│   ├── pcs.zig            # PCS config types and composition
│   ├── basefold.zig       # Basefold impl
│   ├── fri.zig            # FRI impl (future)
│   └── brakedown.zig      # Brakedown impl (future)
├── protocol/
│   ├── protocol.zig       # Protocol composition layer
│   └── configs.zig        # Pre-built configurations
├── poly/                  # (unchanged)
├── fields/                # (unchanged)
└── vm/                    # Uses protocol layer, not PCS directly
```

## Core Types

### ProverContext (Memory Owner)

```zig
// src/core/context.zig

pub fn ProverContext(comptime F: type, comptime max_vars: usize) type {
    const max_size = 1 << max_vars;

    return struct {
        const Self = @This();

        /// Backing allocator for arena
        backing: Allocator,

        /// Arena for proof data - freed all at once
        arena: std.heap.ArenaAllocator,

        /// Primary scratch buffer for in-place folding
        /// Reused every round - size: max_size
        scratch: []F,

        /// Secondary scratch for ops needing two buffers
        scratch_aux: []F,

        /// Transcript for Fiat-Shamir challenges
        transcript: Transcript,

        pub fn init(backing: Allocator) !Self {
            const arena = std.heap.ArenaAllocator.init(backing);
            return .{
                .backing = backing,
                .arena = arena,
                .scratch = try backing.alloc(F, max_size),
                .scratch_aux = try backing.alloc(F, max_size),
                .transcript = Transcript.init(),
            };
        }

        pub fn deinit(self: *Self) void {
            self.arena.deinit();
            self.backing.free(self.scratch);
            self.backing.free(self.scratch_aux);
        }

        /// Reset for next proof - keeps allocations, clears state
        pub fn reset(self: *Self) void {
            _ = self.arena.reset(.retain_capacity);
            self.transcript.reset();
        }

        /// Allocate from arena (lives until reset/deinit)
        pub fn alloc(self: *Self, comptime T: type, n: usize) ![]T {
            return self.arena.allocator().alloc(T, n);
        }
    };
}

pub fn VerifierContext(comptime F: type) type {
    return struct {
        transcript: Transcript,
        // Verifier needs minimal memory - mostly just transcript

        pub fn init() @This() {
            return .{ .transcript = Transcript.init() };
        }
    };
}
```

### Generic Sumcheck

```zig
// src/iop/sumcheck.zig

pub const SumcheckConfig = struct {
    /// Degree of round polynomial (1 = linear, 2 = product, etc.)
    degree: comptime_int = 1,

    /// Number of polynomials being composed
    /// 1 = single poly, 2 = product of two, etc.
    num_polys: comptime_int = 1,

    /// Challenge field (for small fields needing extension)
    /// null = same as base field
    ChallengeField: ?type = null,

    /// Batch multiple sumcheck instances
    batch_size: comptime_int = 1,
};

pub fn Sumcheck(comptime F: type, comptime config: SumcheckConfig) type {
    const CF = config.ChallengeField orelse F;
    const degree = config.degree;

    return struct {
        const Self = @This();

        /// Round polynomial: evaluations at 0, 1, ..., degree
        pub const RoundPoly = [degree + 1]F;

        pub const ProveResult = struct {
            rounds: []const RoundPoly,
            challenges: []const CF,
            final_evals: [config.num_polys]F,
        };

        pub const VerifyResult = struct {
            challenges: []const CF,
            final_claim: F,
            valid: bool,
        };

        /// Compute single round polynomial from current evaluations
        /// polys: array of polynomial evaluations (1 for linear, 2 for product)
        pub fn computeRound(polys: [config.num_polys][]const F) RoundPoly {
            var result: RoundPoly = undefined;

            // Evaluate at each point 0, 1, ..., degree
            inline for (0..degree + 1) |eval_point| {
                result[eval_point] = computeRoundEval(polys, eval_point);
            }

            return result;
        }

        fn computeRoundEval(polys: [config.num_polys][]const F, point: usize) F {
            const half = polys[0].len / 2;
            var sum = F.zero;

            var i: usize = 0;
            while (i < half) : (i += 1) {
                // Interpolate each poly at point, then combine
                var term = F.one;
                inline for (polys) |poly| {
                    const lo = poly[i];
                    const hi = poly[i + half];
                    const interpolated = interpolateAt(lo, hi, point);
                    term = term.mul(interpolated);
                }
                sum = sum.add(term);
            }

            return sum;
        }

        fn interpolateAt(lo: F, hi: F, point: usize) F {
            // Linear interpolation: lo + point * (hi - lo)
            // For point=0: lo, point=1: hi, point=2: 2*hi - lo, etc.
            if (point == 0) return lo;
            if (point == 1) return hi;
            const delta = hi.sub(lo);
            return lo.add(delta.mul(F.fromInt(point)));
        }

        /// Prove sumcheck - returns views into ctx memory
        pub fn prove(
            ctx: anytype,  // *ProverContext
            polys: [config.num_polys][]const F,
            claimed_sum: F,
        ) !ProveResult {
            const num_vars = @ctz(polys[0].len);

            // Allocate results in arena
            const rounds = try ctx.alloc(RoundPoly, num_vars);
            const challenges = try ctx.alloc(CF, num_vars);

            // Copy polys to scratch for in-place folding
            var work_polys: [config.num_polys][]F = undefined;
            inline for (0..config.num_polys) |p| {
                const buf = if (p == 0) ctx.scratch else ctx.scratch_aux;
                @memcpy(buf[0..polys[p].len], polys[p]);
                work_polys[p] = buf[0..polys[p].len];
            }

            // Absorb claimed sum
            ctx.transcript.absorbField(claimed_sum);

            // Main loop
            for (0..num_vars) |round| {
                // Compute round polynomial
                const round_poly = computeRound(work_polys);
                rounds[round] = round_poly;

                // Absorb and squeeze challenge
                inline for (round_poly) |eval| {
                    ctx.transcript.absorbField(eval);
                }
                challenges[round] = ctx.transcript.squeezeField(CF);

                // Fold all polynomials
                inline for (&work_polys) |*wp| {
                    wp.* = fold(wp.*, challenges[round]);
                }
            }

            // Extract final evaluations
            var final_evals: [config.num_polys]F = undefined;
            inline for (0..config.num_polys) |p| {
                final_evals[p] = work_polys[p][0];
            }

            return .{
                .rounds = rounds,
                .challenges = challenges,
                .final_evals = final_evals,
            };
        }

        /// Verify sumcheck (pure, no allocation)
        pub fn verify(
            transcript: *Transcript,
            rounds: []const RoundPoly,
            claimed_sum: F,
        ) VerifyResult {
            const num_vars = rounds.len;
            var challenges = std.BoundedArray(CF, 64){};

            transcript.absorbField(claimed_sum);
            var current_claim = claimed_sum;

            for (rounds) |round_poly| {
                // Check: round_poly(0) + round_poly(1) = current_claim
                const sum = round_poly[0].add(round_poly[1]);
                if (!sum.eql(current_claim)) {
                    return .{ .challenges = &.{}, .final_claim = F.zero, .valid = false };
                }

                // Absorb and derive challenge
                inline for (round_poly) |eval| {
                    transcript.absorbField(eval);
                }
                const challenge = transcript.squeezeField(CF);
                challenges.append(challenge) catch unreachable;

                // Evaluate round poly at challenge for next claim
                current_claim = evalRoundPoly(round_poly, challenge);
            }

            return .{
                .challenges = challenges.constSlice(),
                .final_claim = current_claim,
                .valid = true,
            };
        }

        fn evalRoundPoly(poly: RoundPoly, point: CF) F {
            // Lagrange interpolation through degree+1 points
            // For degree=1: poly[0] + point * (poly[1] - poly[0])
            // For degree=2: Lagrange through (0,p0), (1,p1), (2,p2)
            if (degree == 1) {
                return poly[0].add(point.mul(poly[1].sub(poly[0])));
            } else {
                return lagrangeEval(poly, point);
            }
        }

        fn lagrangeEval(evals: RoundPoly, point: CF) F {
            // General Lagrange interpolation
            var result = F.zero;
            inline for (0..degree + 1) |i| {
                var basis = F.one;
                inline for (0..degree + 1) |j| {
                    if (i != j) {
                        // basis *= (point - j) / (i - j)
                        const num = point.sub(F.fromInt(j));
                        const denom = F.fromInt(i).sub(F.fromInt(j));
                        basis = basis.mul(num).mul(denom.inv());
                    }
                }
                result = result.add(evals[i].mul(basis));
            }
            return result;
        }

        /// Fold polynomial in-place, returns smaller slice
        pub fn fold(evals: []F, challenge: CF) []F {
            const half = evals.len / 2;
            for (0..half) |i| {
                // new[i] = lo + challenge * (hi - lo)
                const lo = evals[i];
                const hi = evals[i + half];
                evals[i] = lo.add(challenge.mul(hi.sub(lo)));
            }
            return evals[0..half];
        }
    };
}

// Convenience aliases
pub fn LinearSumcheck(comptime F: type) type {
    return Sumcheck(F, .{ .degree = 1, .num_polys = 1 });
}

pub fn ProductSumcheck(comptime F: type) type {
    return Sumcheck(F, .{ .degree = 2, .num_polys = 2 });
}
```

### PCS Interface

```zig
// src/pcs/pcs.zig

pub const PCSConfig = struct {
    /// Commitment type
    Commitment: type,

    /// Opening proof type
    OpeningProof: type,

    /// Number of queries for soundness
    num_queries: comptime_int = 80,

    /// Sumcheck config used internally (for folding-based PCS)
    sumcheck: ?SumcheckConfig = null,
};

/// Generic PCS operations - implemented by Basefold, FRI, etc.
pub fn PCS(comptime F: type, comptime Impl: type) type {
    return struct {
        pub const Commitment = Impl.Commitment;
        pub const OpeningProof = Impl.OpeningProof;

        pub fn commit(ctx: anytype, evals: []const F) !Commitment {
            return Impl.commit(ctx, evals);
        }

        pub fn open(
            ctx: anytype,
            evals: []const F,
            point: []const F,
            claimed_value: F,
        ) !OpeningProof {
            return Impl.open(ctx, evals, point, claimed_value);
        }

        pub fn verify(
            ctx: anytype,
            commitment: Commitment,
            point: []const F,
            claimed_value: F,
            proof: *const OpeningProof,
        ) !bool {
            return Impl.verify(ctx, commitment, point, claimed_value, proof);
        }
    };
}
```

### Basefold Implementation

```zig
// src/pcs/basefold.zig

pub fn Basefold(comptime F: type, comptime config: BasefoldConfig) type {
    const SC = Sumcheck(F, .{ .degree = 2, .num_polys = 2 });
    const Merkle = MerkleTree(F, config.hasher);

    return struct {
        pub const Commitment = Merkle.Root;

        pub const OpeningProof = struct {
            sumcheck: SC.ProveResult,
            layer_commitments: []const Commitment,
            final_f: F,
            final_eq: F,
            queries: []const QueryProof,
        };

        pub const QueryProof = struct {
            layer_proofs: []const LayerProof,
        };

        pub const LayerProof = struct {
            left: F,
            right: F,
            merkle_proof: Merkle.Proof,
        };

        pub fn commit(ctx: anytype, evals: []const F) !Commitment {
            return Merkle.commit(ctx, evals);
        }

        pub fn open(
            ctx: anytype,
            evals: []const F,
            point: []const F,
            claimed_value: F,
        ) !OpeningProof {
            const num_vars = point.len;

            // Build eq polynomial
            const eq_evals = try ctx.alloc(F, evals.len);
            buildEqTable(eq_evals, point);

            // Run sumcheck with layer commitments
            const result = try runSumcheckWithCommitments(
                ctx, evals, eq_evals, claimed_value
            );

            // Generate query proofs
            const query_indices = deriveQueryIndices(&ctx.transcript, config.num_queries);
            const queries = try generateQueryProofs(ctx, result.layers, query_indices);

            return .{
                .sumcheck = result.sumcheck,
                .layer_commitments = result.commitments,
                .final_f = result.final_f,
                .final_eq = result.final_eq,
                .queries = queries,
            };
        }

        pub fn verify(
            ctx: anytype,
            commitment: Commitment,
            point: []const F,
            claimed_value: F,
            proof: *const OpeningProof,
        ) !bool {
            // 1. Verify sumcheck
            const sc_result = SC.verify(
                &ctx.transcript,
                proof.sumcheck.rounds,
                claimed_value,
            );
            if (!sc_result.valid) return false;

            // 2. Check final claim
            const expected_final = proof.final_f.mul(proof.final_eq);
            if (!expected_final.eql(sc_result.final_claim)) return false;

            // 3. Verify eq polynomial
            const expected_eq = evalEq(sc_result.challenges, point);
            if (!expected_eq.eql(proof.final_eq)) return false;

            // 4. Verify query proofs
            const query_indices = deriveQueryIndices(&ctx.transcript, config.num_queries);
            return verifyQueries(commitment, proof, query_indices, sc_result.challenges);
        }

        // ... internal helpers
    };
}
```

### Protocol Composition

```zig
// src/protocol/protocol.zig

pub const ProtocolConfig = struct {
    /// IOP for constraint satisfaction (sumcheck/zerocheck)
    ConstraintIOP: type,

    /// Polynomial commitment scheme
    PCS: type,

    /// Optional: GKR for layered circuits
    GKR: ?type = null,

    /// Optional: Lookup arguments
    Lookup: ?type = null,
};

pub fn Protocol(comptime F: type, comptime config: ProtocolConfig) type {
    return struct {
        pub const Proof = struct {
            // Constraint IOP proof
            constraint_proof: config.ConstraintIOP.ProveResult,

            // PCS commitment and opening
            commitment: config.PCS.Commitment,
            opening_proof: config.PCS.OpeningProof,

            // Optional components
            gkr_proof: if (config.GKR) |G| G.Proof else void,
            lookup_proof: if (config.Lookup) |L| L.Proof else void,
        };

        /// Prove constraint polynomial sums to zero (zerocheck)
        pub fn prove(
            ctx: anytype,
            constraint_evals: []const F,
        ) !Proof {
            // 1. Commit to constraint polynomial
            const commitment = try config.PCS.commit(ctx, constraint_evals);

            // 2. Derive random point via Fiat-Shamir
            ctx.transcript.absorb(commitment);
            const point = deriveRandomPoint(&ctx.transcript, @ctz(constraint_evals.len));

            // 3. Build eq polynomial and run zerocheck
            const eq_evals = try ctx.alloc(F, constraint_evals.len);
            buildEqTable(eq_evals, point);

            const constraint_proof = try config.ConstraintIOP.prove(
                ctx,
                .{ constraint_evals, eq_evals },
                F.zero,  // claiming sum is zero
            );

            // 4. Open PCS at challenge point
            const eval_at_point = evalMultilinear(constraint_evals, constraint_proof.challenges);
            const opening_proof = try config.PCS.open(
                ctx,
                constraint_evals,
                constraint_proof.challenges,
                eval_at_point,
            );

            return .{
                .constraint_proof = constraint_proof,
                .commitment = commitment,
                .opening_proof = opening_proof,
                .gkr_proof = if (config.GKR) |_| {} else {},
                .lookup_proof = if (config.Lookup) |_| {} else {},
            };
        }

        pub fn verify(
            ctx: anytype,
            proof: *const Proof,
        ) !bool {
            // 1. Derive same random point
            ctx.transcript.absorb(proof.commitment);
            const point = deriveRandomPoint(&ctx.transcript, proof.constraint_proof.rounds.len);

            // 2. Verify constraint sumcheck
            const sc_result = config.ConstraintIOP.verify(
                &ctx.transcript,
                proof.constraint_proof.rounds,
                F.zero,
            );
            if (!sc_result.valid) return false;

            // 3. Check final evaluation matches claimed
            // For zerocheck: final should be 0 at random point (w.h.p.)

            // 4. Verify PCS opening
            return config.PCS.verify(
                ctx,
                proof.commitment,
                sc_result.challenges,
                sc_result.final_claim,
                &proof.opening_proof,
            );
        }
    };
}
```

### Pre-built Configurations

```zig
// src/protocol/configs.zig

const F = @import("../fields/mersenne31.zig").Mersenne31;

/// Standard configuration: Product sumcheck + Basefold
pub const StandardConfig = ProtocolConfig{
    .ConstraintIOP = Sumcheck(F, .{ .degree = 2, .num_polys = 2 }),
    .PCS = Basefold(F, .{ .num_queries = 80 }),
};

/// Fast configuration: Linear sumcheck + Basefold (less secure, faster)
pub const FastConfig = ProtocolConfig{
    .ConstraintIOP = Sumcheck(F, .{ .degree = 1, .num_polys = 1 }),
    .PCS = Basefold(F, .{ .num_queries = 40 }),
};

/// Future: FRI-based
pub const FRIConfig = ProtocolConfig{
    .ConstraintIOP = Sumcheck(F, .{ .degree = 2, .num_polys = 2 }),
    .PCS = FRI(F, .{ .fold_factor = 4 }),
};

// Instantiate protocols
pub const StandardProtocol = Protocol(F, StandardConfig);
pub const FastProtocol = Protocol(F, FastConfig);
```

## Migration Steps

### Phase 1: Core Infrastructure

1. **Create `src/core/context.zig`**
   - Implement `ProverContext` with arena + scratch buffers
   - Implement `VerifierContext`
   - Move `Transcript` to `src/core/transcript.zig`

2. **Test context isolation**
   - Verify scratch buffer reuse works
   - Verify arena reset clears all proof data
   - Benchmark memory usage vs current impl

### Phase 2: Generic Sumcheck

3. **Create `src/iop/sumcheck.zig`**
   - Implement generic `Sumcheck(F, config)`
   - Support degree 1, 2, 3 via comptime
   - Add `fold()` that returns smaller slice (in-place)

4. **Migrate existing code**
   - Replace `sumcheck.zig` usage with `Sumcheck(F, .{ .degree = 1 })`
   - Replace `product_sumcheck.zig` usage with `Sumcheck(F, .{ .degree = 2 })`
   - Keep old files temporarily for comparison testing

5. **Verify equivalence**
   - Test new generic sumcheck produces identical outputs
   - Benchmark performance (should be equal or better due to inlining)

### Phase 3: PCS Refactor

6. **Create `src/pcs/pcs.zig`**
   - Define `PCSConfig` and generic `PCS` wrapper
   - This is the interface all PCS implementations follow

7. **Refactor Basefold**
   - Update to use generic sumcheck internally
   - Accept `*ProverContext` instead of allocator
   - Use context's scratch buffers for folding

8. **Verify Basefold equivalence**
   - Test against current implementation
   - Ensure proofs are identical (or verify both)

### Phase 4: Protocol Layer

9. **Create `src/protocol/protocol.zig`**
   - Implement `Protocol(F, config)` composition
   - Wire together IOP + PCS

10. **Create `src/protocol/configs.zig`**
    - Define standard configurations
    - Make it easy to swap IOP/PCS

11. **Update VM prover/verifier**
    - Use `Protocol` instead of calling Basefold directly
    - Pass `ProverContext` through

### Phase 5: Cleanup

12. **Remove old files**
    - Delete `sumcheck.zig` (replaced by generic)
    - Delete `product_sumcheck.zig` (replaced by generic)
    - Update `protocol.zig` or remove if redundant

13. **Update imports in `root.zig`**

14. **Documentation**
    - Document configuration options
    - Add examples for custom configurations

## Future Extensions

### GKR (Phase 6)

```zig
// src/iop/gkr.zig
pub fn GKR(comptime F: type, comptime config: GKRConfig) type {
    const SC = Sumcheck(F, config.sumcheck);

    return struct {
        pub fn proveLayer(ctx: anytype, ...) !LayerProof {
            // Uses SC.prove() for each layer
            // Reuses ctx.scratch across layers
        }
    };
}
```

### Lookups (Phase 7)

```zig
// src/iop/lookup.zig
pub fn LogupLookup(comptime F: type) type {
    const SC = Sumcheck(F, .{ .degree = 2, .num_polys = 2 });

    return struct {
        pub fn prove(ctx: anytype, table: []const F, queries: []const F) !Proof {
            // Logup: prove Σ 1/(table[i] - x) = Σ 1/(query[i] - x)
            // Uses fractional sumcheck
        }
    };
}
```

### Alternative PCS (Phase 8+)

```zig
// src/pcs/fri.zig
pub fn FRI(comptime F: type, comptime config: FRIConfig) type {
    // RS-code based, different folding strategy
}

// src/pcs/brakedown.zig
pub fn Brakedown(comptime F: type, comptime config: BrakedownConfig) type {
    // Linear-time, expander-based
}
```

## Memory Layout Summary

```
ProverContext
├── arena: ArenaAllocator
│   └── [proof data: rounds, commitments, query proofs]
│       (freed all at once on reset/deinit)
├── scratch: []F [size = 2^max_vars]
│   └── [reused every round for primary poly folding]
├── scratch_aux: []F [size = 2^max_vars]
│   └── [reused every round for secondary poly (eq, etc)]
└── transcript: Transcript
    └── [Fiat-Shamir state]

Data Flow:
1. Copy input poly → scratch
2. For each round:
   a. Compute round poly from scratch (read-only)
   b. Absorb into transcript
   c. Fold scratch in-place → scratch shrinks by half
3. Allocate round poly in arena (persists in proof)
4. Return proof with views into arena

No per-round allocations. Two fixed buffers reused throughout.
```

## Testing Strategy

```zig
test "generic sumcheck equivalence" {
    // Old implementation
    const old_result = old_sumcheck.prove(allocator, evals, claimed);

    // New implementation
    var ctx = try ProverContext.init(allocator, 20);
    const new_result = try Sumcheck(F, .{ .degree = 1 }).prove(&ctx, .{evals}, claimed);

    // Must be identical
    try expectEqualSlices(RoundPoly, old_result.rounds, new_result.rounds);
}

test "protocol swappability" {
    const constraint_poly = generateTestConstraints();

    // Test with different configurations
    inline for (.{ StandardConfig, FastConfig }) |config| {
        const P = Protocol(F, config);
        var ctx = try ProverContext.init(allocator, 20);

        const proof = try P.prove(&ctx, constraint_poly);
        try expect(try P.verify(&ctx, &proof));
    }
}
```

## Open Questions

1. **Extension fields for M31**: When do we need `ChallengeField` to be different from `F`?
   - Answer: When field is small (M31 = 2^31 - 1), challenges need more bits for soundness
   - Implementation: Add `ExtensionField` type and use for challenges in sumcheck

2. **Batched sumcheck**: How to batch multiple claims efficiently?
   - Answer: Random linear combination of claims, single sumcheck on combined poly
   - Implementation: Add `batch_size` to config, adjust `prove()` signature

3. **Parallelization**: Where are the parallelization opportunities?
   - Round poly computation: sum over half the domain (embarrassingly parallel)
   - Merkle tree building: standard parallel tree construction
   - Query proof generation: independent per query
   - Implementation: Add `thread_pool` to `ProverContext`, use in hot loops
