# Proof System Flow

This document describes the data flow through the proof system, from trace to final proof.

## Core Principle

**Commit before challenge, always.**

The prover must commit to polynomials before any challenges that depend on them are derived. This prevents the prover from crafting polynomials to pass checks after seeing challenges.

With Fiat-Shamir (non-interactive), this means:
```
transcript.absorb(commitment)    // commitment goes into hash state
challenge = transcript.squeeze() // challenge = hash(... || commitment)
```

Prover and verifier maintain identical transcript state. Same absorptions, same order, same squeeze points → same challenges.

## Three-Phase Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: COMMIT                                                │
│                                                                 │
│  Build all polynomials from witness/trace                       │
│  Commit to them (Merkle root, etc.)                             │
│  Absorb commitments into transcript                             │
│                                                                 │
│  Polynomials are now "locked in"                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: IOP (Interactive Oracle Proof)                        │
│                                                                 │
│  Derive challenges from transcript (Fiat-Shamir)                │
│  Run sumcheck / zerocheck / lookups / GKR                       │
│  Each IOP absorbs its rounds, squeezes its challenges           │
│                                                                 │
│  Output: claims about polynomial evaluations                    │
│          e.g., "f(r) = v" for challenge point r                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: OPEN                                                  │
│                                                                 │
│  PCS opens committed polynomials at challenge points            │
│  Proves that f(r) = v for the committed f                       │
│                                                                 │
│  Output: opening proofs                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Flow

### Phase 1: Commit

```
INPUT:  Execution trace (VM state at each step)

        step | pc | opcode | stack[0] | stack[1] | mem_addr | mem_val
        ─────┼────┼────────┼──────────┼──────────┼──────────┼─────────
          0  |  0 |  PUSH  |    0     |    0     |    -     |    -
          1  |  1 |  PUSH  |    5     |    0     |    -     |    -
          2  |  2 |  ADD   |    5     |    3     |    -     |    -
          3  |  3 |  STORE |    8     |    0     |   100    |    8
         ...

PROCESS:

        1. Pad trace to power of 2

        2. Extract column polynomials (each column = multilinear poly)
           - pc_poly[i]      = trace[i].pc
           - opcode_poly[i]  = trace[i].opcode
           - stack0_poly[i]  = trace[i].stack[0]
           - ...

        3. Build constraint polynomial
           - For each row: constraint[i] = Σ selector[op] * check[op](row[i], row[i+1])
           - Valid trace → constraint[i] = 0 for all i

        4. Build lookup helper polynomials (if using lookups)
           - queries: values being looked up
           - table: allowed values
           - multiplicities: count of each table entry used

        5. Commit to all polynomials
           - trace_commitment = MerkleTree.commit(trace_polys)
           - lookup_commitment = MerkleTree.commit(lookup_polys)  // if applicable

        6. Absorb into transcript
           - transcript.absorb(trace_commitment)
           - transcript.absorb(lookup_commitment)  // if applicable

OUTPUT: Commitments (Merkle roots)
        Polynomials locked in, ready for IOP phase
```

### Phase 2: IOP

Multiple IOPs may run in sequence, all sharing the same transcript.

#### 2a. Zerocheck (Constraint Satisfaction)

```
GOAL:   Prove constraint polynomial is zero everywhere
        i.e., Σ constraint(x) = 0 over boolean hypercube {0,1}^n

PROCESS:

        1. Derive random evaluation point
           - r = transcript.squeeze(n field elements)

        2. Build equality polynomial
           - eq(x, r) = Π (xᵢ * rᵢ + (1-xᵢ)(1-rᵢ))
           - Property: Σ_x eq(x, r) = 1, and eq concentrates at r

        3. Run product sumcheck on (constraint, eq)
           - Proves: Σ_x constraint(x) * eq(x, r) = 0
           - This equals constraint(r) * 1 = constraint(r)
           - If constraint is zero everywhere, constraint(r) = 0 w.h.p.

        4. For each round i = 0 to n-1:

           a. Compute round polynomial (degree 2)
              - g_i(X) = Σ_{x_{i+1},...,x_n} constraint(...) * eq(...)
              - Send evaluations: g_i(0), g_i(1), g_i(2)

           b. Absorb into transcript
              - transcript.absorb(g_i(0), g_i(1), g_i(2))

           c. Derive challenge
              - c_i = transcript.squeeze()

           d. Fold polynomials
              - constraint' = constraint bound at x_i = c_i
              - eq' = eq bound at x_i = c_i
              - Size halves each round

        5. Final state
           - challenges = [c_0, c_1, ..., c_{n-1}]
           - final_constraint = constraint(challenges)
           - final_eq = eq(challenges, r)

OUTPUT: Sumcheck rounds (for verifier to check)
        Challenge point (where PCS must open)
        Claim: constraint(challenges) * eq(challenges, r) = 0
```

#### 2b. Lookup Argument (if enabled)

```
GOAL:   Prove all query values exist in table
        queries = [q_0, q_1, ..., q_m]
        table = [t_0, t_1, ..., t_k]

PROCESS (Logup):

        1. Derive random challenge
           - α = transcript.squeeze()

        2. Build fractional polynomials
           - For queries: f(x) = Σ 1/(q_i - α)
           - For table:   g(x) = Σ m_i/(t_i - α)  where m_i = multiplicity

        3. Run sumcheck to prove Σ f(x) = Σ g(x)
           - If queries ⊆ table, sums match
           - If any query not in table, sums differ w.h.p.

        4. Absorb rounds, derive challenges (same pattern as zerocheck)

OUTPUT: Lookup sumcheck rounds
        Challenge point for opening lookup polynomials
```

#### 2c. GKR (if enabled, for layered circuits)

```
GOAL:   Prove correct evaluation of layered circuit
        Layer 0 = output, Layer n = input

PROCESS:

        1. Start with claim about output layer
           - "output(r_0) = v"

        2. For each layer i = 0 to n-1:

           a. Run sumcheck to reduce claim
              - Claim about layer i → claim about layer i+1
              - Uses gate structure (add/mul gates)

           b. Absorb rounds, derive challenges

        3. Final claim is about input layer
           - "input(r_n) = v'"
           - This needs PCS opening

OUTPUT: Per-layer sumcheck rounds
        Challenge point for opening input polynomial
```

### Phase 3: Open

```
GOAL:   Prove committed polynomials evaluate to claimed values at challenge points

INPUT:  From Phase 2:
        - (constraint_poly, challenge_point, claimed_value = 0)
        - (lookup_poly, lookup_challenge, lookup_value)  // if applicable
        - (input_poly, gkr_challenge, gkr_value)         // if applicable

PROCESS (Basefold):

        For each (polynomial, point, value) tuple:

        1. Build eq polynomial for the point
           - eq(x, point) as before

        2. Run sumcheck on (polynomial, eq)
           - Proves: Σ polynomial(x) * eq(x, point) = value
           - This equals polynomial(point) = value

        3. Commit to each folding layer
           - As polynomial folds, commit intermediate states
           - Creates a "commitment chain"

        4. Generate query proofs
           - Random indices from transcript
           - Merkle openings at those indices
           - Proves folding was done correctly

OUTPUT: Opening proofs (sumcheck rounds + Merkle proofs)
```

## Complete Prover Flow

```
prove(trace) -> Proof
═══════════════════════════════════════════════════════════════════

    ctx = ProverContext.init()

    ┌─────────────────────────────────────────────────────────────┐
    │ PHASE 1: COMMIT                                             │
    └─────────────────────────────────────────────────────────────┘

    // Build polynomials
    trace_polys = trace.toPolynomials()
    constraint_poly = buildConstraints(trace_polys)

    // Commit
    trace_commitment = Merkle.commit(trace_polys)
    ctx.transcript.absorb(trace_commitment)

    // Lookup helpers (if enabled)
    if (config.lookup) {
        lookup_polys = buildLookupPolys(trace)
        lookup_commitment = Merkle.commit(lookup_polys)
        ctx.transcript.absorb(lookup_commitment)
    }

    ┌─────────────────────────────────────────────────────────────┐
    │ PHASE 2: IOP                                                │
    └─────────────────────────────────────────────────────────────┘

    // Zerocheck on constraints
    zerocheck_result = Zerocheck.prove(ctx, constraint_poly)
    // Claim: constraint_poly(zerocheck_result.challenges) = 0

    // Lookup (if enabled)
    if (config.lookup) {
        lookup_result = Lookup.prove(ctx, lookup_polys)
    }

    // GKR (if enabled)
    if (config.gkr) {
        gkr_result = GKR.prove(ctx, circuit, inputs)
    }

    ┌─────────────────────────────────────────────────────────────┐
    │ PHASE 3: OPEN                                               │
    └─────────────────────────────────────────────────────────────┘

    // Open at zerocheck challenge point
    constraint_opening = PCS.open(
        ctx,
        constraint_poly,
        zerocheck_result.challenges,
        claimed_value = 0
    )

    // Open lookup polys (if applicable)
    if (config.lookup) {
        lookup_opening = PCS.open(ctx, lookup_polys, lookup_result.challenges, ...)
    }

    // Open GKR input (if applicable)
    if (config.gkr) {
        gkr_opening = PCS.open(ctx, input_poly, gkr_result.challenges, ...)
    }

    ┌─────────────────────────────────────────────────────────────┐
    │ RETURN PROOF                                                │
    └─────────────────────────────────────────────────────────────┘

    return Proof {
        // Commitments
        .trace_commitment = trace_commitment,
        .lookup_commitment = lookup_commitment,  // optional

        // IOP transcripts
        .zerocheck = zerocheck_result,
        .lookup = lookup_result,                 // optional
        .gkr = gkr_result,                       // optional

        // PCS openings
        .constraint_opening = constraint_opening,
        .lookup_opening = lookup_opening,        // optional
        .gkr_opening = gkr_opening,              // optional
    }
```

## Complete Verifier Flow

```
verify(proof) -> bool
═══════════════════════════════════════════════════════════════════

    ctx = VerifierContext.init()

    ┌─────────────────────────────────────────────────────────────┐
    │ REPLAY PHASE 1: Absorb commitments                          │
    └─────────────────────────────────────────────────────────────┘

    ctx.transcript.absorb(proof.trace_commitment)

    if (config.lookup) {
        ctx.transcript.absorb(proof.lookup_commitment)
    }

    // Transcript state now matches prover's state after Phase 1

    ┌─────────────────────────────────────────────────────────────┐
    │ REPLAY PHASE 2: Verify IOPs                                 │
    └─────────────────────────────────────────────────────────────┘

    // Verify zerocheck
    zerocheck_ok = Zerocheck.verify(
        ctx,
        proof.zerocheck.rounds,
        claimed_sum = 0
    )
    if (!zerocheck_ok) return false

    // Challenges derived during verify() match prover's challenges
    // Because transcript state is identical

    // Verify lookup (if applicable)
    if (config.lookup) {
        lookup_ok = Lookup.verify(ctx, proof.lookup)
        if (!lookup_ok) return false
    }

    // Verify GKR (if applicable)
    if (config.gkr) {
        gkr_ok = GKR.verify(ctx, proof.gkr)
        if (!gkr_ok) return false
    }

    ┌─────────────────────────────────────────────────────────────┐
    │ REPLAY PHASE 3: Verify PCS openings                         │
    └─────────────────────────────────────────────────────────────┘

    // Verify constraint opening
    constraint_ok = PCS.verify(
        ctx,
        proof.trace_commitment,
        zerocheck_challenges,      // derived during zerocheck.verify()
        claimed_value = 0,
        proof.constraint_opening
    )
    if (!constraint_ok) return false

    // Verify lookup opening (if applicable)
    if (config.lookup) {
        lookup_open_ok = PCS.verify(ctx, proof.lookup_commitment, ...)
        if (!lookup_open_ok) return false
    }

    // Verify GKR opening (if applicable)
    if (config.gkr) {
        gkr_open_ok = PCS.verify(ctx, ...)
        if (!gkr_open_ok) return false
    }

    return true
```

## Transcript Synchronization

The prover and verifier must have identical transcript state at every squeeze point:

```
Prover                              Verifier
────────────────────────────────────────────────────────────────────
transcript.init()                   transcript.init()
        │                                   │
        ▼                                   ▼
absorb(commitment)                  absorb(proof.commitment)
        │                                   │
        ▼                                   ▼
r = squeeze() ─────────────────────── r = squeeze()  // MUST match
        │                                   │
        ▼                                   ▼
absorb(round_0)                     absorb(proof.round_0)
        │                                   │
        ▼                                   ▼
c_0 = squeeze() ───────────────────── c_0 = squeeze()  // MUST match
        │                                   │
       ...                                 ...
```

If any absorption differs or happens in wrong order, challenges diverge and verification fails.

## Memory Flow (DOD Perspective)

```
ProverContext
│
├── transcript: Transcript
│   └── Absorptions accumulate hash state
│       Squeezes derive challenges deterministically
│
├── scratch: []F [size = 2^n]
│   └── Round 0: [f_0, f_1, ..., f_{2^n - 1}]
│       Round 1: [f'_0, ..., f'_{2^{n-1} - 1}] (folded in-place, half size)
│       Round 2: [f''_0, ..., f''_{2^{n-2} - 1}] (folded again)
│       ...
│       Final:   [f_final] (single element)
│
├── scratch_aux: []F [size = 2^n]
│   └── Same pattern for eq polynomial (or second poly in product)
│
└── arena: ArenaAllocator
    └── Round polys, commitments, proofs accumulate here
        All freed at once when proof is complete
```

No per-round allocations. Two scratch buffers reused throughout. Proof data accumulates in arena.

## Soundness Intuition

Why does this work?

1. **Commitment binding**: Prover can't change polynomials after committing (Merkle root is collision-resistant)

2. **Fiat-Shamir**: Challenges depend on everything absorbed so far. Changing any committed value changes all subsequent challenges.

3. **Sumcheck soundness**: If claimed sum is wrong, prover must "lie" in some round. But the random challenge catches lies with high probability.

4. **PCS soundness**: Opening proof demonstrates the committed polynomial actually evaluates to the claimed value. Can't fake without breaking Merkle/hash.

5. **Composition**: Each IOP produces a claim. PCS verifies the claim against the commitment. If any step fails, proof is rejected.

The prover can only succeed if:
- Constraints are actually satisfied (zerocheck passes)
- Lookups are valid (lookup argument passes)
- All openings match commitments (PCS passes)

Which means: the trace is valid.
