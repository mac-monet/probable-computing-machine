# FOLDING.md

## Purpose

Enable incremental verifiable computation (IVC) for libzero. IVC allows proving long computations step-by-step with constant memory, rather than materializing the entire execution trace.

## Motivation

Current architecture proves execution traces monolithically:
```
Execute full program → Generate trace → Prove all constraints at once
```

This requires O(n) memory for n-step computations. IVC enables:
- Streaming execution (prove as you go)
- Constant prover memory
- Recursive proof composition
- Proofs of unbounded computation

## Background

### Folding vs Accumulation

**Folding** (Nova-style): Combines two R1CS instances into one "relaxed" instance. Defers constraint checking via an error term.

**Accumulation**: Combines multiple membership claims into a single accumulator. More general primitive that implies folding.

Both achieve IVC. The distinction matters for implementation but not for the end goal.

### The Hash-Based Landscape

Group-based schemes (Nova, HyperNova) require elliptic curves for additive homomorphism. Our Basefold prover is hash-based (Merkle + Blake3), so we need hash-based accumulation.

**Arc** (Bünz, Chiesa, Fenzi, Nguyen, Wang 2024)
- Accumulates Reed-Solomon proximity claims
- Quasilinear prover: O(n log n) due to polynomial quotients
- Works in list-decoding regime (fewer queries)
- Drop-in for existing IOP-based SNARKs

**WARP** (Bünz, Chiesa, Fenzi, Wang 2025)
- Linear-time prover: O(n)
- Works with ANY linear code (not just RS)
- Uses multilinear extensions instead of univariate quotients
- Erasure correction for extraction (no efficient decoder needed)

### Core Mechanism

Both schemes accumulate **proximity claims** - statements of the form "function f is δ-close to code C with constraints."

Given two claims:
1. Random linear combination: `f = f₁ + r·f₂`
2. Out-of-domain sampling: force prover to "commit" to single nearby codeword
3. In-domain queries: verify consistency
4. Output: single proximity claim about combined function

Distance is preserved, enabling unbounded accumulation depth.

## Relation to Current Architecture

Current flow (`src/vm/prover.zig`):
```
TraceRow[] → TracePolynomials → Constraint evaluation → Basefold proof
```

With accumulation:
```
Step 1: witness → encode → accumulate with acc₀ → acc₁
Step 2: witness → encode → accumulate with acc₁ → acc₂
...
Step n: witness → encode → accumulate with accₙ₋₁ → accₙ
Final:  prove proximity of accₙ (single Basefold proof)
```

### What We Already Have

- Multilinear polynomial representation (`src/poly/multilinear.zig`)
- Sumcheck protocol (`src/iop/sumcheck.zig`)
- Merkle commitments (`src/merkle/`)
- Fiat-Shamir transcript (`src/core/transcript.zig`)
- Basefold PCS (`src/pcs/basefold.zig`)

### What We'd Need

1. **Accumulator type**: Holds proximity claim state between steps
2. **Step circuit**: Proves single VM transition (already have constraints)
3. **Accumulation operation**: Combines old accumulator + new step
4. **Multilinear extension evaluation**: For arbitrary points (not just hypercube)
5. **Final decider**: Proves accumulated claim via Basefold

## Key Considerations

### Arc vs WARP

| Factor | Arc | WARP |
|--------|-----|------|
| Prover complexity | O(n log n) | O(n) |
| Code requirement | Reed-Solomon | Any linear code |
| Implementation complexity | Moderate | Higher |
| Maturity | More established | Newer |

**Recommendation**: Start with Arc. It's closer to our current RS-based Basefold. WARP's linear-time benefit matters more for very long traces or if we want linear-time encodable codes.

### Constraint Representation

Arc/WARP accumulate PESAT (Polynomial Equation Satisfaction) - a generalization covering R1CS, CCS, etc. Our constraint system (`src/vm/constraints.zig`) uses selector polynomials:

```
constraint(x) = Σ (opcode == OP_i) · constraint_i(x)
```

This maps naturally to PESAT. The "bundling" technique combines multiple constraints via random linear combination.

### Out-of-Domain Sampling

**For Arc (RS codes)**: Evaluate polynomial at point outside evaluation domain L. Standard univariate interpolation.

**For WARP (any code)**: Use multilinear extension of codeword. Evaluate at random point in F^m. Requires:
- Computing multilinear extension of generator matrix
- Efficient evaluation algorithm

### Incremental vs Parallel

Accumulation is inherently sequential - each step depends on previous accumulator. However:
- Independent sub-computations can be proven in parallel, then accumulated
- Final accumulation is cheap relative to proving

For a stack VM, execution is already sequential, so this isn't a limitation.

### Memory Layout

Current trace stores full execution history. With IVC:
- Only need current step's witness
- Accumulator state (constant size)
- Final proof

This enables proving computations that don't fit in memory.

### Field Size Requirements

Both schemes require sufficiently large fields for soundness. With Mersenne-31:
- May need extension field for challenges
- Or: accept slightly higher soundness error
- Or: use multiple independent challenges

The WARP paper notes this is an open problem for small fields.

## Open Questions

1. **Which scheme first?** Arc is simpler but WARP is more future-proof.

2. **Code choice for WARP?** If we go WARP route:
   - Stick with RS (quasilinear encoding, optimal distance)
   - Switch to expander codes (linear encoding, worse distance)
   - RAA codes (linear encoding, practical)

3. **Step granularity?** Accumulate per-instruction or per-block? Smaller steps = more accumulations but less work per step.

4. **Recursive composition?** Do we need proofs-of-proofs? This affects whether we need the full PCD machinery or just IVC.

5. **Decider circuit?** The final proof proves the decider accepts. Decider complexity affects recursive proof size.

## References

- [Arc: Accumulation for Reed-Solomon Codes](https://eprint.iacr.org/2024/1731) - Bünz et al. 2024
- [Linear-Time Accumulation Schemes (WARP)](https://eprint.iacr.org/2025/753) - Bünz et al. 2025
- [Proof-Carrying Data from Accumulation Schemes](https://eprint.iacr.org/2020/499) - BCMS 2020
- [Nova: Recursive Zero-Knowledge Arguments from Folding Schemes](https://eprint.iacr.org/2021/370) - KST 2022

---

## Application: Distributed State and P2P Cash

This section explores using accumulation for distributed state systems where multiple independent parties prove state transitions client-side.

### IVC vs PCD

**IVC** (Incrementally Verifiable Computation) handles sequential computation by a single prover:
```
Prover → Prover → Prover → ...  (single chain)
```

**PCD** (Proof-Carrying Data) generalizes to distributed computation with multiple provers:
```
Prover_A ──┐
           ├──→ Prover_C → ...  (DAG structure)
Prover_B ──┘
```

PCD allows independent provers to create proofs that can be **merged** by a third party. Arc and WARP both construct full PCD, not just IVC.

### P2P Cash Model

Consider a UTXO-based system where each user proves their own state transitions:

```
Alice has UTXO_A (with proof of valid history)
Bob has UTXO_B (with proof of valid history)

Alice pays Carol:
  - Alice proves: "I own UTXO_A, here's valid spend to Carol"
  - Carol receives: UTXO_C with accumulated proof

Bob pays Carol:
  - Bob proves: "I own UTXO_B, here's valid spend to Carol"
  - Carol receives: UTXO_D with accumulated proof

Carol spends both to Dave:
  - Carol proves: "I own UTXO_C and UTXO_D, here's valid spend"
  - This MERGES two proof chains → requires PCD
  - Dave receives: UTXO_E with single accumulated proof
```

Key insight: **receiving from multiple sources requires merging proofs**, which is PCD.

### Accumulator as Proof-of-History

Each UTXO carries an accumulator proving its validity:
```
UTXO = {
    value: u64,
    owner: PublicKey,
    accumulator: Accumulator,  // proves all prior transactions valid
}
```

When spending:
```
new_accumulator = accumulate(
    old_accumulator,      // carry forward history
    spend_witness,        // prove this spend is valid
    transcript,
)
```

The accumulator grows by O(1) per transaction regardless of history length.

### What Each Party Proves

**Sender** (client-side proving):
1. Ownership: knows witness for UTXO
2. History: carries forward accumulator (valid lineage)
3. Transition: this spend satisfies rules (signature, amounts)

**Receiver** (verification):
1. Check accumulator is valid
2. Accept new UTXO with updated accumulator

No global state required - each UTXO is self-certifying.

### Design Considerations

**What gets accumulated?**
- Full transaction validity: accumulator proves entire UTXO lineage
- State transition only: proves "previous valid + this transition valid"

**When to run the decider?**

The accumulator is cheap to verify but the **decider** (full SNARK) provides final soundness:
- Every transaction: expensive, full security
- Periodically: batched security
- On "cash out" to L1: deferred security

**Merge strategy for multiple inputs:**
- Merge immediately: single accumulator, simpler wallet state
- Keep separate: multiple accumulators, spend individually
- Merge lazily: combine on next spend

**Data availability:**

Accumulator proves validity but receiver may want:
- Full transaction history (for disputes)
- Just current state (minimal storage)
- Merkle inclusion proofs (anchoring to external root)

### Multi-Input Accumulation

The accumulation schemes support multiple input accumulators:
```
Accumulate(acc₁, acc₂, ..., accₖ, new_witness) → acc_new
```

This is essential for:
- Spending multiple UTXOs in one transaction
- Receiving from multiple senders
- Merging sharded state

### Comparison to Existing Systems

**Mina**: Uses recursive SNARKs (Pickles) for constant-size blockchain proofs. Similar goal, different machinery (group-based vs hash-based).

**Zcash**: Uses SNARKs for privacy but not for state compression. Full nodes still validate all transactions.

**Rollups**: Batch many transactions into one proof, but prover is centralized. PCD enables decentralized proving.
