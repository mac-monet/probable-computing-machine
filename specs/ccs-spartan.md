# CCS Compilation and SPARTAN Sumcheck

## Overview

This spec describes adding CCS (Customizable Constraint Systems) as a compilation target for our Plonkish constraints, and implementing SPARTAN-style sumcheck to replace zerocheck for constraint satisfaction proving.

### Why This Change

Our current flow:
```
Constraints + Trace → evaluate() → dense []F → zerocheck
```

This materializes constraint evaluations at ALL rows before running zerocheck. For constraints with selector sparsity (many inactive rows), this is wasteful.

Proposed flow:
```
Constraints → compile() → CCS{sparse matrices}
CCS + Witness → spartan_sumcheck (evaluates lazily, exploits sparsity)
```

### When Sparsity Helps

- **Helps**: Many selectors, few active per row (VMs, conditional logic, lookups)
- **Doesn't help**: Uniform constraints applied to all rows equally

For lookup arguments specifically, multiplicities are inherently sparse (large tables, few entries used), making this a significant win.

## Architecture

### New Files

```
src/constraint/ccs.zig       -- CCS data structure and sparse matrix
src/constraint/compile.zig   -- Plonkish → CCS compilation
src/iop/spartan.zig          -- SPARTAN sumcheck (replaces zerocheck for CCS)
```

### Unchanged

- `src/constraint/constraint.zig` -- Plonkish constraint definitions stay as-is
- `src/constraint/builder.zig` -- CircuitBuilder API unchanged
- `src/iop/zerocheck.zig` -- Keep for comparison/fallback

## CCS Data Structure

### Conceptual Model

CCS represents constraints as:
```
Σ_j c_j · ∏_k (M_{j,k} · z) = 0
```

Where:
- `z` = flattened witness vector `[col_0_evals..., col_1_evals..., ..., 1]`
- `M_{j,k}` = sparse matrices that select/permute elements of z
- `c_j` = scalar coefficients
- The product is Hadamard (element-wise)

### Witness Vector Layout

Flatten trace columns into a single vector:
```
z[col * num_rows + row] = trace.get(col, row)
z[num_cols * num_rows] = 1  // constant term
```

Total size: `num_cols * num_rows + 1`

### Sparse Matrix Representation

Use CSR (Compressed Sparse Row) format:
- `row_ptr: []usize` -- row i has entries at indices `[row_ptr[i], row_ptr[i+1])`
- `col_indices: []usize` -- column index for each non-zero
- `values: []F` -- value for each non-zero

For Plonkish cell references, matrices are permutation matrices (exactly one entry per row), so:
- `nnz = num_rows` per matrix
- Very efficient evaluation: O(1) per row

### Handling Rotations

A cell reference `(col, rot)` becomes a matrix where:
```
row i → column (col * num_rows + (i + rot) mod num_rows)
```

This is a shifted identity matrix. Store the shift rather than materializing if beneficial.

### CCS Struct

```zig
pub const CCS = struct {
    matrices: []SparseMatrix,
    gates: []Gate,  // each gate = coeff + list of matrix indices to multiply
    num_vars: usize,  // log2(num_rows), for sumcheck
    witness_size: usize,  // |z|

    pub const Gate = struct {
        coeff: F,
        matrix_indices: []usize,
    };

    pub const SparseMatrix = struct {
        row_ptr: []usize,
        col_indices: []usize,
        values: []F,
        num_rows: usize,
        num_cols: usize,

        // Evaluate (M · z)[row]
        pub fn evalRow(self, z: []F, row: usize) F;

        // Batch evaluate for a range (optimization for sumcheck rounds)
        pub fn evalRows(self, z: []F, rows: Range) []F;
    };
};
```

## Compilation: Plonkish → CCS

### Input
- `[]Constraint` from the existing constraint system
- `num_cols`, `num_rows` from trace dimensions

### Output
- `CCS` struct with sparse matrices

### Algorithm

```
for each constraint in constraints:
    for each term in constraint.terms:
        if term is constant:
            handle separately (becomes a matrix selecting the "1" column of z)
        if term is product:
            for each cell in term.cells:
                create or reuse matrix for (col, rot) pair
            create gate with term.coeff and matrix indices
```

### Matrix Deduplication

Many terms reference the same `(col, rot)` pairs. Deduplicate matrices:
- Key: `(col, rot)` tuple
- Value: matrix index
- Reuse existing matrix if same key seen before

### Constant Terms

A constant `c` in a constraint becomes:
- Matrix that selects the last element of z (which is 1)
- Coefficient c
- Gate: `c · M_const · z = c · 1 = c`

## SPARTAN Sumcheck

### Relation to Zerocheck

Zerocheck proves: `Σ_x f(x) · eq(x, r) = 0` where f is the constraint polynomial.

SPARTAN proves the same thing, but computes f(x) lazily during sumcheck rounds rather than materializing f upfront.

### Protocol

1. Verifier sends random τ ∈ F^n (or derive via Fiat-Shamir)
2. Prover computes eq(x, τ) table
3. Run sumcheck on: `Σ_x eq(x, τ) · CCS_constraint(x)`
4. Each round:
   - Compute round polynomial (degree = max constraint degree + 1)
   - Absorb into transcript
   - Squeeze challenge
   - Fold eq table
5. Final claim: verifier has challenges r, needs to verify CCS_constraint(r)

### Sparse Round Computation

The key optimization. Instead of:
```
for x in 0..2^(n-round):
    sum += eq[x] * evaluate_all_constraints(x)  // dense
```

Do:
```
for x in 0..2^(n-round):
    sum += eq[x] * evaluate_ccs_sparse(ccs, z, x)  // sparse matrix ops
```

Where `evaluate_ccs_sparse` only touches non-zero matrix entries.

For very sparse selectors, can further optimize by iterating over non-zero eq·selector products, but the basic sparse matrix evaluation is the first win.

### Round Polynomial Degree

For CCS gate with d matrices multiplied together, the round polynomial has degree d+1 (product of d linear interpolations times linear eq interpolation).

Our Plonkish constraints have max degree ~3-4, so round polynomials need ~4-5 evaluations.

### Memory

- eq table: O(n) field elements, halves each round (reuse buffer)
- z vector: O(num_cols × num_rows) -- the witness
- CCS matrices: O(num_rows × num_matrices) for permutation matrices
- Round polynomials: O(degree) per round

### Proof Structure

```zig
pub const Proof = struct {
    tau: []F,  // random point for eq (derived from transcript, included for verification)
    rounds: []RoundPoly,  // one per variable
    final_evals: []F,  // M_k · z evaluated at challenge point, for each matrix
};
```

Verifier checks:
1. Sumcheck rounds are valid (round_poly(0) + round_poly(1) = prev_claim)
2. Final claim equals CCS evaluated at challenge point using final_evals

## Integration with Existing Code

### Prover Flow

```zig
// Build circuit (unchanged)
var builder = CircuitBuilder.init(allocator, num_rows);
const a = try builder.addWitness();
// ... add gates ...
const result = builder.build();

// NEW: Compile to CCS (once per circuit)
const ccs = try compile(result.constraints, trace.numColumns(), trace.num_rows);

// NEW: Flatten witness
const z = try flattenWitness(trace, allocator);

// NEW: SPARTAN prove (replaces zerocheck)
const proof = try SpartanSumcheck.prove(ccs, z, &transcript, allocator);
```

### Verifier Flow

```zig
// Verify SPARTAN sumcheck
const claim = try SpartanSumcheck.verify(ccs, &proof, &transcript, allocator);

// claim contains the evaluation point and expected value
// PCS must open the witness commitments at this point
```

### PCS Integration

The final claim from SPARTAN is about `z` evaluated at the challenge point. Since `z` is a flattening of trace columns, the PCS opening becomes:
- Open each column polynomial at the appropriate sub-point
- Or: commit to z directly as one multilinear polynomial

This may require adjusting how we commit to the trace. Current approach commits to columns separately; CCS naturally wants a single z commitment.

**Decision needed**: Keep per-column commitments and derive z openings, or switch to single z commitment?

Recommendation: Keep per-column for now, compute z(challenge) from column openings. This maintains compatibility with existing Basefold integration.

## Key Considerations

### When to Use CCS vs Zerocheck

| Scenario | Recommendation |
|----------|----------------|
| Uniform constraints (no selectors) | Zerocheck (simpler, same performance) |
| Many selectors, few active per row | SPARTAN (sparse wins) |
| Lookup arguments | SPARTAN (multiplicity sparsity) |
| Small circuits | Zerocheck (compilation overhead not worth it) |

Consider adding a heuristic or explicit flag to choose.

### Selector Representation

For Plonkish selectors that are 0/1:
- Could store as bitmap instead of full field elements
- Matrix becomes even sparser (only rows where selector=1)
- More complex but bigger win

Start with naive approach, optimize later if profiling shows benefit.

### Rotation Handling

Rotations that go out of bounds (negative rot at row 0, positive rot at last row) need handling:
- Current Plonkish: `validRowRange()` computes safe range
- CCS: Could use cyclic indexing (mod num_rows) or explicit boundary matrices

Recommendation: Match current behavior - constraints only apply to valid row ranges.

### Testing Strategy

1. **Unit tests**: CCS compilation produces correct matrices for simple gates
2. **Round-trip tests**: Plonkish eval == CCS eval for same witness
3. **Soundness tests**: SPARTAN rejects invalid witnesses
4. **Completeness tests**: SPARTAN accepts valid witnesses
5. **Comparison tests**: Zerocheck and SPARTAN produce equivalent proofs

### Performance Benchmarks

Compare against existing zerocheck:
- Uniform constraints (expect: similar)
- 10 selectors, 10% each active (expect: ~10x sparse win)
- 100 selectors, 1% each active (expect: ~100x sparse win)
- Lookup multiplicities (expect: significant win)

## Implementation Order

1. **CCS struct and sparse matrix** -- data structures only
2. **Compile function** -- Plonkish → CCS, with tests comparing evaluations
3. **SPARTAN sumcheck prover** -- basic version, not optimized
4. **SPARTAN sumcheck verifier** -- complete the protocol
5. **Integration tests** -- full prove/verify cycle
6. **Benchmarks** -- compare against zerocheck
7. **Optimizations** -- based on profiling (sparse selector iteration, etc.)

## Open Questions

1. **z commitment strategy**: Per-column vs single z polynomial?
2. **Boundary handling**: Cyclic or bounded rotations?
3. **Selector optimization**: Worth special-casing 0/1 selectors?
4. **Hybrid approach**: Auto-select CCS vs zerocheck based on circuit analysis?

## References

- [Customizable Constraint Systems (CCS)](https://eprint.iacr.org/2023/552) - Setty et al.
- [SPARTAN](https://eprint.iacr.org/2019/550) - Setty
- [Lasso/Jolt](https://eprint.iacr.org/2023/1217) - for sparse lookup techniques
