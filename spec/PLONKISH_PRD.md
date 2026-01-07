# PRD: Minimal Plonkish Constraint System

## Summary

A constraint system frontend that produces polynomial evaluations for the protocol layer. Constraints are sums of terms, terms are products of cells, and cells are `(column, row_offset)` references. Supports state-machine semantics via row rotations.

**Status:** Core implementation complete (Phases 1-4). Protocol integration pending (Phase 5).

---

## Problem Statement

libzero needs a way to express computational statements as polynomial constraints that can be proven via sumcheck + PCS. The constraint system must:

1. Support arbitrary polynomial gates (not just R1CS)
2. Enable state-machine patterns via row rotations (like AIR)
3. Integrate cleanly with existing sumcheck and commitment infrastructure
4. Remain minimal—complexity deferred to future extensions

---

## Design

### Architecture

```
src/constraint/
├── constraint.zig   # Cell, Term, Constraint, ConstraintSet, evaluate functions
├── trace.zig        # Column storage with set-tracking
└── builder.zig      # CircuitBuilder orchestrator for common gates
```

**Component relationships:**
```
CircuitBuilder (orchestrator)
├── Trace (witness storage)        — no knowledge of constraints
└── ConstraintSet (constraint storage) — no knowledge of trace
         ↓
    evaluate(constraints, trace) → []F → Protocol.prove()
```

`Trace` and `ConstraintSet` are fully decoupled. `CircuitBuilder` coordinates them and handles column validation.

### Core Types

| Type | Description |
|------|-------------|
| `Cell` | `{ col: usize, rot: i32 }` — reference to a cell at row offset |
| `Term` | Tagged union: `constant: F` or `product: { coeff: F, cells: []Cell }` |
| `Constraint` | `{ terms: []Term }` — sum of terms that must equal zero |
| `ConstraintSet` | Simple storage for constraints (no validation, decoupled from Trace) |
| `Trace` | Column-major storage with power-of-2 rows (decoupled from constraints) |
| `CircuitBuilder` | Orchestrator owning Trace + ConstraintSet, handles validation |

### Design Decisions (Locked)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Field type | Generic `comptime F: type` | Modules parameterized by field |
| Rotation type | `i32` | Match Plonky3, no artificial limits |
| Term representation | Tagged union `{ product, constant }` | Explicit constant handling |
| Public inputs | Embedded in trace (marked columns) | Simpler for MVP |
| Error handling | Return errors, no panics | Consistent, testable |
| Cyclic traces | Not supported | Keep simple, extendable later |
| Padding | Explicit only | User must pad to power-of-2 |
| Memory management | Caller-provided allocator via `ArrayListUnmanaged` | Arena-friendly |
| Eval loop order | Constraint-first | Simpler, benchmark later |
| Column bounds | Check in CircuitBuilder | Fail fast; keeps ConstraintSet decoupled from Trace |

---

## Implementation Status

### Phase 1: Core Types ✅
- [x] `Cell` struct with `col: usize`, `rot: i32`
- [x] `Term` tagged union with constant and product variants
- [x] `Term.evaluate()` evaluates against trace at row
- [x] `Constraint` with `validRowRange()` and `evaluate()`
- [x] `ConstraintSet` as simple constraint storage (decoupled from Trace)

**Location:** `src/constraint/constraint.zig:1-125`

### Phase 2: Trace ✅
- [x] `Trace` struct with column storage and `set_flags`
- [x] `addColumn()`, `set()`, `get()` with error handling
- [x] `markPublic()`, `getPublicValues()`
- [x] Power-of-2 validation
- [x] Rotation bounds checking

**Location:** `src/constraint/trace.zig`

### Phase 3: Evaluation ✅
- [x] `evaluate(constraints, trace, allocator)` → `![]F`
- [x] `evaluateDebug()` for per-constraint granularity
- [x] `isSatisfied(evals)` → `?usize`
- [x] Respects `validRowRange` for each constraint

**Location:** `src/constraint/constraint.zig:142-204`

### Phase 4: Builder API ✅
- [x] `CircuitBuilder` struct (orchestrates Trace + ConstraintSet)
- [x] `addWitness()`, `addPublic()`
- [x] `mulGate()`, `addGate()`, `subGate()`
- [x] `constGate()`, `assertZero()`
- [x] `conditionalMul()`, `transition()`, `conditionalTransition()`
- [x] `addConstraint()` for custom constraints
- [x] `build()` returns trace and constraints
- [x] Column validation in builder (not in ConstraintSet)
- [x] Arena allocator for term/cell memory management

**Location:** `src/constraint/builder.zig`

### Phase 5: Protocol Integration
- [ ] Protocol stub that checks constraint satisfaction
- [ ] Wire to existing sumcheck infrastructure
- [ ] Wire to PCS for polynomial commitment
- [ ] End-to-end test: build → evaluate → prove → verify

---

## Acceptance Criteria

### Phase 5: Protocol Integration

1. **Stub prover exists** that returns success iff all evaluations are zero
2. **End-to-end test passes** for:
   - Multiplication gate: `a * b = c`
   - State transition: `state[row+1] = state[row] + delta[row]`
   - Fibonacci: `fib[row] + fib[row+1] = fib[row+2]`
3. **Sumcheck integration** (when sumcheck module is ready):
   - Constraint evaluations are passed to sumcheck as multilinear polynomial
   - Sumcheck verifier receives evaluations
4. **PCS integration** (when PCS module is ready):
   - Trace columns are committed
   - Verifier receives commitments and opening proofs

### Bug Fix ✅

**Problem:** `CircuitBuilder.init()` passed `&trace` (stack-local) to `ConstraintSet.init()`. After return, the struct was moved but the stored pointer became dangling.

**Solution:** Removed trace pointer from `ConstraintSet` entirely. Now:
1. `ConstraintSet` is simple storage (no trace reference)
2. `CircuitBuilder` validates columns via `validateColumns()` helper
3. `Trace` and `ConstraintSet` are fully decoupled

This also fixed memory leaks by adding an `ArenaAllocator` to `CircuitBuilder` for term/cell allocations.

---

## Future Extensions (Out of Scope)

| Extension | Description |
|-----------|-------------|
| Lookups (LogUp) | Prove trace values exist in precomputed tables |
| Copy constraints | Permutation argument for wire equality |
| Chips | Multiple traces with cross-trace interactions |
| Preprocessed columns | Fixed columns known at setup time |
| Cyclic traces | Optional wraparound mode |
| Parallelization | Thread pool for evaluation |

These extend the column model without changing `Cell`/`Term`/`Constraint` types.

---

## Test Coverage

Current test coverage (all in respective `.zig` files):

| Module | Tests |
|--------|-------|
| `constraint.zig` | ~30 tests covering Cell, Term, Constraint, ConstraintSet, evaluate functions |
| `trace.zig` | 12 tests covering init, columns, set/get, rotation, public columns |
| `builder.zig` | 15 tests covering all gate types, chained operations, and column validation |

---

## References

- **Spec:** `spec/PLONKISH_CCS.md`
- **Implementation:** `src/constraint/`
- **Field:** `src/fields/mersenne31.zig` (used in tests)
