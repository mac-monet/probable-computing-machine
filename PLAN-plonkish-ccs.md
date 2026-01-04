# Plan: Multilinear CCS-based Plonkish Frontend

## Overview

Implement a **Customizable Constraint System (CCS)** frontend that generates constraint polynomials for the composable protocol layer (see `TODO-ARCH.md`). This is purely a **frontend** — it produces traces and constraints, then hands off to `Protocol` for proving.

## What is CCS?

CCS (from HyperNova) generalizes R1CS to support arbitrary-degree constraints:

```
R1CS:  Az ∘ Bz = Cz                    (degree 2 only)
CCS:   Σᵢ cᵢ · ∏ⱼ∈Sᵢ (Mⱼ · z) = 0     (arbitrary degree)
```

Where:
- `z` = witness vector (trace columns flattened)
- `Mⱼ` = sparse matrices selecting/combining columns
- `Sᵢ` = multisets defining which matrices to multiply
- `cᵢ` = scalar coefficients

**Example - Arithmetic gates in CCS:**
```
ADD: a + b - c = 0
     M0·z + M1·z - M2·z = 0

MUL: a * b - c = 0
     (M0·z) ∘ (M1·z) - M2·z = 0
```

## Architecture

```
src/trace/
├── trace.zig           # Dynamic column trace storage
├── ccs.zig             # CCS constraint definition
├── builder.zig         # Circuit builder API
└── constraint.zig      # Constraint polynomial evaluation (trace + CCS → []F)

# Proving handled by existing composable layers:
src/protocol/protocol.zig   # Takes constraint polynomial, runs IOP + PCS
src/iop/sumcheck.zig        # Generic sumcheck
src/pcs/basefold.zig        # PCS implementation
```

**Data flow:**
```
CircuitBuilder → Trace + CCS → evaluateConstraints() → []F → Protocol.prove()
```

---

## Phase 1: Core Data Structures

### 1.1 Trace (Dynamic Columns)

```zig
pub const Trace = struct {
    /// Number of rows (padded to power of 2)
    num_rows: usize,

    /// Column data - each column is a multilinear polynomial
    /// evaluated on {0,1}^log2(num_rows)
    columns: std.ArrayList(Column),

    pub const Column = struct {
        name: []const u8,
        values: []F,          // length = num_rows
        kind: ColumnKind,
    };

    pub const ColumnKind = enum {
        witness,    // private inputs computed by prover
        public,     // public inputs/outputs
        selector,   // gate selectors (usually 0 or 1)
        fixed,      // preprocessed constants
    };
};
```

**Key operations:**
- `addColumn(name, kind)` → column index
- `setCell(row, col, value)`
- `getColumn(col)` → []F (multilinear evaluations)
- `pad()` → ensure power-of-2 rows

### 1.2 CCS Definition

```zig
pub const CCS = struct {
    /// Number of matrices
    num_matrices: usize,

    /// Each matrix is sparse: maps (row, col) → coefficient
    /// For simple column selection: M[i][i] = 1
    matrices: []SparseMatrix,

    /// Constraint terms: each term is (coefficient, multiset of matrix indices)
    terms: []Term,

    pub const Term = struct {
        coeff: F,
        matrix_indices: []usize,  // product of these matrices' results
    };

    pub const SparseMatrix = struct {
        entries: []Entry,

        pub const Entry = struct {
            row: usize,
            col: usize,
            val: F,
        };
    };
};
```

### 1.3 Gate Abstraction (High-Level API)

For ergonomic circuit building, wrap CCS in a gate-based API:

```zig
pub const GateType = enum {
    add,      // a + b = c
    mul,      // a * b = c
    const_,   // a = constant

    // Future extensions:
    // sub, div, pow, boolean, range, lookup, ...
};

pub const Gate = struct {
    type: GateType,
    inputs: []usize,   // column indices
    output: usize,     // column index
    constant: ?F,      // for const gates
};
```

---

## Phase 2: Circuit Builder

High-level API that generates trace + CCS:

```zig
pub const CircuitBuilder = struct {
    trace: Trace,
    gates: std.ArrayList(Gate),

    /// Add a witness column, returns column index
    pub fn witness(self: *Self, name: []const u8) usize;

    /// Add a public input/output column
    pub fn public(self: *Self, name: []const u8) usize;

    /// Place an addition gate: c = a + b
    pub fn add(self: *Self, a: usize, b: usize, c: usize) void;

    /// Place a multiplication gate: c = a * b
    pub fn mul(self: *Self, a: usize, b: usize, c: usize) void;

    /// Set a cell value (during witness generation)
    pub fn set(self: *Self, row: usize, col: usize, val: F) void;

    /// Finalize: generate selector columns, pad trace, build CCS
    pub fn build(self: *Self) struct { trace: Trace, ccs: CCS };
};
```

**Example usage - simple multiplication:**
```zig
var builder = CircuitBuilder.init(allocator);

// Define columns
const a = builder.witness("a");
const b = builder.witness("b");
const c = builder.public("c");  // output

// Define constraint
builder.mul(a, b, c);

// Fill witness (for row 0)
builder.set(0, a, F.fromU32(3));
builder.set(0, b, F.fromU32(7));
builder.set(0, c, F.fromU32(21));

// Build
const result = builder.build();
```

---

## Phase 3: Constraint Evaluation

Convert CCS + Trace into a constraint polynomial for sumcheck.

### 3.1 CCS to Multilinear Constraint

For each row, the CCS constraint is:
```
constraint(row) = Σᵢ cᵢ · ∏ⱼ∈Sᵢ (Mⱼ · z)[row]
```

Where `(Mⱼ · z)[row]` is the row-th entry of the matrix-vector product.

For simple column-selecting matrices (identity on one column):
```
(Mⱼ · z)[row] = trace.columns[j].values[row]
```

### 3.2 Constraint Polynomial

The constraint polynomial is a multilinear polynomial C(x) where:
- x ∈ {0,1}^n indexes rows (n = log2(num_rows))
- C(x) = constraint value at row index(x)

**For valid witness:** C(x) = 0 for all x ∈ {0,1}^n

### 3.3 Implementation

```zig
pub fn evaluateConstraints(ccs: CCS, trace: Trace) []F {
    const num_rows = trace.num_rows;
    var result = allocator.alloc(F, num_rows);

    for (0..num_rows) |row| {
        var sum = F.zero;
        for (ccs.terms) |term| {
            var product = term.coeff;
            for (term.matrix_indices) |mat_idx| {
                // For column-selecting matrices:
                const col_val = applyMatrix(ccs.matrices[mat_idx], trace, row);
                product = product.mul(col_val);
            }
            sum = sum.add(product);
        }
        result[row] = sum;
    }
    return result;  // This IS the multilinear polynomial evaluations
}
```

---

## Phase 4: Integration with Protocol Layer

The trace module produces a constraint polynomial. The protocol layer (from `TODO-ARCH.md`) handles proving.

### 4.1 Usage Pattern

```zig
const protocol = @import("protocol/protocol.zig");
const trace_mod = @import("trace/trace.zig");
const ccs_mod = @import("trace/ccs.zig");
const constraint = @import("trace/constraint.zig");

// 1. Build circuit and fill witness
var builder = CircuitBuilder.init(allocator);
// ... define columns, gates, fill values ...
const result = builder.build();

// 2. Evaluate constraint polynomial (this is what trace module provides)
const constraint_evals = constraint.evaluate(result.ccs, result.trace, allocator);

// 3. Extract public values
const public_values = result.trace.getPublicValues();

// 4. Hand off to protocol layer for proving
var ctx = try ProverContext.init(allocator);
const proof = try Protocol.prove(&ctx, constraint_evals);

// Verification uses same protocol layer
const valid = try Protocol.verify(&ctx, &proof);
```

### 4.2 What the Trace Module Provides

```zig
// trace/constraint.zig - the key integration point
pub fn evaluate(ccs: CCS, trace: Trace, allocator: Allocator) []F {
    // Returns multilinear polynomial evaluations on {0,1}^n
    // This []F goes directly to Protocol.prove()
}

// trace/trace.zig
pub fn getPublicValues(trace: Trace) []F {
    // Extract public inputs/outputs for verification
}
```

The trace module is **stateless** — it doesn't hold proving context, transcripts, or commitments. Those belong to the protocol layer.

---

## Phase 5: Testing

### 5.1 Basic Arithmetic Tests

```zig
test "multiplication gate - constraint evaluation" {
    var builder = CircuitBuilder.init(allocator);
    const a = builder.witness("a");
    const b = builder.witness("b");
    const c = builder.public("c");
    builder.mul(a, b, c);

    // 3 * 7 = 21 (valid witness)
    builder.set(0, a, F.fromU32(3));
    builder.set(0, b, F.fromU32(7));
    builder.set(0, c, F.fromU32(21));

    const result = builder.build();
    const constraint_evals = constraint.evaluate(result.ccs, result.trace, allocator);

    // Valid witness → all constraints evaluate to zero
    for (constraint_evals) |v| {
        try expect(v.isZero());
    }

    // Public output extraction
    const public_vals = result.trace.getPublicValues();
    try expect(public_vals[0].eq(F.fromU32(21)));
}

test "invalid witness produces non-zero constraints" {
    var builder = CircuitBuilder.init(allocator);
    const a = builder.witness("a");
    const b = builder.witness("b");
    const c = builder.public("c");
    builder.mul(a, b, c);

    // 3 * 7 = 20 (WRONG!)
    builder.set(0, a, F.fromU32(3));
    builder.set(0, b, F.fromU32(7));
    builder.set(0, c, F.fromU32(20));  // Should be 21

    const result = builder.build();
    const constraint_evals = constraint.evaluate(result.ccs, result.trace, allocator);

    // Invalid witness → at least one non-zero constraint
    var has_nonzero = false;
    for (constraint_evals) |v| {
        if (!v.isZero()) has_nonzero = true;
    }
    try expect(has_nonzero);
}

test "chained arithmetic" {
    // a * b = t1
    // t1 + c = output
    // Tests multiple gates, intermediate values
}
```

### 5.2 Fibonacci as Plonkish Circuit

```zig
test "fibonacci(7) = 13 - constraint satisfaction" {
    var builder = CircuitBuilder.init(allocator);

    // Columns for each fib value
    const fib = [_]usize{
        builder.public("fib0"),
        builder.witness("fib1"),
        builder.witness("fib2"),
        builder.witness("fib3"),
        builder.witness("fib4"),
        builder.witness("fib5"),
        builder.witness("fib6"),
        builder.public("fib7"),  // output
    };

    // Constraints: fib[i] + fib[i+1] = fib[i+2]
    for (0..6) |i| {
        builder.add(fib[i], fib[i+1], fib[i+2]);
    }

    // Witness: actual fibonacci values
    const values = [_]u32{1, 1, 2, 3, 5, 8, 13};
    for (values, 0..) |v, i| {
        builder.set(0, fib[i], F.fromU32(v));
    }
    builder.set(0, fib[7], F.fromU32(13));

    const result = builder.build();
    const constraint_evals = constraint.evaluate(result.ccs, result.trace, allocator);

    // All constraints satisfied
    for (constraint_evals) |v| {
        try expect(v.isZero());
    }

    // This []F can now be passed to Protocol.prove()
}
```

---

## Implementation Order

### Step 1: Trace Module
- [ ] `Trace` struct with dynamic columns
- [ ] Column types (witness, public, selector, fixed)
- [ ] Power-of-2 padding
- [ ] `getPublicValues()` extraction
- [ ] Basic tests

### Step 2: CCS Core
- [ ] `CCS` struct definition
- [ ] `SparseMatrix` for column selection (or simpler column-selector abstraction)
- [ ] Term evaluation
- [ ] Constraint polynomial generation

### Step 3: Builder API
- [ ] `CircuitBuilder` struct
- [ ] `witness()`, `public()` column creation
- [ ] `add()`, `mul()` gate placement
- [ ] `build()` → Trace + CCS
- [ ] Automatic selector column generation

### Step 4: Constraint Evaluation
- [ ] `constraint.evaluate(ccs, trace)` → `[]F`
- [ ] Verify output is multilinear polynomial on {0,1}^n
- [ ] Sanity check: all zeros for valid witness

### Step 5: Integration Test
- [ ] Wire to protocol layer (or mock it)
- [ ] Single gate tests (add, mul)
- [ ] Multi-gate circuits
- [ ] Fibonacci circuit
- [ ] Invalid witness produces non-zero constraints

---

## Future Extensions (Not in v1)

1. **Copy constraints**: Permutation argument for wire connections
2. **Lookup gates**: Plookup-style table lookups
3. **Custom gates**: User-defined constraint polynomials
4. **Range constraints**: Prove value in [0, 2^k)
5. **Boolean constraints**: Prove value is 0 or 1
6. **HyperNova folding**: IVC for incremental proofs

---

## Key Design Decisions

1. **CCS over PLONK gates**: More flexible, native to HyperNova, cleaner multilinear fit
2. **Dynamic columns**: Maximum flexibility for circuit structure
3. **No copy constraints v1**: Simplifies implementation, can add later
4. **Frontend only**: Trace module produces `[]F`, protocol layer handles proving
5. **Column-major storage**: Better cache locality for polynomial operations
6. **Builder pattern**: Ergonomic API that generates correct CCS automatically
7. **Composable with TODO-ARCH.md**: Fits into `Protocol(F, config)` architecture

---

## Questions / Decisions Needed

1. **Matrix representation**: Full sparse vs. specialized "column selector" type?
2. **Multi-row circuits**: One constraint per row, or batch multiple logical gates per row?
3. **Public input handling**: Separate vector or marked columns in trace?

Ready to implement when approved.
