# Plan: Minimal Plonkish Constraint System

## Overview

Implement a minimal **Plonkish constraint system** that generates constraint polynomials for the protocol layer. This is purely a **frontend** — it produces traces and constraints, then hands off to `Protocol` for proving.

**Design principle:** Start minimal. The core is ~100 lines. Add sugar later.

## What is Plonkish?

Plonkish constraints are polynomials over **cells**, where a cell is `(column, row_offset)`:

```
Constraint = Σ (coeff · ∏ cells)

Example - multiplication gate:
a[row] * b[row] - c[row] = 0

Example - state transition:
state[row+1] - state[row] - delta[row] = 0
```

The row offset (rotation) enables state machine semantics like AIR, while the polynomial structure supports arbitrary custom gates.

## Architecture

```
src/constraint/
├── trace.zig           # Column storage
├── constraint.zig      # Constraint types + evaluation
└── builder.zig         # Optional ergonomic API

# Proving handled by existing composable layers:
src/protocol/protocol.zig   # Takes constraint polynomial, runs IOP + PCS
```

**Data flow:**
```
Define constraints → Fill trace → evaluate() → []F → Protocol.prove()
```

---

## Phase 1: Core Data Structures

### 1.1 Trace (Column Storage)

```zig
pub const Trace = struct {
    allocator: Allocator,

    /// Number of rows (must be power of 2)
    num_rows: usize,

    /// Column data - each column is values[row]
    /// Column-major for cache-friendly polynomial ops
    columns: std.ArrayList([]F),

    /// Which columns are public (for verification)
    public_columns: std.ArrayList(usize),

    pub fn init(allocator: Allocator, num_rows: usize) Trace;
    pub fn deinit(self: *Trace) void;

    /// Add a column, returns column index
    pub fn addColumn(self: *Trace) usize;

    /// Mark a column as public
    pub fn markPublic(self: *Trace, col: usize) void;

    /// Set a cell value
    pub fn set(self: *Trace, col: usize, row: usize, val: F) void;

    /// Get a cell value with rotation (wraps around)
    pub fn get(self: Trace, col: usize, row: usize, rot: i8) F {
        const num_rows_i: i32 = @intCast(self.num_rows);
        const actual_row: usize = @intCast(@mod(@as(i32, @intCast(row)) + rot, num_rows_i));
        return self.columns.items[col][actual_row];
    }

    /// Extract public column values (for verifier)
    pub fn getPublicValues(self: Trace) [][]F;
};
```

**Key points:**
- Column-major storage (good for commitments, polynomial ops)
- Power-of-2 rows (required for multilinear)
- Rotation handled in `get()` with wraparound

### 1.2 Constraint Types

```zig
/// A cell reference: column at row offset
pub const Cell = struct {
    col: usize,
    rot: i8 = 0,  // 0 = current row, 1 = next, -1 = prev
};

/// A term: coefficient × product of cells
pub const Term = struct {
    coeff: F,
    cells: []const Cell,
};

/// A constraint: sum of terms (must equal zero)
pub const Constraint = []const Term;
```

**That's it.** This is the irreducible core of Plonkish:
- Cell = where to read a value
- Term = coefficient × product of values
- Constraint = sum of terms = 0

### 1.3 Constraint Set

```zig
pub const ConstraintSet = struct {
    constraints: std.ArrayList(Constraint),
    allocator: Allocator,

    pub fn init(allocator: Allocator) ConstraintSet;
    pub fn deinit(self: *ConstraintSet) void;

    /// Add a constraint
    pub fn add(self: *ConstraintSet, constraint: Constraint) void;

    /// Number of constraints
    pub fn len(self: ConstraintSet) usize;
};
```

---

## Phase 2: Constraint Evaluation

The core algorithm: evaluate all constraints at each row.

```zig
/// Evaluate constraints over trace, returns polynomial evaluations
/// Result[row] = sum of all constraint evaluations at that row
/// For valid witness: all values should be zero
pub fn evaluate(
    constraints: []const Constraint,
    trace: Trace,
    allocator: Allocator,
) ![]F {
    const num_rows = trace.num_rows;
    var result = try allocator.alloc(F, num_rows);
    @memset(result, F.zero);

    for (0..num_rows) |row| {
        for (constraints) |constraint| {
            // Evaluate this constraint at this row
            for (constraint) |term| {
                var prod = term.coeff;
                for (term.cells) |cell| {
                    const val = trace.get(cell.col, row, cell.rot);
                    prod = prod.mul(val);
                }
                result[row] = result[row].add(prod);
            }
        }
    }

    return result;
}

/// Check if all constraint evaluations are zero
pub fn isSatisfied(evals: []const F) bool {
    for (evals) |v| {
        if (!v.isZero()) return false;
    }
    return true;
}
```

**Output:** `[]F` — multilinear polynomial evaluations on `{0,1}^n`, ready for sumcheck.

---

## Phase 3: Builder API (Optional Sugar)

Ergonomic layer for common patterns. Not required — you can construct constraints directly.

```zig
pub const CircuitBuilder = struct {
    allocator: Allocator,
    trace: Trace,
    constraints: ConstraintSet,

    pub fn init(allocator: Allocator, num_rows: usize) CircuitBuilder;
    pub fn deinit(self: *CircuitBuilder) void;

    // === Column creation ===

    pub fn addWitness(self: *CircuitBuilder) usize {
        return self.trace.addColumn();
    }

    pub fn addPublic(self: *CircuitBuilder) usize {
        const col = self.trace.addColumn();
        self.trace.markPublic(col);
        return col;
    }

    // === Common gate patterns ===

    /// a + b = c
    pub fn addGate(self: *CircuitBuilder, a: usize, b: usize, c: usize) void {
        self.constraints.add(&.{
            .{ .coeff = F.one, .cells = &.{.{ .col = a }} },
            .{ .coeff = F.one, .cells = &.{.{ .col = b }} },
            .{ .coeff = F.neg_one, .cells = &.{.{ .col = c }} },
        });
    }

    /// a * b = c
    pub fn mulGate(self: *CircuitBuilder, a: usize, b: usize, c: usize) void {
        self.constraints.add(&.{
            .{ .coeff = F.one, .cells = &.{.{ .col = a }, .{ .col = b }} },
            .{ .coeff = F.neg_one, .cells = &.{.{ .col = c }} },
        });
    }

    /// a = constant
    pub fn constGate(self: *CircuitBuilder, a: usize, constant: F) void {
        self.constraints.add(&.{
            .{ .coeff = F.one, .cells = &.{.{ .col = a }} },
            .{ .coeff = constant.neg(), .cells = &.{} },  // empty cells = 1
        });
    }

    /// selector * (a * b - c) = 0  (conditional multiplication)
    pub fn conditionalMul(self: *CircuitBuilder, sel: usize, a: usize, b: usize, c: usize) void {
        self.constraints.add(&.{
            .{ .coeff = F.one, .cells = &.{.{ .col = sel }, .{ .col = a }, .{ .col = b }} },
            .{ .coeff = F.neg_one, .cells = &.{.{ .col = sel }, .{ .col = c }} },
        });
    }

    /// col[row+1] = col[row] + delta[row]  (state transition)
    pub fn transition(self: *CircuitBuilder, col: usize, delta: usize) void {
        self.constraints.add(&.{
            .{ .coeff = F.one, .cells = &.{.{ .col = col, .rot = 1 }} },      // next
            .{ .coeff = F.neg_one, .cells = &.{.{ .col = col, .rot = 0 }} },  // current
            .{ .coeff = F.neg_one, .cells = &.{.{ .col = delta, .rot = 0 }} },
        });
    }

    // === Custom constraints ===

    /// Add arbitrary constraint
    pub fn constraint(self: *CircuitBuilder, c: Constraint) void {
        self.constraints.add(c);
    }

    // === Cell access ===

    pub fn set(self: *CircuitBuilder, col: usize, row: usize, val: F) void {
        self.trace.set(col, row, val);
    }

    // === Build ===

    pub fn build(self: *CircuitBuilder) struct { trace: Trace, constraints: []const Constraint } {
        return .{
            .trace = self.trace,
            .constraints = self.constraints.constraints.items,
        };
    }
};
```

---

## Phase 4: Integration with Protocol Layer

```zig
const constraint_mod = @import("constraint/constraint.zig");
const protocol = @import("protocol/protocol.zig");

// 1. Build circuit
var builder = CircuitBuilder.init(allocator, 8);  // 8 rows
const a = builder.addWitness();
const b = builder.addWitness();
const c = builder.addPublic();
builder.mulGate(a, b, c);

// 2. Fill witness
builder.set(a, 0, F.from(3));
builder.set(b, 0, F.from(7));
builder.set(c, 0, F.from(21));
// ... pad remaining rows ...

// 3. Evaluate constraints
const result = builder.build();
const evals = try constraint_mod.evaluate(result.constraints, result.trace, allocator);

// 4. Verify satisfaction locally (optional sanity check)
std.debug.assert(constraint_mod.isSatisfied(evals));

// 5. Prove
const proof = try protocol.prove(evals);
```

---

## Phase 5: Testing

### 5.1 Core Constraint Tests

```zig
test "multiplication constraint" {
    var trace = Trace.init(allocator, 4);
    const a = trace.addColumn();
    const b = trace.addColumn();
    const c = trace.addColumn();

    // 3 * 7 = 21
    trace.set(a, 0, F.from(3));
    trace.set(b, 0, F.from(7));
    trace.set(c, 0, F.from(21));

    // Constraint: a * b - c = 0
    const constraints = &[_]Constraint{
        &.{
            .{ .coeff = F.one, .cells = &.{.{ .col = a }, .{ .col = b }} },
            .{ .coeff = F.neg_one, .cells = &.{.{ .col = c }} },
        },
    };

    const evals = try evaluate(constraints, trace, allocator);
    try expect(evals[0].isZero());  // satisfied at row 0
}

test "invalid witness produces non-zero" {
    // Same setup but c = 20 (wrong)
    trace.set(c, 0, F.from(20));

    const evals = try evaluate(constraints, trace, allocator);
    try expect(!evals[0].isZero());  // NOT satisfied
}
```

### 5.2 Rotation Test (State Transition)

```zig
test "state transition with rotation" {
    var trace = Trace.init(allocator, 4);
    const state = trace.addColumn();

    // state[i+1] = state[i] + 1 (counting)
    trace.set(state, 0, F.from(0));
    trace.set(state, 1, F.from(1));
    trace.set(state, 2, F.from(2));
    trace.set(state, 3, F.from(3));

    // Constraint: state[row+1] - state[row] - 1 = 0
    const constraints = &[_]Constraint{
        &.{
            .{ .coeff = F.one, .cells = &.{.{ .col = state, .rot = 1 }} },
            .{ .coeff = F.neg_one, .cells = &.{.{ .col = state, .rot = 0 }} },
            .{ .coeff = F.neg_one, .cells = &.{} },  // constant -1
        },
    };

    const evals = try evaluate(constraints, trace, allocator);

    // Rows 0, 1, 2 should satisfy (next - current = 1)
    try expect(evals[0].isZero());
    try expect(evals[1].isZero());
    try expect(evals[2].isZero());
    // Row 3 wraps: state[0] - state[3] - 1 = 0 - 3 - 1 = -4 ≠ 0
    try expect(!evals[3].isZero());
}
```

### 5.3 Fibonacci with Single Column

```zig
test "fibonacci via rotation - single column" {
    var trace = Trace.init(allocator, 8);
    const fib = trace.addColumn();
    trace.markPublic(fib);

    // fib[row] + fib[row+1] = fib[row+2]
    const constraints = &[_]Constraint{
        &.{
            .{ .coeff = F.one, .cells = &.{.{ .col = fib, .rot = 0 }} },
            .{ .coeff = F.one, .cells = &.{.{ .col = fib, .rot = 1 }} },
            .{ .coeff = F.neg_one, .cells = &.{.{ .col = fib, .rot = 2 }} },
        },
    };

    // Fill: 1, 1, 2, 3, 5, 8, 13, 21
    const fibs = [_]u32{ 1, 1, 2, 3, 5, 8, 13, 21 };
    for (fibs, 0..) |v, i| {
        trace.set(fib, i, F.from(v));
    }

    const evals = try evaluate(constraints, trace, allocator);

    // Rows 0-5 satisfy the constraint
    for (0..6) |i| {
        try expect(evals[i].isZero());
    }
    // Rows 6-7 wrap around and won't satisfy
}
```

### 5.4 Builder API Test

```zig
test "builder - chained arithmetic" {
    var builder = CircuitBuilder.init(allocator, 4);

    const a = builder.addWitness();
    const b = builder.addWitness();
    const t = builder.addWitness();      // intermediate
    const c = builder.addWitness();
    const out = builder.addPublic();

    // t = a * b
    builder.mulGate(a, b, t);
    // out = t + c
    builder.addGate(t, c, out);

    // Witness: (3 * 7) + 5 = 26
    builder.set(a, 0, F.from(3));
    builder.set(b, 0, F.from(7));
    builder.set(t, 0, F.from(21));
    builder.set(c, 0, F.from(5));
    builder.set(out, 0, F.from(26));

    const result = builder.build();
    const evals = try evaluate(result.constraints, result.trace, allocator);

    try expect(isSatisfied(evals));
}
```

---

## Implementation Order

### Step 1: Core Types (~30 lines)
- [ ] `Cell`, `Term`, `Constraint` type definitions
- [ ] Unit tests for type construction

### Step 2: Trace (~50 lines)
- [ ] `Trace` struct with column storage
- [ ] `addColumn()`, `set()`, `get()` with rotation
- [ ] `markPublic()`, `getPublicValues()`
- [ ] Tests for rotation wraparound

### Step 3: Evaluation (~30 lines)
- [ ] `evaluate(constraints, trace)` → `[]F`
- [ ] `isSatisfied(evals)` helper
- [ ] Tests: mul gate, add gate, invalid witness

### Step 4: Builder (~80 lines, optional)
- [ ] `CircuitBuilder` struct
- [ ] Sugar methods: `mulGate`, `addGate`, `constGate`
- [ ] `transition()` for state machine patterns
- [ ] Tests: chained arithmetic, fibonacci

### Step 5: Integration
- [ ] Wire to protocol layer
- [ ] End-to-end test: build → evaluate → prove → verify

---

## Phase 6: Lookups (Fast-Follow)

Lookups prove trace values exist in precomputed tables. Essential for range checks, bitwise ops, etc.

### Core Idea

```
Table: [0, 1, 2, ..., 255]
Lookup: "column a must contain values from this table"
→ Proves a is 8-bit without 8 boolean constraints
```

### Key Types

```zig
pub const LookupTable = struct {
    columns: []const []const F,  // table data
    num_rows: usize,
};

pub const Lookup = struct {
    table: *const LookupTable,
    queries: []const Cell,  // reuses Cell from constraints
};
```

### Protocol: LogUp

Use LogUp (logarithmic derivatives) - same as Plonky3/SP1:

```
Σ (multiplicity[t] / (α - t)) = Σ (1 / (α - query))
```

- Single random challenge `α` batches all lookups
- Naturally extends to chip interactions later
- Produces a polynomial for sumcheck (like constraints)

### Integration Pattern

```
Trace + Constraints → constraint_evals ──┐
                                         ├→ batched sumcheck
Trace + Lookups → lookup_evals ──────────┘
```

### Key Considerations

1. **Table sharing**: Multiple lookups can reference same table (e.g., one range table for all range checks)
2. **Multi-column lookups**: XOR table has 3 columns `(a, b, a^b)`, query must match all
3. **Multiplicities**: Must track how many times each table row is used
4. **Challenge ordering**: Lookup challenge comes after constraint commitment (Fiat-Shamir)

### Common Tables

- **Range**: `[0..2^n)` for n-bit range checks
- **XOR/AND/OR**: Bitwise ops as 3-column tables
- **Byte decomposition**: For word → bytes conversion

### Implementation Scope

- [ ] `LookupTable`, `Lookup` types
- [ ] `evaluateLookups()` → polynomial for sumcheck
- [ ] Builder sugar: `lookup()`, `rangeCheck()`
- [ ] Common table constructors

---

## Future Extensions (After Lookups)

1. **Copy constraints**: Permutation argument for wire equality
2. **Multiple constraint types**: Different constraints for different rows
3. **Chips**: Multiple traces with cross-trace interactions (uses LogUp)
4. **Preprocessed columns**: Fixed columns known at setup time
5. **Degree optimization**: Track max degree for sumcheck efficiency
6. **Vector lookups**: Multiple lookups batched efficiently

---

## Design Decisions

1. **Minimal core**: Cell + Term + Constraint is irreducible
2. **Rotation built-in**: Enables AIR-like state transitions
3. **Column-major storage**: Cache-friendly for polynomial ops
4. **Builder is optional**: Power users can construct constraints directly
5. **No matrices**: Direct cell references, not CCS matrix indirection
6. **Wraparound rotation**: Simplifies implementation, matches FFT domain

---

## Answered Questions

1. ~~Matrix representation~~ → **No matrices.** Direct cell references are simpler.
2. ~~Multi-row circuits~~ → **Rotation handles it.** Single constraint can span rows.
3. ~~Public input handling~~ → **Marked columns.** `trace.markPublic(col)` tracks which columns are public.

---

## Stability Note

**Columns are the stable abstraction.** Future extensions add to this model:

| Extension | Change |
|-----------|--------|
| Chips | Multiple `Trace` instances + interactions |
| Lookups | New `LookupTable` type alongside `Trace` |
| M3/Binius | Add `column_type: TowerLevel` field |
| Copy constraints | Add `Permutation` argument |

The `columns: [][]F` storage and `Cell`/`Term`/`Constraint` types will not change.

Ready to implement when approved.
