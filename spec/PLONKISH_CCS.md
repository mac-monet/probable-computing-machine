# Spec: Minimal Plonkish Constraint System

## Overview

Implement a minimal **Plonkish constraint system** that generates constraint polynomials for the protocol layer. This is purely a **frontend** — it produces traces and constraints, then hands off to `Protocol` for proving.

**Design principle:** Start minimal. The core is ~150 lines. Add sugar later.

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
├── trace.zig           # Column storage (replaces existing trace.zig)
├── constraint.zig      # Cell, Term, Constraint types + evaluation
└── builder.zig         # Ergonomic API for common gates

# Proving handled by existing composable layers:
src/pcs/pcs.zig         # Polynomial commitment scheme
src/sumcheck/           # Sumcheck protocol (exists)
```

**Data flow:**
```
Define constraints → Fill trace → evaluate() → []F → Protocol.prove()
```

---

## Design Decisions

These decisions were made during design review:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Field type | Import from `field.zig` | Use existing field implementation |
| Rotation type | `i32` | Match Plonky3, no artificial limits |
| Term representation | Tagged union `{ product, constant }` | Explicit constant handling |
| Public inputs | Embedded in trace (marked columns) | Simpler for MVP |
| Error handling | Return errors, no panics | Consistent, testable |
| Cyclic traces | Not supported | Keep simple, extendable later |
| Padding | Explicit only | User must pad to power-of-2 |
| Memory management | Arena allocator | All constraint data freed together |
| Eval loop order | Constraint-first | Simpler, benchmark later |
| Column bounds | Check at creation time | Fail fast |
| Zero coefficients | Keep as no-op | Simpler, optimize later |

---

## Phase 1: Core Data Structures

### 1.1 Cell and Term Types

```zig
// constraint.zig

const std = @import("std");
const Allocator = std.mem.Allocator;
const F = @import("../field.zig").Field;

/// A cell reference: column at row offset
pub const Cell = struct {
    col: usize,
    rot: i32 = 0,  // 0 = current row, 1 = next, -1 = prev
};

/// A term in a constraint polynomial
/// Either a constant value or a coefficient times a product of cells
pub const Term = union(enum) {
    /// Constant term (e.g., -5 in "a*b - 5 = 0")
    constant: F,

    /// Product term: coefficient × product of cells
    product: struct {
        coeff: F,
        cells: []const Cell,
    },

    /// Evaluate this term at a given row
    pub fn evaluate(self: Term, trace: *const Trace, row: usize) !F {
        return switch (self) {
            .constant => |c| c,
            .product => |p| {
                var result = p.coeff;
                for (p.cells) |cell| {
                    const val = try trace.get(cell.col, row, cell.rot);
                    result = result.mul(val);
                }
                return result;
            },
        };
    }
};
```

### 1.2 Constraint Type

```zig
/// A constraint: sum of terms that must equal zero
pub const Constraint = struct {
    terms: []const Term,

    /// Compute valid row range from rotations in terms
    /// A constraint with rot=1 is valid for rows [0, num_rows-1)
    /// A constraint with rot=-1 is valid for rows [1, num_rows)
    pub fn validRowRange(self: Constraint, num_rows: usize) !struct { start: usize, end: usize } {
        var min_rot: i32 = 0;
        var max_rot: i32 = 0;

        for (self.terms) |term| {
            switch (term) {
                .constant => {},
                .product => |p| {
                    for (p.cells) |cell| {
                        min_rot = @min(min_rot, cell.rot);
                        max_rot = @max(max_rot, cell.rot);
                    }
                },
            }
        }

        // If min_rot = -1, start = 1 (can't access row -1)
        // If max_rot = 2, end = num_rows - 2 (can't access beyond last row)
        const start: usize = if (min_rot < 0) @intCast(-min_rot) else 0;
        const end_offset: usize = @intCast(max_rot);

        // Check if constraint is compatible with trace size
        if (start >= num_rows or end_offset >= num_rows) {
            return error.InvalidRowRange;
        }

        const end: usize = num_rows - end_offset;
        if (start >= end) {
            return error.InvalidRowRange;
        }

        return .{ .start = start, .end = end };
    }

    /// Evaluate this constraint at a specific row
    pub fn evaluate(self: Constraint, trace: *const Trace, row: usize) !F {
        var sum = F.zero;
        for (self.terms) |term| {
            const val = try term.evaluate(trace, row);
            sum = sum.add(val);
        }
        return sum;
    }
};
```

### 1.3 ConstraintSet

```zig
pub const ConstraintSet = struct {
    constraints: std.ArrayList(Constraint),
    allocator: Allocator,
    trace: *const Trace,  // Reference for column bounds checking

    pub fn init(allocator: Allocator, trace: *const Trace) ConstraintSet {
        return .{
            .constraints = std.ArrayList(Constraint).init(allocator),
            .allocator = allocator,
            .trace = trace,
        };
    }

    pub fn deinit(self: *ConstraintSet) void {
        self.constraints.deinit();
    }

    /// Add a constraint, validating column references
    pub fn add(self: *ConstraintSet, terms: []const Term) !void {
        // Validate all column references
        for (terms) |term| {
            switch (term) {
                .constant => {},
                .product => |p| {
                    for (p.cells) |cell| {
                        if (cell.col >= self.trace.numColumns()) {
                            return error.InvalidColumn;
                        }
                    }
                },
            }
        }
        try self.constraints.append(.{ .terms = terms });
    }

    pub fn len(self: ConstraintSet) usize {
        return self.constraints.items.len;
    }
};
```

---

## Phase 2: Trace (Column Storage)

```zig
// trace.zig

const std = @import("std");
const Allocator = std.mem.Allocator;
const F = @import("../field.zig").Field;

pub const Trace = struct {
    allocator: Allocator,

    /// Number of rows (must be power of 2)
    num_rows: usize,

    /// Column data - each column is values[row]
    /// Column-major for cache-friendly polynomial ops
    columns: std.ArrayList([]F),

    /// Track which cells have been set (for error detection)
    set_flags: std.ArrayList([]bool),

    /// Which columns are public (for verification)
    public_columns: std.ArrayList(usize),

    pub fn init(allocator: Allocator, num_rows: usize) !Trace {
        // Validate power of 2
        if (num_rows == 0 or (num_rows & (num_rows - 1)) != 0) {
            return error.NotPowerOfTwo;
        }
        return .{
            .allocator = allocator,
            .num_rows = num_rows,
            .columns = std.ArrayList([]F).init(allocator),
            .set_flags = std.ArrayList([]bool).init(allocator),
            .public_columns = std.ArrayList(usize).init(allocator),
        };
    }

    pub fn deinit(self: *Trace) void {
        for (self.columns.items) |col| {
            self.allocator.free(col);
        }
        for (self.set_flags.items) |flags| {
            self.allocator.free(flags);
        }
        self.columns.deinit();
        self.set_flags.deinit();
        self.public_columns.deinit();
    }

    /// Add a column, returns column index
    pub fn addColumn(self: *Trace) !usize {
        const col = try self.allocator.alloc(F, self.num_rows);
        @memset(col, F.zero);

        const flags = try self.allocator.alloc(bool, self.num_rows);
        @memset(flags, false);

        try self.columns.append(col);
        try self.set_flags.append(flags);

        return self.columns.items.len - 1;
    }

    /// Number of columns
    pub fn numColumns(self: Trace) usize {
        return self.columns.items.len;
    }

    /// Mark a column as public
    pub fn markPublic(self: *Trace, col: usize) !void {
        if (col >= self.columns.items.len) {
            return error.InvalidColumn;
        }
        try self.public_columns.append(col);
    }

    /// Set a cell value
    pub fn set(self: *Trace, col: usize, row: usize, val: F) !void {
        if (col >= self.columns.items.len) {
            return error.InvalidColumn;
        }
        if (row >= self.num_rows) {
            return error.InvalidRow;
        }
        self.columns.items[col][row] = val;
        self.set_flags.items[col][row] = true;
    }

    /// Get a cell value with rotation
    /// Returns error if cell was never set or rotation is out of bounds
    pub fn get(self: *const Trace, col: usize, row: usize, rot: i32) !F {
        if (col >= self.columns.items.len) {
            return error.InvalidColumn;
        }

        const row_i64: i64 = @intCast(row);
        const actual_row_i64 = row_i64 + rot;

        if (actual_row_i64 < 0 or actual_row_i64 >= self.num_rows) {
            return error.RotationOutOfBounds;
        }

        const actual_row: usize = @intCast(actual_row_i64);

        if (!self.set_flags.items[col][actual_row]) {
            return error.CellNotSet;
        }

        return self.columns.items[col][actual_row];
    }

    /// Extract public column values (for verifier)
    pub fn getPublicValues(self: Trace, allocator: Allocator) ![][]F {
        var result = try allocator.alloc([]F, self.public_columns.items.len);
        for (self.public_columns.items, 0..) |col_idx, i| {
            result[i] = try allocator.dupe(F, self.columns.items[col_idx]);
        }
        return result;
    }
};
```

**Key points:**
- Column-major storage (good for commitments, polynomial ops)
- Power-of-2 rows required (for multilinear)
- Tracks which cells are set (returns `error.CellNotSet` for uninitialized)
- Rotation handled in `get()` with bounds checking (returns `error.RotationOutOfBounds`)

---

## Phase 3: Constraint Evaluation

```zig
// constraint.zig (continued)

/// Evaluate constraints over trace, returns polynomial evaluations
/// Each constraint is only evaluated on rows where all rotations are valid
/// Result[row] = sum of all constraint evaluations at that row
/// For valid witness: all values should be zero
pub fn evaluate(
    constraints: []const Constraint,
    trace: *const Trace,
    allocator: Allocator,
) ![]F {
    const num_rows = trace.num_rows;
    var result = try allocator.alloc(F, num_rows);
    @memset(result, F.zero);

    for (constraints) |constraint| {
        // Only evaluate on valid rows for this constraint
        const range = try constraint.validRowRange(num_rows);

        for (range.start..range.end) |row| {
            const val = try constraint.evaluate(trace, row);
            result[row] = result[row].add(val);
        }
    }

    return result;
}

/// Evaluate constraints with per-constraint granularity (debug mode)
/// Returns [constraint_idx][row] for debugging which constraint failed where
pub fn evaluateDebug(
    constraints: []const Constraint,
    trace: *const Trace,
    allocator: Allocator,
) ![][]F {
    const num_rows = trace.num_rows;
    var result = try allocator.alloc([]F, constraints.len);

    for (constraints, 0..) |constraint, ci| {
        result[ci] = try allocator.alloc(F, num_rows);
        @memset(result[ci], F.zero);

        const range = try constraint.validRowRange(num_rows);
        for (range.start..range.end) |row| {
            result[ci][row] = try constraint.evaluate(trace, row);
        }
    }

    return result;
}

/// Check if all constraint evaluations are zero
/// Returns null if satisfied, or the first non-zero row index
pub fn isSatisfied(evals: []const F) ?usize {
    for (evals, 0..) |v, i| {
        if (!v.isZero()) return i;
    }
    return null;
}
```

**Output:** `[]F` — multilinear polynomial evaluations on `{0,1}^n`, ready for sumcheck.

---

## Phase 4: Builder API

Ergonomic layer for common patterns. Not required — you can construct constraints directly.

```zig
// builder.zig

const std = @import("std");
const Allocator = std.mem.Allocator;
const F = @import("../field.zig").Field;
const constraint_mod = @import("constraint.zig");
const Trace = @import("trace.zig").Trace;
const Cell = constraint_mod.Cell;
const Term = constraint_mod.Term;
const Constraint = constraint_mod.Constraint;
const ConstraintSet = constraint_mod.ConstraintSet;

pub const CircuitBuilder = struct {
    allocator: Allocator,
    trace: Trace,
    constraints: ConstraintSet,

    pub fn init(allocator: Allocator, num_rows: usize) !CircuitBuilder {
        var trace = try Trace.init(allocator, num_rows);
        return .{
            .allocator = allocator,
            .trace = trace,
            .constraints = ConstraintSet.init(allocator, &trace),
        };
    }

    pub fn deinit(self: *CircuitBuilder) void {
        self.constraints.deinit();
        self.trace.deinit();
    }

    // === Column creation ===

    pub fn addWitness(self: *CircuitBuilder) !usize {
        return self.trace.addColumn();
    }

    pub fn addPublic(self: *CircuitBuilder) !usize {
        const col = try self.trace.addColumn();
        try self.trace.markPublic(col);
        return col;
    }

    // === Common gate patterns ===

    /// a + b = c
    pub fn addGate(self: *CircuitBuilder, a: usize, b: usize, c: usize) !void {
        const terms = try self.allocator.alloc(Term, 3);
        terms[0] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{.{ .col = a }}) } };
        terms[1] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{.{ .col = b }}) } };
        terms[2] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{.{ .col = c }}) } };
        try self.constraints.add(terms);
    }

    /// a - b = c
    pub fn subGate(self: *CircuitBuilder, a: usize, b: usize, c: usize) !void {
        const terms = try self.allocator.alloc(Term, 3);
        terms[0] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{.{ .col = a }}) } };
        terms[1] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{.{ .col = b }}) } };
        terms[2] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{.{ .col = c }}) } };
        try self.constraints.add(terms);
    }

    /// a * b = c
    pub fn mulGate(self: *CircuitBuilder, a: usize, b: usize, c: usize) !void {
        const terms = try self.allocator.alloc(Term, 2);
        terms[0] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{ .{ .col = a }, .{ .col = b } }) } };
        terms[1] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{.{ .col = c }}) } };
        try self.constraints.add(terms);
    }

    /// a = constant
    pub fn constGate(self: *CircuitBuilder, a: usize, constant: F) !void {
        const terms = try self.allocator.alloc(Term, 2);
        terms[0] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{.{ .col = a }}) } };
        terms[1] = .{ .constant = constant.neg() };
        try self.constraints.add(terms);
    }

    /// a = 0
    pub fn assertZero(self: *CircuitBuilder, a: usize) !void {
        const terms = try self.allocator.alloc(Term, 1);
        terms[0] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{.{ .col = a }}) } };
        try self.constraints.add(terms);
    }

    /// selector * (a * b - c) = 0  (conditional multiplication)
    pub fn conditionalMul(self: *CircuitBuilder, sel: usize, a: usize, b: usize, c: usize) !void {
        const terms = try self.allocator.alloc(Term, 2);
        terms[0] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{ .{ .col = sel }, .{ .col = a }, .{ .col = b } }) } };
        terms[1] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{ .{ .col = sel }, .{ .col = c } }) } };
        try self.constraints.add(terms);
    }

    /// col[row+1] = col[row] + delta[row]  (state transition)
    pub fn transition(self: *CircuitBuilder, col: usize, delta: usize) !void {
        const terms = try self.allocator.alloc(Term, 3);
        terms[0] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{.{ .col = col, .rot = 1 }}) } };
        terms[1] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{.{ .col = col, .rot = 0 }}) } };
        terms[2] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{.{ .col = delta, .rot = 0 }}) } };
        try self.constraints.add(terms);
    }

    /// selector * (col[row+1] - col[row] - delta[row]) = 0  (conditional transition)
    pub fn conditionalTransition(self: *CircuitBuilder, sel: usize, col: usize, delta: usize) !void {
        const terms = try self.allocator.alloc(Term, 3);
        terms[0] = .{ .product = .{ .coeff = F.one, .cells = try self.cellSlice(&.{ .{ .col = sel }, .{ .col = col, .rot = 1 } }) } };
        terms[1] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{ .{ .col = sel }, .{ .col = col, .rot = 0 } }) } };
        terms[2] = .{ .product = .{ .coeff = F.one.neg(), .cells = try self.cellSlice(&.{ .{ .col = sel }, .{ .col = delta, .rot = 0 } }) } };
        try self.constraints.add(terms);
    }

    // === Custom constraints ===

    /// Add arbitrary constraint
    pub fn addConstraint(self: *CircuitBuilder, terms: []const Term) !void {
        try self.constraints.add(terms);
    }

    // === Cell access ===

    pub fn set(self: *CircuitBuilder, col: usize, row: usize, val: F) !void {
        try self.trace.set(col, row, val);
    }

    // === Build ===

    pub fn build(self: *CircuitBuilder) struct { trace: *Trace, constraints: []const Constraint } {
        return .{
            .trace = &self.trace,
            .constraints = self.constraints.constraints.items,
        };
    }

    // === Internal helpers ===

    fn cellSlice(self: *CircuitBuilder, cells: []const Cell) ![]const Cell {
        return self.allocator.dupe(Cell, cells);
    }
};
```

---

## Phase 5: Protocol Stub

Minimal stub for testing constraint satisfaction. Real protocol integration comes later.

```zig
// protocol_stub.zig

const F = @import("../field.zig").Field;
const constraint_mod = @import("constraint.zig");

/// Stub prover - just checks constraint satisfaction
/// Real implementation will: commit → sumcheck → PCS.open
pub fn prove(evals: []const F) bool {
    return constraint_mod.isSatisfied(evals) == null;
}

/// Stub verifier - placeholder
pub fn verify(_: []const F) bool {
    return true;
}
```

---

## Phase 6: Testing

### 6.1 Multiplication Constraint

```zig
test "multiplication constraint" {
    const allocator = std.testing.allocator;

    var trace = try Trace.init(allocator, 4);
    defer trace.deinit();

    const a = try trace.addColumn();
    const b = try trace.addColumn();
    const c = try trace.addColumn();

    // 3 * 7 = 21
    try trace.set(a, 0, F.from(3));
    try trace.set(b, 0, F.from(7));
    try trace.set(c, 0, F.from(21));

    // Pad remaining rows
    for (1..4) |row| {
        try trace.set(a, row, F.zero);
        try trace.set(b, row, F.zero);
        try trace.set(c, row, F.zero);
    }

    // Constraint: a * b - c = 0
    const terms = &[_]Term{
        .{ .product = .{ .coeff = F.one, .cells = &.{ .{ .col = a }, .{ .col = b } } } },
        .{ .product = .{ .coeff = F.one.neg(), .cells = &.{.{ .col = c }} } },
    };
    const constraints = &[_]Constraint{.{ .terms = terms }};

    const evals = try evaluate(constraints, &trace, allocator);
    defer allocator.free(evals);

    try std.testing.expect(isSatisfied(evals) == null);
}

test "invalid witness produces non-zero" {
    const allocator = std.testing.allocator;

    var trace = try Trace.init(allocator, 4);
    defer trace.deinit();

    const a = try trace.addColumn();
    const b = try trace.addColumn();
    const c = try trace.addColumn();

    // 3 * 7 = 20 (WRONG)
    try trace.set(a, 0, F.from(3));
    try trace.set(b, 0, F.from(7));
    try trace.set(c, 0, F.from(20));

    for (1..4) |row| {
        try trace.set(a, row, F.zero);
        try trace.set(b, row, F.zero);
        try trace.set(c, row, F.zero);
    }

    const terms = &[_]Term{
        .{ .product = .{ .coeff = F.one, .cells = &.{ .{ .col = a }, .{ .col = b } } } },
        .{ .product = .{ .coeff = F.one.neg(), .cells = &.{.{ .col = c }} } },
    };
    const constraints = &[_]Constraint{.{ .terms = terms }};

    const evals = try evaluate(constraints, &trace, allocator);
    defer allocator.free(evals);

    // Should fail at row 0
    try std.testing.expect(isSatisfied(evals) == 0);
}
```

### 6.2 State Transition with Rotation

```zig
test "state transition with rotation" {
    const allocator = std.testing.allocator;

    var trace = try Trace.init(allocator, 4);
    defer trace.deinit();

    const state = try trace.addColumn();

    // state[i+1] = state[i] + 1 (counting)
    try trace.set(state, 0, F.from(0));
    try trace.set(state, 1, F.from(1));
    try trace.set(state, 2, F.from(2));
    try trace.set(state, 3, F.from(3));

    // Constraint: state[row+1] - state[row] - 1 = 0
    const terms = &[_]Term{
        .{ .product = .{ .coeff = F.one, .cells = &.{.{ .col = state, .rot = 1 }} } },
        .{ .product = .{ .coeff = F.one.neg(), .cells = &.{.{ .col = state, .rot = 0 }} } },
        .{ .constant = F.one.neg() },
    };
    const constraint = Constraint{ .terms = terms };

    // Valid range: rot=1 means rows [0, 3) = rows 0, 1, 2
    const range = try constraint.validRowRange(4);
    try std.testing.expect(range.start == 0);
    try std.testing.expect(range.end == 3);

    const evals = try evaluate(&.{constraint}, &trace, allocator);
    defer allocator.free(evals);

    try std.testing.expect(isSatisfied(evals) == null);
}
```

### 6.3 Fibonacci with Single Column (Rotation Pattern)

```zig
test "fibonacci via rotation - single column" {
    const allocator = std.testing.allocator;

    var trace = try Trace.init(allocator, 8);
    defer trace.deinit();

    const fib = try trace.addColumn();
    try trace.markPublic(fib);

    // fib[row] + fib[row+1] = fib[row+2]
    const terms = &[_]Term{
        .{ .product = .{ .coeff = F.one, .cells = &.{.{ .col = fib, .rot = 0 }} } },
        .{ .product = .{ .coeff = F.one, .cells = &.{.{ .col = fib, .rot = 1 }} } },
        .{ .product = .{ .coeff = F.one.neg(), .cells = &.{.{ .col = fib, .rot = 2 }} } },
    };
    const constraint = Constraint{ .terms = terms };

    // Valid range: rot=2 means rows [0, 6) = rows 0-5
    const range = try constraint.validRowRange(8);
    try std.testing.expect(range.start == 0);
    try std.testing.expect(range.end == 6);

    // Fill: 1, 1, 2, 3, 5, 8, 13, 21
    const fibs = [_]u32{ 1, 1, 2, 3, 5, 8, 13, 21 };
    for (fibs, 0..) |v, i| {
        try trace.set(fib, i, F.from(v));
    }

    const evals = try evaluate(&.{constraint}, &trace, allocator);
    defer allocator.free(evals);

    try std.testing.expect(isSatisfied(evals) == null);
}
```

### 6.4 Builder API Test

```zig
test "builder - chained arithmetic" {
    const allocator = std.testing.allocator;

    var builder = try CircuitBuilder.init(allocator, 4);
    defer builder.deinit();

    const a = try builder.addWitness();
    const b = try builder.addWitness();
    const t = try builder.addWitness();      // intermediate
    const c = try builder.addWitness();
    const out = try builder.addPublic();

    // t = a * b
    try builder.mulGate(a, b, t);
    // out = t + c
    try builder.addGate(t, c, out);

    // Witness: (3 * 7) + 5 = 26
    try builder.set(a, 0, F.from(3));
    try builder.set(b, 0, F.from(7));
    try builder.set(t, 0, F.from(21));
    try builder.set(c, 0, F.from(5));
    try builder.set(out, 0, F.from(26));

    // Pad remaining rows
    for (1..4) |row| {
        try builder.set(a, row, F.zero);
        try builder.set(b, row, F.zero);
        try builder.set(t, row, F.zero);
        try builder.set(c, row, F.zero);
        try builder.set(out, row, F.zero);
    }

    const result = builder.build();
    const evals = try evaluate(result.constraints, result.trace, allocator);
    defer allocator.free(evals);

    try std.testing.expect(isSatisfied(evals) == null);
}
```

### 6.5 Error Handling Tests

```zig
test "error on unset cell" {
    const allocator = std.testing.allocator;

    var trace = try Trace.init(allocator, 4);
    defer trace.deinit();

    _ = try trace.addColumn();
    // Don't set any values

    try std.testing.expectError(error.CellNotSet, trace.get(0, 0, 0));
}

test "error on rotation out of bounds" {
    const allocator = std.testing.allocator;

    var trace = try Trace.init(allocator, 4);
    defer trace.deinit();

    const col = try trace.addColumn();
    try trace.set(col, 0, F.one);

    // Try to access row -1 from row 0
    try std.testing.expectError(error.RotationOutOfBounds, trace.get(col, 0, -1));

    // Try to access row 4 from row 3
    try trace.set(col, 3, F.one);
    try std.testing.expectError(error.RotationOutOfBounds, trace.get(col, 3, 1));
}

test "error on invalid column" {
    const allocator = std.testing.allocator;

    var trace = try Trace.init(allocator, 4);
    defer trace.deinit();

    // No columns added, try to set
    try std.testing.expectError(error.InvalidColumn, trace.set(0, 0, F.one));
}
```

---

## Implementation Order

### Step 1: Core Types (~50 lines)
- [ ] `Cell` struct with col: usize, rot: i32
- [ ] `Term` tagged union with constant and product variants
- [ ] `Constraint` with `validRowRange()` and `evaluate()`
- [ ] Unit tests for type construction and range computation

### Step 2: Trace (~60 lines)
- [ ] `Trace` struct with column storage and set_flags
- [ ] `addColumn()`, `set()`, `get()` with error handling
- [ ] `markPublic()`, `getPublicValues()`
- [ ] Tests for all error conditions

### Step 3: Evaluation (~40 lines)
- [ ] `evaluate(constraints, trace)` → `![]F`
- [ ] `evaluateDebug()` for per-constraint granularity
- [ ] `isSatisfied(evals)` → `?usize`
- [ ] Tests: mul gate, add gate, invalid witness, rotation ranges

### Step 4: Builder (~80 lines)
- [ ] `CircuitBuilder` struct
- [ ] All gate methods: mul, add, sub, const, assertZero, conditionalMul, transition, conditionalTransition
- [ ] Tests: chained arithmetic, fibonacci

### Step 5: Integration
- [ ] Protocol stub
- [ ] Wire to existing sumcheck + PCS
- [ ] End-to-end test: build → evaluate → prove → verify

---

## Future Extensions (Post-MVP)

1. **Lookups (LogUp)**: Prove trace values exist in precomputed tables
2. **Copy constraints**: Permutation argument for wire equality
3. **Chips**: Multiple traces with cross-trace interactions
4. **Preprocessed columns**: Fixed columns known at setup time
5. **Parallelization**: Thread pool for evaluation
6. **Cyclic traces**: Optional wraparound mode

---

## Stability Note

**Columns are the stable abstraction.** Future extensions add to this model:

| Extension | Change |
|-----------|--------|
| Chips | Multiple `Trace` instances + interactions |
| Lookups | New `LookupTable` type alongside `Trace` |
| Cyclic | Add `cyclic: bool` field to Trace |
| Copy constraints | Add `Permutation` argument |

The `columns: [][]F` storage and `Cell`/`Term`/`Constraint` types will not change.
