//! Protocol stub for constraint testing.
//!
//! This is a minimal stub that wraps constraint satisfaction checks.
//! Real protocol integration (commit -> sumcheck -> PCS.open) comes later.

const constraint = @import("constraint.zig");

/// Protocol stub parameterized by field type.
pub fn ProtocolStub(comptime F: type) type {
    const CS = constraint.ConstraintSystem(F);

    return struct {
        /// Stub prover - just checks constraint satisfaction.
        /// Real implementation will: commit -> sumcheck -> PCS.open
        pub fn prove(evals: []const F) bool {
            return CS.isSatisfied(evals) == null;
        }

        /// Stub verifier - placeholder.
        pub fn verify(_: []const F) bool {
            return true;
        }
    };
}

// ============ Tests ============ //

const std = @import("std");
const testing = std.testing;
const M31 = @import("../fields/mersenne31.zig").Mersenne31;
const CircuitBuilder = @import("builder.zig").CircuitBuilder(M31);
const TestCS = constraint.ConstraintSystem(M31);
const TestProtocol = ProtocolStub(M31);

test "e2e: multiplication circuit with protocol stub" {
    // 1. Build circuit: prove we know a, b such that a * b = 21
    var builder = try CircuitBuilder.init(testing.allocator, 4);
    defer builder.deinit();

    const a = try builder.addWitness();
    const b = try builder.addWitness();
    const c = try builder.addPublic(); // public output

    try builder.mulGate(a, b, c); // a * b = c

    // 2. Fill witness: 3 * 7 = 21
    try builder.set(a, 0, M31.fromU64(3));
    try builder.set(b, 0, M31.fromU64(7));
    try builder.set(c, 0, M31.fromU64(21));

    // Pad remaining rows
    for (1..4) |row| {
        try builder.set(a, row, M31.zero);
        try builder.set(b, row, M31.zero);
        try builder.set(c, row, M31.zero);
    }

    // 3. Evaluate constraints
    const result = builder.build();
    const evals = try TestCS.evaluate(result.constraints, result.trace, testing.allocator);
    defer testing.allocator.free(evals);

    // 4. Prove (stub just checks all evals are zero)
    try testing.expect(TestProtocol.prove(evals));

    // 5. Verify (stub always returns true for now)
    try testing.expect(TestProtocol.verify(evals));
}
