//! Protocol stub for constraint testing.
//!
//! This is a minimal stub that wraps constraint satisfaction checks.
//! Real protocol integration (commit -> sumcheck -> PCS.open) comes later.

const constraint_mod = @import("constraint.zig");

/// Protocol stub parameterized by field type.
pub fn ProtocolStub(comptime F: type) type {
    const CS = constraint_mod.ConstraintSystem(F);

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
