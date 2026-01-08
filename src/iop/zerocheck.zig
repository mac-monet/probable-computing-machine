const std = @import("std");
const Transcript = @import("../core/transcript.zig").Transcript;
const eq = @import("../poly/eq.zig");
const sumcheck = @import("sumcheck.zig");

/// Zerocheck IOP: proves f(x) = 0 for all x ∈ {0,1}ⁿ
///
/// Reduces to product sumcheck via the identity:
///   If f(x) = 0 ∀x ∈ {0,1}ⁿ, then Σ_x f(x)·eq(x,r) = 0 for random r
///
/// After reduction, verifier holds evaluation claims:
///   - f(r') = v_f  (needs PCS opening)
///   - eq(r', r) = v_eq  (verifier computes directly)
///   - v_f · v_eq = final_claim from sumcheck
pub fn Zerocheck(comptime F: type, comptime max_vars: comptime_int) type {
    const ProductSC = sumcheck.ProductSumcheck(F);

    return struct {
        const Self = @This();

        /// Zerocheck proof data
        pub const Proof = struct {
            /// Random point for eq polynomial (squeezed from transcript)
            r: [max_vars]F,
            /// Number of variables
            num_vars: usize,
            /// Sumcheck round polynomials
            rounds: [max_vars]ProductSC.RoundPoly,
            /// Final evaluation of f at the sumcheck point
            final_f: F,
            /// Final evaluation of eq at the sumcheck point
            final_eq: F,

            pub fn getR(self: *const Proof) []const F {
                return self.r[0..self.num_vars];
            }

            pub fn getRounds(self: *const Proof) []const ProductSC.RoundPoly {
                return self.rounds[0..self.num_vars];
            }
        };

        /// Evaluation claim produced by zerocheck
        pub const EvalClaim = struct {
            /// Point at which f must be evaluated
            point: []const F,
            /// Claimed value of f at that point
            value: F,
        };

        /// Prove that f(x) = 0 for all x ∈ {0,1}ⁿ
        ///
        /// Args:
        ///   allocator: For scratch space (eq table, working buffers)
        ///   f_evals: Evaluations of f on {0,1}ⁿ (will be copied, not modified)
        ///   transcript: Fiat-Shamir transcript
        ///
        /// Returns proof and the evaluation point for the final f claim
        pub fn prove(
            allocator: std.mem.Allocator,
            f_evals: []const F,
            transcript: *Transcript(F),
        ) !struct { proof: Proof, eval_point: []const F } {
            const n = f_evals.len;
            const num_vars = @ctz(n);
            std.debug.assert(n == @as(usize, 1) << @intCast(num_vars));
            std.debug.assert(num_vars <= max_vars);

            var proof: Proof = undefined;
            proof.num_vars = num_vars;

            // 1. Squeeze random point r for eq polynomial
            for (0..num_vars) |i| {
                proof.r[i] = transcript.squeeze();
            }

            // 2. Build eq(x, r) table
            const eq_evals = try allocator.alloc(F, n);
            eq.eqEvals(F, proof.r[0..num_vars], eq_evals);

            // 3. Copy f_evals to working buffer (sumcheck folds in place)
            const f_work = try allocator.alloc(F, n);
            @memcpy(f_work, f_evals);

            // 4. Run product sumcheck on f·eq with claim = 0
            //    We use the sumcheck module's prove function
            var work_polys = [_][]F{ f_work, eq_evals };
            var challenges: [max_vars]F = undefined;

            const final_evals = ProductSC.prove(
                &work_polys,
                transcript,
                proof.rounds[0..num_vars],
                challenges[0..num_vars],
            );

            proof.final_f = final_evals[0];
            proof.final_eq = final_evals[1];

            // The evaluation point is r composed with challenges
            // Actually, the sumcheck challenges ARE the evaluation point
            // (each round binds one variable)
            const eval_point = try allocator.alloc(F, num_vars);
            @memcpy(eval_point, challenges[0..num_vars]);

            return .{
                .proof = proof,
                .eval_point = eval_point,
            };
        }

        /// Verify a zerocheck proof
        ///
        /// Returns the evaluation claim that must be verified via PCS:
        ///   f(challenges) = proof.final_f
        ///
        /// The verifier checks:
        ///   1. Sumcheck rounds are valid with claimed_sum = 0
        ///   2. eq(challenges, r) matches proof.final_eq
        ///   3. final_f · final_eq = sumcheck's final_claim
        ///
        /// If all pass, returns the eval claim for PCS verification
        pub fn verify(
            allocator: std.mem.Allocator,
            proof: *const Proof,
            transcript: *Transcript(F),
        ) !?EvalClaim {
            const num_vars = proof.num_vars;

            // 1. Re-derive r from transcript (must match prover)
            var r: [max_vars]F = undefined;
            for (0..num_vars) |i| {
                r[i] = transcript.squeeze();
            }

            // Verify r matches proof (sanity check, should always pass for honest prover)
            for (0..num_vars) |i| {
                if (!r[i].eql(proof.r[i])) {
                    return null; // transcript mismatch
                }
            }

            // 2. Verify sumcheck with claimed_sum = 0
            const challenges = try allocator.alloc(F, num_vars);
            defer allocator.free(challenges);

            const final_claim = ProductSC.verify(
                F.zero, // zerocheck claims the sum is zero
                proof.rounds[0..num_vars],
                transcript,
                challenges,
            ) catch return null;

            // 3. Check eq(challenges, r) = proof.final_eq
            const expected_eq = eq.eqEvalField(F, challenges, r[0..num_vars]);
            if (!expected_eq.eql(proof.final_eq)) {
                return null;
            }

            // 4. Check final_f · final_eq = final_claim
            if (!proof.final_f.mul(proof.final_eq).eql(final_claim)) {
                return null;
            }

            // 5. Return eval claim for PCS
            //    Caller must verify: f(challenges) = proof.final_f
            const point = try allocator.alloc(F, num_vars);
            @memcpy(point, challenges);

            return EvalClaim{
                .point = point,
                .value = proof.final_f,
            };
        }
    };
}

// ============ Tests ============ //

const M31 = @import("../fields/mersenne31.zig").Mersenne31;

test "zerocheck accepts zero polynomial" {
    const ZC = Zerocheck(M31, 8);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // f(x) = 0 everywhere
    const f_evals = [_]M31{ M31.zero, M31.zero, M31.zero, M31.zero };

    // Prove
    var prover_transcript = Transcript(M31).init("zerocheck-test");
    const result = try ZC.prove(allocator, &f_evals, &prover_transcript);

    // Verify
    var verifier_transcript = Transcript(M31).init("zerocheck-test");
    const claim = try ZC.verify(allocator, &result.proof, &verifier_transcript);

    // Should succeed
    try std.testing.expect(claim != null);

    // Eval claim should have value = 0 (since f is zero everywhere)
    try std.testing.expect(claim.?.value.eql(M31.zero));
}

test "zerocheck rejects non-zero polynomial" {
    const ZC = Zerocheck(M31, 8);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // f(x) is NOT zero everywhere
    const f_evals = [_]M31{
        M31.fromU64(1), // non-zero!
        M31.zero,
        M31.zero,
        M31.zero,
    };

    // Prove (prover is dishonest, but follows protocol)
    var prover_transcript = Transcript(M31).init("zerocheck-test");
    const result = try ZC.prove(allocator, &f_evals, &prover_transcript);

    // Verify should fail because sum ≠ 0
    var verifier_transcript = Transcript(M31).init("zerocheck-test");
    const claim = try ZC.verify(allocator, &result.proof, &verifier_transcript);

    // Should fail (sumcheck check will fail since actual sum ≠ 0)
    try std.testing.expect(claim == null);
}

test "zerocheck with larger polynomial" {
    const ZC = Zerocheck(M31, 8);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // f(x) = 0 on {0,1}^4 (16 points)
    var f_evals: [16]M31 = undefined;
    for (&f_evals) |*e| {
        e.* = M31.zero;
    }

    var prover_transcript = Transcript(M31).init("zerocheck-large");
    const result = try ZC.prove(allocator, &f_evals, &prover_transcript);

    var verifier_transcript = Transcript(M31).init("zerocheck-large");
    const claim = try ZC.verify(allocator, &result.proof, &verifier_transcript);

    try std.testing.expect(claim != null);
    try std.testing.expect(claim.?.value.eql(M31.zero));
    try std.testing.expectEqual(@as(usize, 4), claim.?.point.len);
}

test "zerocheck eval claim matches direct evaluation" {
    const ZC = Zerocheck(M31, 8);
    const multilinear = @import("../poly/multilinear.zig");

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Use a polynomial that IS zero on the hypercube
    // but has non-trivial structure when extended
    // f(x,y) = x(1-x) + y(1-y) - but evaluated on booleans this is always 0
    // Actually let's just use the zero polynomial for simplicity
    const f_evals = [_]M31{ M31.zero, M31.zero, M31.zero, M31.zero };

    var prover_transcript = Transcript(M31).init("zerocheck-eval");
    const result = try ZC.prove(allocator, &f_evals, &prover_transcript);

    // The eval claim says f(point) = value
    // Verify by direct multilinear evaluation
    var f_copy = f_evals;
    const direct_eval = multilinear.evaluate(M31, &f_copy, result.eval_point);

    try std.testing.expect(direct_eval.eql(result.proof.final_f));
}

test "zerocheck proof contains correct challenges" {
    const ZC = Zerocheck(M31, 8);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const f_evals = [_]M31{ M31.zero, M31.zero, M31.zero, M31.zero };

    var prover_transcript = Transcript(M31).init("zerocheck-challenges");
    const result = try ZC.prove(allocator, &f_evals, &prover_transcript);

    // eval_point should have num_vars elements
    try std.testing.expectEqual(@as(usize, 2), result.eval_point.len);

    // r should also have num_vars elements
    try std.testing.expectEqual(@as(usize, 2), result.proof.getR().len);
}
