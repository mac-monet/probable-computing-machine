const std = @import("std");
const Eq = @import("poly/eq.zig");
const Multilinear = @import("poly/multilinear.zig");
const Transcript = @import("transcript.zig").Transcript;

pub fn ProductSumcheck(comptime F: type) type {
    return struct {
        pub const VerifyError = error{RoundSumMismatch}; // TODO duplicate

        pub const RoundPoly2 = struct {
            const Self = @This();

            eval_0: F, // g(0)
            eval_1: F, // g(1)
            eval_2: F, // g(2)

            /// Verify: g(0) + g(1) should equal the claimed sum
            pub fn checkSum(self: *const Self, claimed: F) bool {
                return self.eval_0.add(self.eval_1).eql(claimed);
            }

            /// Evaluate at challenge point using Lagrange interpolation
            pub fn evaluate(self: *const Self, r: F) F {
                // g(X) passes through (0, eval_0), (1, eval_1), (2, eval_2)
                // Lagrange basis:
                // L_0(X) = (X-1)(X-2) / (0-1)(0-2) = (X-1)(X-2) / 2
                // L_1(X) = (X-0)(X-2) / (1-0)(1-2) = X(X-2) / (-1)
                // L_2(X) = (X-0)(X-1) / (2-0)(2-1) = X(X-1) / 2

                const two = F.one.add(F.one);
                const two_inv = two.inv();

                const r_m1 = r.sub(F.one);
                const r_m2 = r.sub(two);

                const t0 = self.eval_0.mul(r_m1).mul(r_m2).mul(two_inv);
                const t1 = self.eval_1.mul(r).mul(r_m2).neg();
                const t2 = self.eval_2.mul(r).mul(r_m1).mul(two_inv);

                return t0.add(t1).add(t2);
            }
        };

        pub fn computeRound(f_evals: []const F, eq_evals: []const F) RoundPoly2 {
            std.debug.assert(f_evals.len == eq_evals.len);

            const half = f_evals.len / 2;
            var g0 = F.zero;
            var g1 = F.zero;
            var g2 = F.zero;

            for (0..half) |i| {
                const f_lo = f_evals[i];
                const f_hi = f_evals[i + half];
                const eq_lo = eq_evals[i];
                const eq_hi = eq_evals[i + half];

                g0 = g0.add(f_lo.mul(eq_lo));
                g1 = g1.add(f_hi.mul(eq_hi));

                const f_2 = f_hi.double().sub(f_lo);
                const eq_2 = eq_hi.double().sub(eq_lo);
                g2 = g2.add(f_2.mul(eq_2));
            }

            return .{ .eval_0 = g0, .eval_1 = g1, .eval_2 = g2 };
        }

        // TODO delete, replaced with basefold's usage
        /// Prover: generate all round polynomials for f·eq sumcheck
        /// Mutates both f_evals and eq_evals in-place.
        /// Returns final f·eq evaluation at the random point.
        pub fn prove(
            f_evals: []F,
            eq_evals: []F,
            rounds: []RoundPoly2,
            transcript: *Transcript(F),
        ) F {
            std.debug.assert(f_evals.len == eq_evals.len);

            var f_current = f_evals;
            var eq_current = eq_evals;

            for (rounds) |*round| {
                // Compute degree-2 round polynomial
                round.* = computeRound(f_current, eq_current);

                // Absorb into transcript
                transcript.absorb(round.eval_0);
                transcript.absorb(round.eval_1);
                transcript.absorb(round.eval_2);

                // Derive challenge
                const c = transcript.squeeze();

                // Bind both polynomials
                f_current = Multilinear.bind(F, f_current, c);
                eq_current = Multilinear.bind(F, eq_current, c);
            }

            // Final evaluation: f(r) · eq(r)
            return f_current[0].mul(eq_current[0]);
        }

        pub fn verify(
            claimed_sum: F,
            rounds: []const RoundPoly2,
            transcript: *Transcript(F),
            challenges_out: []F,
        ) VerifyError!F {
            std.debug.assert(rounds.len == challenges_out.len);

            var current_claim = claimed_sum;

            for (rounds, 0..) |round, i| {
                transcript.absorb(round.eval_0);
                transcript.absorb(round.eval_1);
                transcript.absorb(round.eval_2);

                // Check: g(0) + g(1) = current_claim
                if (!round.eval_0.add(round.eval_1).eql(current_claim)) {
                    return error.RoundSumMismatch;
                }

                const c = transcript.squeeze();
                challenges_out[i] = c;

                // Next claim: g(c)
                current_claim = round.evaluate(c);
            }

            return current_claim;
        }
    };
}

const M31 = @import("fields/mersenne31.zig").Mersenne31;

test "product sumcheck round - sum check" {
    const ProductSC = ProductSumcheck(M31);

    // f(x0, x1) with some values
    var f_evals = [_]M31{
        M31.fromU64(1),
        M31.fromU64(4),
        M31.fromU64(3),
        M31.fromU64(10),
    };

    // Build eq table for some random point r
    const r = [_]M31{ M31.fromU64(5), M31.fromU64(7) };
    var eq_evals: [4]M31 = undefined;
    Eq.eqEvals(M31, &r, &eq_evals);

    // Claimed sum: Σ f(b)·eq(b,r)
    var claimed = M31.zero;
    for (f_evals, eq_evals) |f_, eq_| {
        claimed = claimed.add(f_.mul(eq_));
    }

    // Compute round polynomial
    const round = ProductSC.computeRound(&f_evals, &eq_evals);

    // Verifier check: g(0) + g(1) = claimed
    try std.testing.expect(round.checkSum(claimed));
}

test "product sumcheck prove/verify roundtrip" {
    const ProductSC = ProductSumcheck(M31);

    // f(x0, x1) = 1 + 2x0 + 3x1 + 4x0x1
    var f_evals = [_]M31{
        M31.fromU64(1), // f(0,0)
        M31.fromU64(4), // f(0,1)
        M31.fromU64(3), // f(1,0)
        M31.fromU64(10), // f(1,1)
    };
    const num_vars = 2;

    // Opening point r
    const r = [_]M31{ M31.fromU64(5), M31.fromU64(7) };

    // Claimed value: f(r) = f(5,7) = 1 + 2*5 + 3*7 + 4*5*7 = 1 + 10 + 21 + 140 = 172
    const claimed_value = M31.fromU64(172);

    // Build eq table
    var eq_evals: [4]M31 = undefined;
    Eq.eqEvals(M31, &r, &eq_evals);

    // The sum Σ f(b)·eq(b,r) should equal f(r)
    var expected_sum = M31.zero;
    for (f_evals, eq_evals) |f, eq| {
        expected_sum = expected_sum.add(f.mul(eq));
    }
    try std.testing.expect(expected_sum.eql(claimed_value));

    // Prover
    var prover_transcript = Transcript(M31).init("product-sumcheck");
    var rounds: [num_vars]ProductSC.RoundPoly2 = undefined;
    const final_eval = ProductSC.prove(&f_evals, &eq_evals, &rounds, &prover_transcript);

    // Verifier
    var verifier_transcript = Transcript(M31).init("product-sumcheck");
    var challenges: [num_vars]M31 = undefined;
    const verified_claim = try ProductSC.verify(claimed_value, &rounds, &verifier_transcript, &challenges);

    // Final check: f(c)·eq(c,r) should equal verified_claim
    // Prover returned final_eval = f(c)·eq(c,r)
    try std.testing.expect(final_eval.eql(verified_claim));

    // Verifier can independently compute eq(c, r)
    const eq_at_c = Eq.eqEvalField(M31, &challenges, &r);

    // And derive what f(c) must be
    const f_at_c = verified_claim.mul(eq_at_c.inv());

    // This should match the prover's folded f value (which is f_evals[0] after prove)
    try std.testing.expect(f_at_c.eql(f_evals[0]));
}
