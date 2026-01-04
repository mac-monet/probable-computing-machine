const std = @import("std");
const Transcript = @import("../core/transcript.zig").Transcript;

/// Configuration for generic sumcheck.
pub const SumcheckConfig = struct {
    /// Degree of round polynomial (1 = linear, 2 = product, etc.)
    degree: comptime_int = 1,

    /// Number of polynomials being composed
    /// 1 = single poly, 2 = product of two, etc.
    num_polys: comptime_int = 1,
};

/// Generic sumcheck protocol parameterized by field and configuration.
/// Supports linear sumcheck (degree=1, num_polys=1) and product sumcheck (degree=2, num_polys=2).
pub fn Sumcheck(comptime F: type, comptime config: SumcheckConfig) type {
    const degree = config.degree;
    const num_polys = config.num_polys;

    return struct {
        const Self = @This();

        pub const VerifyError = error{RoundSumMismatch};

        /// Round polynomial: evaluations at 0, 1, ..., degree
        pub const RoundPoly = [degree + 1]F;

        pub const ProveResult = struct {
            rounds: []const RoundPoly,
            challenges: []const F,
            final_evals: [num_polys]F,
        };

        pub const VerifyResult = struct {
            challenges: []const F,
            final_claim: F,
            valid: bool,
        };

        /// Compute single round polynomial from current evaluations.
        /// polys: array of polynomial evaluations (1 for linear, 2 for product)
        pub fn computeRound(polys: [num_polys][]const F) RoundPoly {
            std.debug.assert(polys[0].len >= 2);

            var result: RoundPoly = undefined;

            // Evaluate at each point 0, 1, ..., degree
            inline for (0..degree + 1) |eval_point| {
                result[eval_point] = computeRoundEval(polys, eval_point);
            }

            return result;
        }

        fn computeRoundEval(polys: [num_polys][]const F, point: usize) F {
            const half = polys[0].len / 2;
            var sum = F.zero;

            for (0..half) |i| {
                // Interpolate each poly at point, then combine
                var term = F.one;
                inline for (polys) |poly| {
                    const lo = poly[i];
                    const hi = poly[i + half];
                    const interpolated = interpolateAt(lo, hi, point);
                    term = term.mul(interpolated);
                }
                sum = sum.add(term);
            }

            return sum;
        }

        fn interpolateAt(lo: F, hi: F, point: usize) F {
            // Linear interpolation: lo + point * (hi - lo)
            // For point=0: lo, point=1: hi, point=2: 2*hi - lo, etc.
            if (point == 0) return lo;
            if (point == 1) return hi;
            const delta = hi.sub(lo);
            return lo.add(delta.mul(F.fromU64(point)));
        }

        /// Evaluate round polynomial at challenge point.
        pub fn evalRoundPoly(poly: RoundPoly, challenge: F) F {
            if (degree == 1) {
                // Linear: poly[0] + challenge * (poly[1] - poly[0])
                return poly[0].add(challenge.mul(poly[1].sub(poly[0])));
            } else {
                return lagrangeEval(&poly, challenge);
            }
        }

        fn lagrangeEval(evals: []const F, point: F) F {
            // General Lagrange interpolation through (0, evals[0]), (1, evals[1]), ...
            var result = F.zero;

            for (0..evals.len) |i| {
                var basis = F.one;
                const i_field = F.fromU64(i);

                for (0..evals.len) |j| {
                    if (i != j) {
                        const j_field = F.fromU64(j);
                        // basis *= (point - j) / (i - j)
                        const num = point.sub(j_field);
                        const denom = i_field.sub(j_field);
                        basis = basis.mul(num).mul(denom.inv());
                    }
                }
                result = result.add(evals[i].mul(basis));
            }
            return result;
        }

        /// Fold polynomial in-place, returns smaller slice.
        /// new[i] = lo + challenge * (hi - lo)
        pub fn fold(evals: []F, challenge: F) []F {
            const half = evals.len / 2;
            F.linearCombineBatch(evals[0..half], evals[0..half], evals[half..], challenge);
            return evals[0..half];
        }

        /// Prove sumcheck with working buffers.
        /// work_polys: mutable slices that will be folded in-place
        /// rounds_out: pre-allocated array for round polynomials
        /// challenges_out: pre-allocated array for challenges
        pub fn prove(
            work_polys: *[num_polys][]F,
            transcript: *Transcript(F),
            rounds_out: []RoundPoly,
            challenges_out: []F,
        ) [num_polys]F {
            const num_vars = @ctz(work_polys[0].len);
            std.debug.assert(rounds_out.len == num_vars);
            std.debug.assert(challenges_out.len == num_vars);

            for (0..num_vars) |round| {
                // Compute round polynomial
                const round_poly = computeRound(work_polys.*);
                rounds_out[round] = round_poly;

                // Absorb into transcript
                inline for (round_poly) |eval| {
                    transcript.absorb(eval);
                }

                // Derive challenge
                const challenge = transcript.squeeze();
                challenges_out[round] = challenge;

                // Fold all polynomials
                inline for (work_polys) |*wp| {
                    wp.* = fold(wp.*, challenge);
                }
            }

            // Extract final evaluations
            var final_evals: [num_polys]F = undefined;
            inline for (0..num_polys) |p| {
                final_evals[p] = work_polys[p][0];
            }

            return final_evals;
        }

        /// Verify sumcheck rounds.
        pub fn verify(
            claimed_sum: F,
            rounds: []const RoundPoly,
            transcript: *Transcript(F),
            challenges_out: []F,
        ) VerifyError!F {
            std.debug.assert(rounds.len == challenges_out.len);

            var current_claim = claimed_sum;

            for (rounds, 0..) |round_poly, i| {
                // Absorb (same as prover)
                inline for (round_poly) |eval| {
                    transcript.absorb(eval);
                }

                // Check: round_poly(0) + round_poly(1) = current_claim
                if (!round_poly[0].add(round_poly[1]).eql(current_claim)) {
                    return error.RoundSumMismatch;
                }

                // Derive challenge
                const challenge = transcript.squeeze();
                challenges_out[i] = challenge;

                // Evaluate round poly at challenge for next claim
                current_claim = evalRoundPoly(round_poly, challenge);
            }

            return current_claim;
        }
    };
}

// Convenience aliases
pub fn LinearSumcheck(comptime F: type) type {
    return Sumcheck(F, .{ .degree = 1, .num_polys = 1 });
}

pub fn ProductSumcheck(comptime F: type) type {
    return Sumcheck(F, .{ .degree = 2, .num_polys = 2 });
}

// ============ Tests ============ //

const M31 = @import("../fields/mersenne31.zig").Mersenne31;

test "linear sumcheck prove and verify" {
    const SC = LinearSumcheck(M31);

    var evals = [_]M31{
        M31.fromU64(1), // f(0,0)
        M31.fromU64(4), // f(0,1)
        M31.fromU64(3), // f(1,0)
        M31.fromU64(10), // f(1,1)
    };

    const claimed_sum = M31.fromU64(18); // 1 + 4 + 3 + 10

    // Prover
    var prover_transcript = @import("../core/transcript.zig").Transcript(M31).init("linear-sumcheck");
    var rounds: [2]SC.RoundPoly = undefined;
    var challenges: [2]M31 = undefined;
    var work_polys = [_][]M31{evals[0..]};
    const final_evals = SC.prove(&work_polys, &prover_transcript, &rounds, &challenges);

    // Verifier
    var verifier_transcript = @import("../core/transcript.zig").Transcript(M31).init("linear-sumcheck");
    var v_challenges: [2]M31 = undefined;
    const final_claim = try SC.verify(claimed_sum, &rounds, &verifier_transcript, &v_challenges);

    // Final claim should match final_evals[0]
    try std.testing.expect(final_claim.eql(final_evals[0]));

    // Challenges should match
    try std.testing.expect(challenges[0].eql(v_challenges[0]));
    try std.testing.expect(challenges[1].eql(v_challenges[1]));
}

test "product sumcheck prove and verify" {
    const SC = ProductSumcheck(M31);
    const Eq = @import("../poly/eq.zig");

    // f(x0, x1)
    var f_evals = [_]M31{
        M31.fromU64(1), // f(0,0)
        M31.fromU64(4), // f(0,1)
        M31.fromU64(3), // f(1,0)
        M31.fromU64(10), // f(1,1)
    };

    // Opening point
    const r = [_]M31{ M31.fromU64(5), M31.fromU64(7) };

    // Build eq table
    var eq_evals: [4]M31 = undefined;
    Eq.eqEvals(M31, &r, &eq_evals);

    // Claimed sum: Σ f(b)·eq(b,r) = f(r)
    var claimed_sum = M31.zero;
    for (f_evals, eq_evals) |f, eq| {
        claimed_sum = claimed_sum.add(f.mul(eq));
    }

    // Prover
    var prover_transcript = @import("../core/transcript.zig").Transcript(M31).init("product-sumcheck");
    var rounds: [2]SC.RoundPoly = undefined;
    var challenges: [2]M31 = undefined;
    var work_polys = [_][]M31{ f_evals[0..], eq_evals[0..] };
    const final_evals = SC.prove(&work_polys, &prover_transcript, &rounds, &challenges);

    // Verifier
    var verifier_transcript = @import("../core/transcript.zig").Transcript(M31).init("product-sumcheck");
    var v_challenges: [2]M31 = undefined;
    const final_claim = try SC.verify(claimed_sum, &rounds, &verifier_transcript, &v_challenges);

    // Final claim should be final_f * final_eq
    const expected_final = final_evals[0].mul(final_evals[1]);
    try std.testing.expect(final_claim.eql(expected_final));
}

test "sumcheck verification fails on wrong claim" {
    const SC = LinearSumcheck(M31);

    var evals = [_]M31{
        M31.fromU64(1),
        M31.fromU64(4),
        M31.fromU64(3),
        M31.fromU64(10),
    };

    const wrong_sum = M31.fromU64(99); // actual sum is 18

    var prover_transcript = @import("../core/transcript.zig").Transcript(M31).init("test");
    var rounds: [2]SC.RoundPoly = undefined;
    var challenges: [2]M31 = undefined;
    var work_polys = [_][]M31{evals[0..]};
    _ = SC.prove(&work_polys, &prover_transcript, &rounds, &challenges);

    var verifier_transcript = @import("../core/transcript.zig").Transcript(M31).init("test");
    var v_challenges: [2]M31 = undefined;
    const result = SC.verify(wrong_sum, &rounds, &verifier_transcript, &v_challenges);

    try std.testing.expectError(SC.VerifyError.RoundSumMismatch, result);
}

test "round polynomial computation matches old implementation" {
    const SC = ProductSumcheck(M31);
    const Eq = @import("../poly/eq.zig");

    const f_evals = [_]M31{
        M31.fromU64(1),
        M31.fromU64(4),
        M31.fromU64(3),
        M31.fromU64(10),
    };

    const r = [_]M31{ M31.fromU64(5), M31.fromU64(7) };
    var eq_evals: [4]M31 = undefined;
    Eq.eqEvals(M31, &r, &eq_evals);

    // New generic implementation
    const round = SC.computeRound(.{ &f_evals, &eq_evals });

    // Manual computation for degree-2
    // g(0) = Σ f_lo * eq_lo
    // g(1) = Σ f_hi * eq_hi
    // g(2) = Σ (2*f_hi - f_lo) * (2*eq_hi - eq_lo)
    const half = f_evals.len / 2;
    var g0 = M31.zero;
    var g1 = M31.zero;
    var g2 = M31.zero;

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

    try std.testing.expect(round[0].eql(g0));
    try std.testing.expect(round[1].eql(g1));
    try std.testing.expect(round[2].eql(g2));
}

test "fold matches multilinear bind" {
    const multilinear = @import("../poly/multilinear.zig");
    const SC = LinearSumcheck(M31);

    var evals1 = [_]M31{
        M31.fromU64(1),
        M31.fromU64(4),
        M31.fromU64(3),
        M31.fromU64(10),
    };
    var evals2 = [_]M31{
        M31.fromU64(1),
        M31.fromU64(4),
        M31.fromU64(3),
        M31.fromU64(10),
    };

    const challenge = M31.fromU64(7);

    const folded1 = SC.fold(&evals1, challenge);
    const folded2 = multilinear.bind(M31, &evals2, challenge);

    try std.testing.expectEqual(folded1.len, folded2.len);
    for (folded1, folded2) |a, b| {
        try std.testing.expect(a.eql(b));
    }
}
