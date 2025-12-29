const std = @import("std");
const multilinear = @import("poly/multilinear.zig");

const assert = std.debug.assert;

pub fn Sumcheck(comptime F: type) type {
    return struct {
        pub const VerifyError = error{RoundSumMismatch};

        pub const RoundPoly = struct {
            eval_0: F, // g(0)
            eval_1: F, // g(1)

            /// Evaluate at challenge point: g(r) = g(0) + r(g(1) - g(0))
            pub fn evaluate(self: RoundPoly, r: F) F {
                return self.eval_0.add(r.mul(self.eval_1.sub(self.eval_0)));
            }
        };

        /// Prover: generate all round polynomials, return final evaluation
        /// Mutates evals in-place.
        pub fn prove(evals: []F, challenges: []const F, rounds: []RoundPoly) F {
            assert(evals.len == @as(usize, 1) << @intCast(challenges.len));
            assert(rounds.len == challenges.len);

            var current = evals;

            for (challenges, 0..) |r, i| {
                // Step 1: record round polynomial
                const halves = multilinear.sumHalves(F, current);
                rounds[i] = .{ .eval_0 = halves[0], .eval_1 = halves[1] };

                // Step 2: bind and shrink for next round
                current = multilinear.bind(F, current, r);
            }

            // After n rounds, current has 1 element
            return current[0];
        }

        pub fn verify(
            claimed_sum: F,
            challenges: []const F,
            rounds: []const RoundPoly,
        ) VerifyError!F {
            assert(rounds.len == challenges.len);

            var current_claim = claimed_sum;

            for (rounds, challenges) |round, r| {
                if (!round.eval_0.add(round.eval_1).eql(current_claim)) {
                    return error.RoundSumMismatch;
                }
                // Update claim for next round
                current_claim = round.evaluate(r);
            }

            return current_claim;
        }
    };
}

const M31 = @import("fields/mersenne31.zig").Mersenne31;

test "Sumcheck prove and verify" {
    var evals = [_]M31{
        M31.fromU64(1), // f(0,0)
        M31.fromU64(4), // f(0,1)
        M31.fromU64(3), // f(1,0)
        M31.fromU64(10), // f(1,1)
    };

    const claimed_sum = M31.fromU64(18);
    const challenges = [_]M31{ M31.fromU64(2), M31.fromU64(15) };
    var rounds: [2]Sumcheck(M31).RoundPoly = undefined;

    const final_eval = Sumcheck(M31).prove(&evals, &challenges, &rounds);
    const expected_eval = try Sumcheck(M31).verify(claimed_sum, &challenges, &rounds);

    try std.testing.expect(final_eval.eql(expected_eval));
}

test "Sumcheck verify rejects wrong claimed sum" {
    var evals = [_]M31{
        M31.fromU64(1),
        M31.fromU64(4),
        M31.fromU64(3),
        M31.fromU64(10),
    };

    const wrong_sum = M31.fromU64(99);
    const challenges = [_]M31{ M31.fromU64(2), M31.fromU64(15) };
    var rounds: [2]Sumcheck(M31).RoundPoly = undefined;

    _ = Sumcheck(M31).prove(&evals, &challenges, &rounds);

    const result = Sumcheck(M31).verify(wrong_sum, &challenges, &rounds);
    try std.testing.expectError(error.RoundSumMismatch, result);
}
