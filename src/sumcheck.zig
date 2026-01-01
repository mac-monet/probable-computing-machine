const std = @import("std");
const multilinear = @import("poly/multilinear.zig");
const Transcript = @import("transcript.zig").Transcript;

const assert = std.debug.assert;

pub fn Sumcheck(comptime F: type) type {
    return struct {
        pub const VerifyError = error{RoundSumMismatch};

        pub const RoundPoly = struct {
            const Self = @This();

            eval_0: F, // g(0)
            eval_1: F, // g(1)

            /// Evaluate at challenge point: g(r) = g(0) + r(g(1) - g(0))
            pub fn evaluate(self: *const Self, r: F) F {
                return self.eval_0.add(r.mul(self.eval_1.sub(self.eval_0)));
            }
        };

        /// Prover: generate all round polynomials, return final evaluation
        /// Mutates evals in-place.
        pub fn prove(evals: []F, rounds: []RoundPoly, transcript: *Transcript(F)) F {
            var current = evals;

            for (rounds) |*round| {
                // Step 1: record round polynomial
                const halves = multilinear.sumHalves(F, current);
                round.* = .{ .eval_0 = halves[0], .eval_1 = halves[1] };

                // Absorb round polynomial into transcript
                transcript.absorb(halves[0]);
                transcript.absorb(halves[1]);

                // Derive challenge from transcript
                const c = transcript.squeeze();

                // Bind and shrink for next round
                current = multilinear.bind(F, current, c);
            }

            // After n rounds, current has 1 element
            return current[0];
        }

        pub fn verify(claimed_sum: F, rounds: []const RoundPoly, transcript: *Transcript(F), challenges_out: []F) VerifyError!F {
            std.debug.assert(rounds.len == challenges_out.len);

            var current_claim = claimed_sum;

            for (rounds, 0..) |round, i| {
                // Absorb round polynomials (same as prover)
                transcript.absorb(round.eval_0);
                transcript.absorb(round.eval_1);

                // Check that g(0) + g(1) = current_claim
                if (!round.eval_0.add(round.eval_1).eql(current_claim)) {
                    return error.RoundSumMismatch;
                }
                const c = transcript.squeeze();
                challenges_out[i] = c;

                // Update claim for next round
                current_claim = round.evaluate(c);
            }

            return current_claim;
        }
    };
}

const M31 = @import("fields/mersenne31.zig").Mersenne31;

test "Sumcheck prove and verify with transcript" {
    var evals = [_]M31{
        M31.fromU64(1), // f(0,0)
        M31.fromU64(4), // f(0,1)
        M31.fromU64(3), // f(1,0)
        M31.fromU64(10), // f(1,1)
    };

    const claimed_sum = M31.fromU64(18);

    // Prover
    var prover_transcript = Transcript(M31).init("sumcheck-test");
    var rounds: [2]Sumcheck(M31).RoundPoly = undefined;
    const final_eval = Sumcheck(M31).prove(&evals, &rounds, &prover_transcript);

    // Verifier (fresh transcript, same domain)
    var verifier_transcript = Transcript(M31).init("sumcheck-test");
    var challenges: [2]M31 = undefined;
    const expected_eval = try Sumcheck(M31).verify(claimed_sum, &rounds, &verifier_transcript, &challenges);

    try std.testing.expect(final_eval.eql(expected_eval));
}

test "sumcheck verification fails on bad claim" {
    var evals = [_]M31{
        M31.fromU64(1),
        M31.fromU64(4),
        M31.fromU64(3),
        M31.fromU64(10),
    };
    const wrong_sum = M31.fromU64(99); // actual sum is 18

    var prover_transcript = Transcript(M31).init("sumcheck-test");
    var rounds: [2]Sumcheck(M31).RoundPoly = undefined;
    _ = Sumcheck(M31).prove(&evals, &rounds, &prover_transcript);

    var verifier_transcript = Transcript(M31).init("sumcheck-test");
    var challenges: [2]M31 = undefined;
    const result = Sumcheck(M31).verify(wrong_sum, &rounds, &verifier_transcript, &challenges);

    try std.testing.expectError(Sumcheck(M31).VerifyError.RoundSumMismatch, result);
}

test "sumcheck verification fails on tampered round" {
    var evals = [_]M31{
        M31.fromU64(1),
        M31.fromU64(4),
        M31.fromU64(3),
        M31.fromU64(10),
    };
    const claimed_sum = M31.fromU64(18);

    var prover_transcript = Transcript(M31).init("sumcheck-test");
    var rounds: [2]Sumcheck(M31).RoundPoly = undefined;
    _ = Sumcheck(M31).prove(&evals, &rounds, &prover_transcript);

    // Tamper with round 1
    rounds[1].eval_0 = M31.fromU64(999);

    var verifier_transcript = Transcript(M31).init("sumcheck-test");
    var challenges: [2]M31 = undefined;
    const result = Sumcheck(M31).verify(claimed_sum, &rounds, &verifier_transcript, &challenges);

    try std.testing.expectError(Sumcheck(M31).VerifyError.RoundSumMismatch, result);
}
