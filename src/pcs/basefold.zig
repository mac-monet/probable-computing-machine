const std = @import("std");
const Merkle = @import("../merkle.zig");
const Transcript = @import("../transcript.zig").Transcript;
const Multilinear = @import("../poly/multilinear.zig");
const Eq = @import("../poly/eq.zig");
const ProductSumcheck = @import("../product_sumcheck.zig").ProductSumcheck;

pub fn Basefold(comptime F: type) type {
    const product = ProductSumcheck(F);
    const merkle = Merkle.MerkleTree(F);

    return struct {
        pub const Commitment = merkle.Commitment;

        pub const Proof = struct {
            rounds: []const product.RoundPoly2,
            commitments: []const Commitment, // commitment[i] = folded poly after round i
            final_f: F,
            final_eq: F,
        };

        pub fn prove(allocator: std.mem.Allocator, f_evals: []F, eq_evals: []F, num_vars: usize, transcript: *Transcript(F)) !Proof {
            const rounds = try allocator.alloc(product.RoundPoly2, num_vars);
            const commitments = try allocator.alloc(Commitment, num_vars);

            var f_current = f_evals;
            var eq_current = eq_evals;

            for (0..num_vars) |i| {
                // Commit to current f
                commitments[i] = merkle.commit(f_current);

                transcript.absorbBytes(&commitments[i]);

                // Compute round poly
                rounds[i] = product.computeRound(f_current, eq_current);

                transcript.absorb(rounds[i].eval_0);
                transcript.absorb(rounds[i].eval_1);
                transcript.absorb(rounds[i].eval_2);

                // TODO for now, use deterministic challenge (will add transcript later)
                const challenge = transcript.squeeze();

                // Fold both
                f_current = Multilinear.bind(F, f_current, challenge);
                eq_current = Multilinear.bind(F, eq_current, challenge);
            }

            return Proof{
                .rounds = rounds,
                .commitments = commitments,
                .final_f = f_current[0],
                .final_eq = eq_current[0],
            };
        }

        /// Verifier: check sumcheck rounds (phase 1, before query phase)
        pub fn verify(
            claimed_value: F,
            proof: *const Proof,
            r: []const F,
            transcript: *Transcript(F),
            challenges_out: []F,
        ) product.VerifyError!F {
            std.debug.assert(proof.rounds.len == challenges_out.len);
            std.debug.assert(proof.rounds.len == r.len);

            var current_claim = claimed_value;

            for (proof.rounds, proof.commitments, 0..) |round, commitment, i| {
                // Absorb commitment (same as prover)
                transcript.absorbBytes(&commitment);

                // Absorb round polynomial
                transcript.absorb(round.eval_0);
                transcript.absorb(round.eval_1);
                transcript.absorb(round.eval_2);

                // Check: g(0) + g(1) = current_claim
                if (!round.eval_0.add(round.eval_1).eql(current_claim)) {
                    return error.RoundSumMismatch;
                }

                const challenge = transcript.squeeze();
                challenges_out[i] = challenge;

                // Update claim: g(challenge)
                current_claim = round.evaluate(challenge);
            }

            return current_claim;
        }
    };
}

const M31 = @import("../fields/mersenne31.zig").Mersenne31;
const basefold = @import("basefold.zig").Basefold(M31);
const eq = @import("../poly/eq.zig");

test "basefold prove/verify sumcheck" {
    const allocator = std.testing.allocator;

    var f_evals = [_]M31{
        M31.fromU64(1), M31.fromU64(2), M31.fromU64(3), M31.fromU64(4),
        M31.fromU64(5), M31.fromU64(6), M31.fromU64(7), M31.fromU64(8),
    };
    const num_vars = 3;

    const r = [_]M31{ M31.fromU64(10), M31.fromU64(20), M31.fromU64(30) };

    var eq_evals: [8]M31 = undefined;
    eq.eqEvals(M31, &r, &eq_evals);

    // Compute claimed value: f(r)
    var claimed_value = M31.zero;
    for (f_evals, eq_evals) |f, e| {
        claimed_value = claimed_value.add(f.mul(e));
    }

    // Prover
    var prover_transcript = Transcript(M31).init("basefold");
    const proof = try basefold.prove(allocator, &f_evals, &eq_evals, num_vars, &prover_transcript);
    defer allocator.free(proof.rounds);
    defer allocator.free(proof.commitments);

    // Verifier
    var verifier_transcript = Transcript(M31).init("basefold");
    var challenges: [3]M31 = undefined;
    const final_claim = try basefold.verify(claimed_value, &proof, &r, &verifier_transcript, &challenges);

    // Final check: f(c) * eq(c, r) = final_claim
    const eq_at_c = eq.eqEvalField(M31, &challenges, &r);
    const expected_final = proof.final_f.mul(proof.final_eq);

    try std.testing.expect(final_claim.eql(expected_final));
    try std.testing.expect(proof.final_eq.eql(eq_at_c));
}
