const std = @import("std");
const Sumcheck = @import("sumcheck.zig").Sumcheck;
const MerkleTree = @import("merkle.zig").MerkleTree;
const Transcript = @import("transcript.zig").Transcript;
const multilinear = @import("poly/multilinear.zig");

const e = error{CommitmentMismatch};

pub fn SumcheckProof(comptime F: type) type {
    const SC = Sumcheck(F);
    const Merkle = MerkleTree(F);

    return struct { commitment: Merkle.Commitment, rounds: []SC.RoundPoly, final_eval: F };
}

pub fn proveSum(
    comptime F: type,
    evals: []F,
    rounds_buf: []Sumcheck(F).RoundPoly,
    transcript: *Transcript(F),
) SumcheckProof(F) {
    // 1. commit to evals
    const commitment = MerkleTree(F).commit(evals);
    // 2. add commitment to transcript (binds proof to polynomial)
    transcript.absorbBytes(&commitment);
    // 3. run sumcheck
    const final_eval = Sumcheck(F).prove(evals, rounds_buf, transcript);

    return .{
        .commitment = commitment,
        .rounds = rounds_buf,
        .final_eval = final_eval,
    };
}

/// Verifier: check complete proof
pub fn verifySum(
    comptime F: type,
    claimed_sum: F,
    proof: SumcheckProof(F),
    evals: []F, // mutable, will be used as scratch space
    transcript: *Transcript(F),
    challenges: []F, // buffer for challenges, len must match proof.rounds.len
) !void {
    // 1. Check commitment matches revealed evals
    if (!MerkleTree(F).verifyAll(evals, proof.commitment)) {
        return error.CommitmentMismatch;
    }

    // 2. Add commitment to transcript
    transcript.absorbBytes(&proof.commitment);

    // 3. Verify sumcheck rounds, collect challenges
    const expected_final = try Sumcheck(F).verify(claimed_sum, proof.rounds, transcript, challenges);

    // 4. Evaluate polynomial at challenge point and check
    const actual_eval = multilinear.evaluate(F, evals, challenges);

    if (!actual_eval.eql(expected_final)) {
        return error.FinalEvalMismatch;
    }
}

// TESTS

const M31 = @import("fields/mersenne31.zig").Mersenne31;

test "end-to-end sumcheck proof" {
    // Polynomial: f(x₀, x₁) with 4 evaluations
    var evals = [_]M31{
        M31.fromU64(1), // f(0,0)
        M31.fromU64(4), // f(0,1)
        M31.fromU64(3), // f(1,0)
        M31.fromU64(10), // f(1,1)
    };
    const claimed_sum = M31.fromU64(18); // 1 + 4 + 3 + 10
    const num_vars = 2;

    // Prover side
    var prover_transcript = Transcript(M31).init("sumcheck-proof");
    var rounds: [num_vars]Sumcheck(M31).RoundPoly = undefined;
    const proof = proveSum(M31, &evals, &rounds, &prover_transcript);

    // Verifier side (fresh transcript, same domain)
    var verifier_transcript = Transcript(M31).init("sumcheck-proof");
    var verifier_evals = [_]M31{
        M31.fromU64(1),
        M31.fromU64(4),
        M31.fromU64(3),
        M31.fromU64(10),
    };
    var challenges: [num_vars]M31 = undefined;

    try verifySum(M31, claimed_sum, proof, &verifier_evals, &verifier_transcript, &challenges);
}

test "proof fails with wrong commitment" {
    var evals = [_]M31{
        M31.fromU64(1),
        M31.fromU64(4),
        M31.fromU64(3),
        M31.fromU64(10),
    };
    const claimed_sum = M31.fromU64(18);
    const num_vars = 2;

    var prover_transcript = Transcript(M31).init("sumcheck-proof");
    var rounds: [num_vars]Sumcheck(M31).RoundPoly = undefined;
    const proof = proveSum(M31, &evals, &rounds, &prover_transcript);

    // Verifier uses different evals (won't match commitment)
    var bad_evals = [_]M31{
        M31.fromU64(99),
        M31.fromU64(99),
        M31.fromU64(99),
        M31.fromU64(99),
    };
    var verifier_transcript = Transcript(M31).init("sumcheck-proof");
    var challenges: [num_vars]M31 = undefined;

    const result = verifySum(M31, claimed_sum, proof, &bad_evals, &verifier_transcript, &challenges);
    try std.testing.expectError(error.CommitmentMismatch, result);
}

test "proof fails with wrong claimed sum" {
    var evals = [_]M31{
        M31.fromU64(1),
        M31.fromU64(4),
        M31.fromU64(3),
        M31.fromU64(10),
    };
    const num_vars = 2;

    var prover_transcript = Transcript(M31).init("sumcheck-proof");
    var rounds: [num_vars]Sumcheck(M31).RoundPoly = undefined;
    const proof = proveSum(M31, &evals, &rounds, &prover_transcript);

    var verifier_evals = [_]M31{
        M31.fromU64(1),
        M31.fromU64(4),
        M31.fromU64(3),
        M31.fromU64(10),
    };
    var verifier_transcript = Transcript(M31).init("sumcheck-proof");
    var challenges: [num_vars]M31 = undefined;

    const wrong_sum = M31.fromU64(999);
    const result = verifySum(M31, wrong_sum, proof, &verifier_evals, &verifier_transcript, &challenges);

    // Should fail during sumcheck verification
    try std.testing.expectError(error.RoundSumMismatch, result);
}
