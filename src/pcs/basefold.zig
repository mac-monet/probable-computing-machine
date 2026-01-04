const std = @import("std");

pub fn Basefold(comptime F: type) type {
    const iop_sumcheck = @import("../iop/sumcheck.zig");
    const ProductSC = iop_sumcheck.ProductSumcheck(F);
    const Merkle = @import("../merkle.zig").MerkleTree(F);
    const Transcript = @import("../core/transcript.zig").Transcript;
    const multilinear = @import("../poly/multilinear.zig");
    const eq = @import("../poly/eq.zig");

    return struct {
        const Self = @This();

        pub const Commitment = Merkle.Commitment;
        pub const MerkleProof = Merkle.MerkleProof;

        pub const Config = struct {
            num_queries: usize = 32,
        };

        pub const LayerProof = struct {
            left_value: F,
            right_value: F,
            left_merkle: MerkleProof,
            right_merkle: MerkleProof,
        };

        pub const QueryProof = struct {
            layers: []const LayerProof,
        };

        pub const Proof = struct {
            rounds: []const ProductSC.RoundPoly,
            commitments: []const Commitment,
            final_f: F,
            final_eq: F,
            queries: []const QueryProof,

            pub fn deinit(self: *const Proof, allocator: std.mem.Allocator) void {
                for (self.queries) |query| {
                    for (query.layers) |layer| {
                        allocator.free(layer.left_merkle.path);
                        allocator.free(layer.left_merkle.indices);
                        allocator.free(layer.right_merkle.path);
                        allocator.free(layer.right_merkle.indices);
                    }
                    allocator.free(query.layers);
                }
                allocator.free(self.queries);
                allocator.free(self.rounds);
                allocator.free(self.commitments);
            }
        };

        /// Prover state - internal, holds layers for query generation
        const ProverState = struct {
            layers: [][]F,
            commitments: []Commitment,
            rounds: []ProductSC.RoundPoly,
            final_f: F,
            final_eq: F,
            allocator: std.mem.Allocator,

            fn deinit(self: *ProverState) void {
                for (self.layers) |layer| {
                    self.allocator.free(layer);
                }
                self.allocator.free(self.layers);
                self.allocator.free(self.commitments);
                self.allocator.free(self.rounds);
            }
        };

        // ============ Public API ============ //

        /// Prove that f(r) = claimed_value
        /// f_evals: evaluations of f over {0,1}^n
        /// r: the opening point
        pub fn prove(
            allocator: std.mem.Allocator,
            f_evals: []const F,
            r: []const F,
            config: Config,
        ) !Proof {
            const num_vars = r.len;
            std.debug.assert(f_evals.len == @as(usize, 1) << @intCast(num_vars));

            var transcript = Transcript(F).init("basefold");

            // Build eq table
            const eq_evals = try allocator.alloc(F, f_evals.len);
            defer allocator.free(eq_evals);
            eq.eqEvals(F, r, eq_evals);

            // Phase 1: commit and run sumcheck
            var state = try runSumcheck(allocator, f_evals, eq_evals, num_vars, &transcript);
            defer state.deinit();

            // Derive query indices from transcript
            const max_index = @as(usize, 1) << @intCast(num_vars - 1);
            const query_indices = try deriveQueryIndices(&transcript, config.num_queries, max_index, allocator);
            defer allocator.free(query_indices);

            // Phase 2: generate query proofs
            const queries = try generateQueryProofs(&state, query_indices, allocator);

            // Copy state data into proof (state will be freed)
            const rounds = try allocator.alloc(ProductSC.RoundPoly, num_vars);
            @memcpy(rounds, state.rounds);

            const commitments = try allocator.alloc(Commitment, num_vars);
            @memcpy(commitments, state.commitments);

            return .{
                .rounds = rounds,
                .commitments = commitments,
                .final_f = state.final_f,
                .final_eq = state.final_eq,
                .queries = queries,
            };
        }

        /// Verify a Basefold proof
        pub fn verify(
            allocator: std.mem.Allocator,
            claimed_value: F,
            r: []const F,
            proof: *const Proof,
            config: Config,
        ) !bool {
            const num_vars = proof.commitments.len;
            if (r.len != num_vars) return false;

            var transcript = Transcript(F).init("basefold");

            // Phase 1: verify sumcheck
            const challenges = try allocator.alloc(F, num_vars);
            defer allocator.free(challenges);

            const final_claim = verifySumcheck(claimed_value, proof, &transcript, challenges) catch return false;

            // Check final_eq matches eq(challenges, r)
            const eq_at_c = eq.eqEvalField(F, challenges, r);
            if (!proof.final_eq.eql(eq_at_c)) return false;

            // Check final_claim = final_f * final_eq
            if (!final_claim.eql(proof.final_f.mul(proof.final_eq))) return false;

            // Derive query indices (same as prover)
            const max_index = @as(usize, 1) << @intCast(num_vars - 1);
            const query_indices = try deriveQueryIndices(&transcript, config.num_queries, max_index, allocator);
            defer allocator.free(query_indices);

            // Phase 2: verify query proofs
            return verifyQueries(proof, query_indices, challenges);
        }

        // ============ Internal: Sumcheck ============ //

        fn runSumcheck(
            allocator: std.mem.Allocator,
            f_evals: []const F,
            eq_evals_in: []F,
            num_vars: usize,
            transcript: *Transcript(F),
        ) !ProverState {
            const rounds = try allocator.alloc(ProductSC.RoundPoly, num_vars);
            const commitments = try allocator.alloc(Commitment, num_vars);
            const layers = try allocator.alloc([]F, num_vars);

            // Copy initial f_evals
            var current_f = try allocator.alloc(F, f_evals.len);
            @memcpy(current_f, f_evals);
            var current_eq = eq_evals_in;

            for (0..num_vars) |i| {
                // Save layer
                layers[i] = current_f;

                // Commit and absorb
                commitments[i] = Merkle.commit(current_f);
                transcript.absorbBytes(&commitments[i]);

                // Round polynomial using generic sumcheck
                rounds[i] = ProductSC.computeRound(.{ current_f, current_eq });
                transcript.absorb(rounds[i][0]);
                transcript.absorb(rounds[i][1]);
                transcript.absorb(rounds[i][2]);

                // Challenge
                const challenge = transcript.squeeze();

                // Fold f into new buffer
                const next_size = current_f.len / 2;
                const next_f = try allocator.alloc(F, next_size);
                for (0..next_size) |j| {
                    const lo = current_f[j];
                    const hi = current_f[j + next_size];
                    next_f[j] = lo.add(challenge.mul(hi.sub(lo)));
                }
                current_f = next_f;

                // Fold eq in place
                current_eq = multilinear.bind(F, current_eq, challenge);
            }

            const final_f = current_f[0];
            const final_eq = current_eq[0];
            allocator.free(current_f);

            return .{
                .layers = layers,
                .commitments = commitments,
                .rounds = rounds,
                .final_f = final_f,
                .final_eq = final_eq,
                .allocator = allocator,
            };
        }

        fn verifySumcheck(
            claimed_value: F,
            proof: *const Proof,
            transcript: *Transcript(F),
            challenges_out: []F,
        ) !F {
            var current_claim = claimed_value;

            for (proof.rounds, proof.commitments, 0..) |round, commitment, i| {
                // Absorb (same order as prover)
                transcript.absorbBytes(&commitment);
                transcript.absorb(round[0]);
                transcript.absorb(round[1]);
                transcript.absorb(round[2]);

                // Check g(0) + g(1) = current_claim
                if (!round[0].add(round[1]).eql(current_claim)) {
                    return error.RoundSumMismatch;
                }

                // Challenge
                const challenge = transcript.squeeze();
                challenges_out[i] = challenge;

                // Next claim using generic sumcheck evaluation
                current_claim = ProductSC.evalRoundPoly(round, challenge);
            }

            return current_claim;
        }

        // ============ Internal: Query Phase ============ //

        fn deriveQueryIndices(
            transcript: *Transcript(F),
            num_queries: usize,
            max_index: usize,
            allocator: std.mem.Allocator,
        ) ![]usize {
            var indices = try allocator.alloc(usize, num_queries);
            for (0..num_queries) |i| {
                const challenge = transcript.squeeze();
                indices[i] = challenge.value % max_index;
            }
            return indices;
        }

        fn generateQueryProofs(
            state: *const ProverState,
            query_indices: []const usize,
            allocator: std.mem.Allocator,
        ) ![]QueryProof {
            const num_vars = state.layers.len;
            var queries = try allocator.alloc(QueryProof, query_indices.len);

            for (query_indices, 0..) |q, qi| {
                var layer_proofs = try allocator.alloc(LayerProof, num_vars);

                for (0..num_vars) |i| {
                    const layer = state.layers[i];
                    const half = layer.len / 2;
                    const idx = q % half;

                    const left_opening = try Merkle.open(layer, idx, allocator);
                    const right_opening = try Merkle.open(layer, idx + half, allocator);

                    layer_proofs[i] = .{
                        .left_value = left_opening.value,
                        .right_value = right_opening.value,
                        .left_merkle = left_opening.proof,
                        .right_merkle = right_opening.proof,
                    };
                }

                queries[qi] = .{ .layers = layer_proofs };
            }

            return queries;
        }

        fn verifyQueries(
            proof: *const Proof,
            query_indices: []const usize,
            challenges: []const F,
        ) bool {
            const num_vars = proof.commitments.len;

            for (proof.queries, query_indices) |query, initial_q| {
                var expected_next: ?F = null;
                var current_idx = initial_q;

                for (0..num_vars) |i| {
                    const layer_proof = query.layers[i];
                    const commitment = proof.commitments[i];
                    const half = @as(usize, 1) << @intCast(num_vars - 1 - i);
                    const idx = current_idx % half;

                    // Verify Merkle openings
                    if (!Merkle.verifyOpening(commitment, idx, layer_proof.left_value, layer_proof.left_merkle)) {
                        return false;
                    }
                    if (!Merkle.verifyOpening(commitment, idx + half, layer_proof.right_value, layer_proof.right_merkle)) {
                        return false;
                    }

                    // Check folding from previous layer
                    // If current_idx >= half, the folded value is at right position, else at left
                    if (expected_next) |exp| {
                        const value_to_check = if (current_idx >= half) layer_proof.right_value else layer_proof.left_value;
                        if (!value_to_check.eql(exp)) {
                            return false;
                        }
                    }

                    // Compute expected next
                    const lo = layer_proof.left_value;
                    const hi = layer_proof.right_value;
                    expected_next = lo.add(challenges[i].mul(hi.sub(lo)));

                    // Update index for next layer
                    current_idx = idx;
                }

                // Final fold should equal final_f
                if (expected_next) |exp| {
                    if (!exp.eql(proof.final_f)) {
                        return false;
                    }
                }
            }

            return true;
        }
    };
}

// ============ Tests ============ //

test "basefold prove and verify" {
    const M31 = @import("../fields/mersenne31.zig").Mersenne31;
    const BasefoldM31 = Basefold(M31);
    const eqMod = @import("../poly/eq.zig");

    const allocator = std.testing.allocator;

    const f_evals = [_]M31{
        M31.fromU64(1), M31.fromU64(2), M31.fromU64(3), M31.fromU64(4),
        M31.fromU64(5), M31.fromU64(6), M31.fromU64(7), M31.fromU64(8),
    };

    const r = [_]M31{ M31.fromU64(10), M31.fromU64(20), M31.fromU64(30) };

    // Compute claimed value
    var eq_evals: [8]M31 = undefined;
    eqMod.eqEvals(M31, &r, &eq_evals);
    var claimed_value = M31.zero;
    for (f_evals, eq_evals) |f, e| {
        claimed_value = claimed_value.add(f.mul(e));
    }

    // Prove
    const config = BasefoldM31.Config{ .num_queries = 4 };
    const proof = try BasefoldM31.prove(allocator, &f_evals, &r, config);
    defer proof.deinit(allocator);

    // Verify
    const valid = try BasefoldM31.verify(allocator, claimed_value, &r, &proof, config);
    try std.testing.expect(valid);
}

test "basefold rejects wrong claimed value" {
    const M31 = @import("../fields/mersenne31.zig").Mersenne31;
    const BasefoldM31 = Basefold(M31);
    const eqMod = @import("../poly/eq.zig");

    const allocator = std.testing.allocator;

    const f_evals = [_]M31{
        M31.fromU64(1), M31.fromU64(2), M31.fromU64(3), M31.fromU64(4),
        M31.fromU64(5), M31.fromU64(6), M31.fromU64(7), M31.fromU64(8),
    };

    const r = [_]M31{ M31.fromU64(10), M31.fromU64(20), M31.fromU64(30) };

    // Compute correct claimed value
    var eq_evals: [8]M31 = undefined;
    eqMod.eqEvals(M31, &r, &eq_evals);
    var claimed_value = M31.zero;
    for (f_evals, eq_evals) |f, e| {
        claimed_value = claimed_value.add(f.mul(e));
    }

    // Prove with correct value
    const config = BasefoldM31.Config{ .num_queries = 4 };
    const proof = try BasefoldM31.prove(allocator, &f_evals, &r, config);
    defer proof.deinit(allocator);

    // Verify with WRONG claimed value
    const wrong_value = claimed_value.add(M31.one);
    const valid = try BasefoldM31.verify(allocator, wrong_value, &r, &proof, config);
    try std.testing.expect(!valid);
}
