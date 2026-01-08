const std = @import("std");
const merkle = @import("../merkle/root.zig");
const Blake3Hasher = merkle.Blake3Hasher;
const Transcript = @import("../core/transcript.zig").Transcript;
const multilinear = @import("../poly/multilinear.zig");
const eq = @import("../poly/eq.zig");

pub const Config = struct {
    /// Maximum number of variables (log2 of max polynomial size)
    /// Determines proof buffer sizes. Default 20 = 1M evaluations.
    max_vars: comptime_int = 20,
    /// Number of queries for soundness amplification
    num_queries: comptime_int = 32,
};

/// Basefold polynomial commitment scheme.
///
/// Comptime-parameterized by field type and configuration.
/// Uses bounded proof structures (no heap allocation in proof).
/// Scratch space allocated from caller-provided allocator (arena recommended).
pub fn Basefold(comptime F: type, comptime config: Config) type {
    const Merkle = merkle.MerkleTree(F, Blake3Hasher, config.max_vars);
    const max_vars = config.max_vars;
    const num_queries = config.num_queries;

    return struct {
        const Self = @This();

        pub const Commitment = Merkle.Commitment;
        pub const Digest = Merkle.Digest;

        /// Round polynomial: evaluations at 0, 1, 2 (degree-2 for product sumcheck)
        pub const RoundPoly = [3]F;

        /// Bounded layer proof - no allocation
        pub const LayerProof = struct {
            left_value: F,
            right_value: F,
            left_merkle: Merkle.Proof,
            right_merkle: Merkle.Proof,
        };

        /// Bounded query proof - fixed array of layer proofs
        pub const QueryProof = struct {
            layers: [max_vars]LayerProof,
            num_layers: usize,

            pub fn getLayer(self: *const QueryProof, i: usize) LayerProof {
                std.debug.assert(i < self.num_layers);
                return self.layers[i];
            }
        };

        /// Fully bounded opening proof - no heap allocation
        pub const OpeningProof = struct {
            rounds: [max_vars]RoundPoly,
            commitments: [max_vars]Commitment,
            num_vars: usize,
            final_f: F,
            final_eq: F,
            queries: [num_queries]QueryProof,

            pub fn getRound(self: *const OpeningProof, i: usize) RoundPoly {
                std.debug.assert(i < self.num_vars);
                return self.rounds[i];
            }

            pub fn getCommitment(self: *const OpeningProof, i: usize) Commitment {
                std.debug.assert(i < self.num_vars);
                return self.commitments[i];
            }
        };

        // ============ Public API ============ //

        /// Commit to polynomial evaluations.
        /// Uses page_allocator for scratch space internally.
        pub fn commit(evals: []const F) Commitment {
            const n = evals.len;
            if (n == 0) return std.mem.zeroes(Commitment);

            var hasher = Blake3Hasher.init();
            const scratch = std.heap.page_allocator.alloc(Digest, n) catch unreachable;
            defer std.heap.page_allocator.free(scratch);

            return Merkle.commit(evals, scratch, &hasher);
        }

        /// Prove that f(r) = claimed_value.
        ///
        /// Allocator used for scratch space only - proof is fully bounded.
        /// Recommended: pass arena.allocator() for efficient batch allocation.
        pub fn prove(
            allocator: std.mem.Allocator,
            f_evals: []const F,
            r: []const F,
        ) !OpeningProof {
            const num_vars = r.len;
            const n = f_evals.len;
            std.debug.assert(n == @as(usize, 1) << @intCast(num_vars));
            std.debug.assert(num_vars <= max_vars);

            var hasher = Blake3Hasher.init();
            var transcript = Transcript(F).init("basefold");

            // Allocate scratch buffers from arena
            const commit_scratch = try allocator.alloc(Digest, n);
            const tree_scratch = try allocator.alloc(Digest, 2 * n - 1);

            // Store layers for query proof generation
            // layers[i] has size n / 2^i
            const layers = try allocator.alloc([]F, num_vars);
            for (0..num_vars) |i| {
                const layer_size = n >> @intCast(i);
                layers[i] = try allocator.alloc(F, layer_size);
            }

            // Build eq table
            const eq_evals = try allocator.alloc(F, n);
            eq.eqEvals(F, r, eq_evals);

            // Copy initial evaluations to layer 0
            @memcpy(layers[0], f_evals);
            var current_eq = eq_evals;

            var proof: OpeningProof = undefined;
            proof.num_vars = num_vars;

            // Phase 1: Sumcheck rounds with commitments
            for (0..num_vars) |i| {
                const current_f = layers[i];
                const layer_size = current_f.len;

                // Commit layer and absorb into transcript
                proof.commitments[i] = Merkle.commit(current_f, commit_scratch[0..layer_size], &hasher);
                transcript.absorbBytes(&proof.commitments[i]);

                // Compute round polynomial (product sumcheck)
                proof.rounds[i] = computeRoundPoly(current_f, current_eq);
                transcript.absorb(proof.rounds[i][0]);
                transcript.absorb(proof.rounds[i][1]);
                transcript.absorb(proof.rounds[i][2]);

                // Squeeze challenge
                const challenge = transcript.squeeze();

                // Fold f into next layer (or compute final_f on last round)
                if (i + 1 < num_vars) {
                    fold(current_f, layers[i + 1], challenge);
                } else {
                    // Last round: layer has size 2, fold to single value
                    std.debug.assert(current_f.len == 2);
                    proof.final_f = current_f[0].add(challenge.mul(current_f[1].sub(current_f[0])));
                }

                // Fold eq in place
                current_eq = multilinear.bind(F, current_eq, challenge);
            }

            proof.final_eq = current_eq[0];

            // Derive query indices from transcript
            var query_indices: [num_queries]usize = undefined;
            const max_index = n >> 1;
            deriveQueryIndices(&transcript, &query_indices, max_index);

            // Phase 2: Generate query proofs
            generateQueryProofs(layers, num_vars, &proof.queries, &query_indices, tree_scratch, &hasher);

            return proof;
        }

        /// Verify a basefold opening proof.
        ///
        /// Allocator used for small scratch buffer only.
        pub fn verify(
            allocator: std.mem.Allocator,
            claimed_value: F,
            r: []const F,
            proof: *const OpeningProof,
        ) !bool {
            const num_vars = proof.num_vars;
            if (r.len != num_vars) return false;

            var hasher = Blake3Hasher.init();
            var transcript = Transcript(F).init("basefold");

            // Allocate challenges buffer
            const challenges = try allocator.alloc(F, num_vars);
            defer allocator.free(challenges);

            // Phase 1: Verify sumcheck rounds
            const final_claim = verifySumcheck(claimed_value, proof, &transcript, challenges) catch return false;

            // Check final_eq matches eq(challenges, r)
            const eq_at_c = eq.eqEvalField(F, challenges, r);
            if (!proof.final_eq.eql(eq_at_c)) return false;

            // Check final_claim = final_f * final_eq
            if (!final_claim.eql(proof.final_f.mul(proof.final_eq))) return false;

            // Derive query indices (must match prover)
            var query_indices: [num_queries]usize = undefined;
            const n = @as(usize, 1) << @intCast(num_vars);
            const max_index = n >> 1;
            deriveQueryIndices(&transcript, &query_indices, max_index);

            // Phase 2: Verify query proofs
            return verifyQueries(proof, &query_indices, challenges, &hasher);
        }

        // ============ Internal: Round Polynomial ============ //

        /// Compute round polynomial for product sumcheck.
        /// Returns evaluations at points 0, 1, 2.
        fn computeRoundPoly(f: []const F, eq_vals: []const F) RoundPoly {
            const half = f.len / 2;
            var result: RoundPoly = undefined;

            // Evaluate at 0, 1, 2
            inline for (0..3) |point| {
                var sum = F.zero;
                for (0..half) |i| {
                    const f_lo = f[i];
                    const f_hi = f[i + half];
                    const eq_lo = eq_vals[i];
                    const eq_hi = eq_vals[i + half];

                    // Interpolate f and eq at point, multiply
                    const f_interp = interpolateAt(f_lo, f_hi, point);
                    const eq_interp = interpolateAt(eq_lo, eq_hi, point);
                    sum = sum.add(f_interp.mul(eq_interp));
                }
                result[point] = sum;
            }

            return result;
        }

        /// Linear interpolation: lo + point * (hi - lo)
        fn interpolateAt(lo: F, hi: F, point: usize) F {
            if (point == 0) return lo;
            if (point == 1) return hi;
            const delta = hi.sub(lo);
            return lo.add(delta.mul(F.fromU64(point)));
        }

        /// Evaluate degree-2 round polynomial at challenge using Lagrange interpolation.
        fn evalRoundPoly(poly: RoundPoly, x: F) F {
            const one = F.one;
            const two = F.fromU64(2);
            const two_inv = two.inv();

            // L_0(x) = (x-1)(x-2) / 2
            const l0 = x.sub(one).mul(x.sub(two)).mul(two_inv);
            // L_1(x) = x(2-x)
            const l1 = x.mul(two.sub(x));
            // L_2(x) = x(x-1) / 2
            const l2 = x.mul(x.sub(one)).mul(two_inv);

            return poly[0].mul(l0).add(poly[1].mul(l1)).add(poly[2].mul(l2));
        }

        // ============ Internal: Folding ============ //

        /// Fold polynomial: dst[i] = src[i] + challenge * (src[i + half] - src[i])
        fn fold(src: []const F, dst: []F, challenge: F) void {
            const half = src.len / 2;
            std.debug.assert(dst.len >= half);
            for (0..half) |i| {
                const lo = src[i];
                const hi = src[i + half];
                dst[i] = lo.add(challenge.mul(hi.sub(lo)));
            }
        }

        // ============ Internal: Sumcheck Verification ============ //

        fn verifySumcheck(
            claimed_value: F,
            proof: *const OpeningProof,
            transcript: *Transcript(F),
            challenges_out: []F,
        ) !F {
            var current_claim = claimed_value;

            for (0..proof.num_vars) |i| {
                const round = proof.rounds[i];
                const commitment = proof.commitments[i];

                // Absorb (same order as prover)
                transcript.absorbBytes(&commitment);
                transcript.absorb(round[0]);
                transcript.absorb(round[1]);
                transcript.absorb(round[2]);

                // Check g(0) + g(1) = current_claim
                if (!round[0].add(round[1]).eql(current_claim)) {
                    return error.RoundSumMismatch;
                }

                // Squeeze challenge
                const challenge = transcript.squeeze();
                challenges_out[i] = challenge;

                // Next claim = g(challenge)
                current_claim = evalRoundPoly(round, challenge);
            }

            return current_claim;
        }

        // ============ Internal: Query Phase ============ //

        fn deriveQueryIndices(
            transcript: *Transcript(F),
            indices_out: *[num_queries]usize,
            max_index: usize,
        ) void {
            for (indices_out) |*idx| {
                const challenge = transcript.squeeze();
                idx.* = challenge.value % max_index;
            }
        }

        fn generateQueryProofs(
            layers: [][]F,
            num_vars: usize,
            queries_out: *[num_queries]QueryProof,
            query_indices: *const [num_queries]usize,
            tree_scratch: []Digest,
            hasher: *Blake3Hasher,
        ) void {
            for (query_indices, 0..) |q, qi| {
                queries_out[qi].num_layers = num_vars;

                for (0..num_vars) |i| {
                    const layer = layers[i];
                    const half = layer.len / 2;
                    const idx = q % half;
                    const tree_size = 2 * layer.len - 1;

                    queries_out[qi].layers[i] = .{
                        .left_value = layer[idx],
                        .right_value = layer[idx + half],
                        .left_merkle = Merkle.open(layer, @intCast(idx), tree_scratch[0..tree_size], hasher),
                        .right_merkle = Merkle.open(layer, @intCast(idx + half), tree_scratch[0..tree_size], hasher),
                    };
                }
            }
        }

        fn verifyQueries(
            proof: *const OpeningProof,
            query_indices: *const [num_queries]usize,
            challenges: []const F,
            hasher: *Blake3Hasher,
        ) bool {
            const num_vars = proof.num_vars;

            for (proof.queries, query_indices.*) |query, initial_q| {
                var expected_next: ?F = null;
                var current_idx = initial_q;

                for (0..num_vars) |i| {
                    const layer_proof = query.layers[i];
                    const commitment = proof.commitments[i];
                    const layer_size = @as(usize, 1) << @intCast(num_vars - i);
                    const half = layer_size / 2;
                    const idx = current_idx % half;

                    // Verify Merkle openings
                    if (!Merkle.verify(commitment, layer_proof.left_value, layer_proof.left_merkle, hasher)) {
                        return false;
                    }
                    if (!Merkle.verify(commitment, layer_proof.right_value, layer_proof.right_merkle, hasher)) {
                        return false;
                    }

                    // Check folding consistency from previous layer
                    if (expected_next) |exp| {
                        // If current_idx >= half, folded value is at right position
                        const value_to_check = if (current_idx >= half)
                            layer_proof.right_value
                        else
                            layer_proof.left_value;
                        if (!value_to_check.eql(exp)) {
                            return false;
                        }
                    }

                    // Compute expected next from folding
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

const M31 = @import("../fields/mersenne31.zig").Mersenne31;

// Test config: small max_vars for fast tests, few queries
const TestConfig = Config{ .max_vars = 8, .num_queries = 4 };

test "basefold prove and verify" {
    const BasefoldM31 = Basefold(M31, TestConfig);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const f_evals = [_]M31{
        M31.fromU64(1), M31.fromU64(2), M31.fromU64(3), M31.fromU64(4),
        M31.fromU64(5), M31.fromU64(6), M31.fromU64(7), M31.fromU64(8),
    };

    const r = [_]M31{ M31.fromU64(10), M31.fromU64(20), M31.fromU64(30) };

    // Compute claimed value: sum_b f(b) * eq(b, r)
    var eq_evals: [8]M31 = undefined;
    eq.eqEvals(M31, &r, &eq_evals);
    var claimed_value = M31.zero;
    for (f_evals, eq_evals) |f, e| {
        claimed_value = claimed_value.add(f.mul(e));
    }

    // Prove
    const proof = try BasefoldM31.prove(allocator, &f_evals, &r);

    // Verify with fresh arena
    var verify_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer verify_arena.deinit();
    const valid = try BasefoldM31.verify(verify_arena.allocator(), claimed_value, &r, &proof);
    try std.testing.expect(valid);
}

test "basefold rejects wrong claimed value" {
    const BasefoldM31 = Basefold(M31, TestConfig);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const f_evals = [_]M31{
        M31.fromU64(1), M31.fromU64(2), M31.fromU64(3), M31.fromU64(4),
        M31.fromU64(5), M31.fromU64(6), M31.fromU64(7), M31.fromU64(8),
    };

    const r = [_]M31{ M31.fromU64(10), M31.fromU64(20), M31.fromU64(30) };

    // Compute correct claimed value
    var eq_evals: [8]M31 = undefined;
    eq.eqEvals(M31, &r, &eq_evals);
    var claimed_value = M31.zero;
    for (f_evals, eq_evals) |f, e| {
        claimed_value = claimed_value.add(f.mul(e));
    }

    // Prove with correct value
    const proof = try BasefoldM31.prove(allocator, &f_evals, &r);

    // Verify with WRONG claimed value
    var verify_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer verify_arena.deinit();
    const wrong_value = claimed_value.add(M31.one);
    const valid = try BasefoldM31.verify(verify_arena.allocator(), wrong_value, &r, &proof);
    try std.testing.expect(!valid);
}

test "basefold commit matches prove commitment" {
    const BasefoldM31 = Basefold(M31, TestConfig);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const f_evals = [_]M31{
        M31.fromU64(1), M31.fromU64(2), M31.fromU64(3), M31.fromU64(4),
    };

    const r = [_]M31{ M31.fromU64(5), M31.fromU64(7) };

    // Standalone commit
    const standalone_commitment = BasefoldM31.commit(&f_evals);

    // Prove and check first commitment matches
    const proof = try BasefoldM31.prove(arena.allocator(), &f_evals, &r);

    try std.testing.expectEqualSlices(u8, &standalone_commitment, &proof.commitments[0]);
}

test "basefold with larger polynomial" {
    const BasefoldM31 = Basefold(M31, .{ .max_vars = 8, .num_queries = 8 });

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // 2^5 = 32 evaluations
    var f_evals: [32]M31 = undefined;
    for (&f_evals, 0..) |*e, i| {
        e.* = M31.fromU64(i + 1);
    }

    const r = [_]M31{
        M31.fromU64(3),
        M31.fromU64(7),
        M31.fromU64(11),
        M31.fromU64(13),
        M31.fromU64(17),
    };

    // Compute claimed value
    var eq_evals: [32]M31 = undefined;
    eq.eqEvals(M31, &r, &eq_evals);
    var claimed_value = M31.zero;
    for (f_evals, eq_evals) |f, e| {
        claimed_value = claimed_value.add(f.mul(e));
    }

    const proof = try BasefoldM31.prove(allocator, &f_evals, &r);

    var verify_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer verify_arena.deinit();
    const valid = try BasefoldM31.verify(verify_arena.allocator(), claimed_value, &r, &proof);
    try std.testing.expect(valid);
}
