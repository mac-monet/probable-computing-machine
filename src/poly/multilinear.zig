const std = @import("std");
const field = @import("../fields/field.zig");

// TODO optimizations
// - consider making num_vars a comptime field
// - unroll the loop over num_vars in evaluate
// - add simd where applicable

/// Multilinear polynomial over a field F
pub fn MultilinearPoly(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Evaluations over the boolean hypercube, length 2^num_vars
        /// Stored in lexicographic order: evals[i] = P(b₀, b₁, ..., bₙ₋₁)
        /// where i = b₀·2^(n-1) + b₁·2^(n-2) + ... + bₙ₋₁·2⁰
        evals: []F,

        num_vars: usize,

        // ========= Constructor ========= //

        pub fn init(allocator: std.mem.Allocator, num_vars: usize) !Self {
            const size = @as(usize, 1) << @intCast(num_vars);
            const evals = try allocator.alloc(F, size);
            return Self{
                .num_vars = num_vars,
                .evals = evals,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.evals);
        }

        // ========= Evaluation ========= //

        /// Evaluate at an arbitrary given point in F^num_vars
        pub fn evaluate(self: *const Self, point: []const F) F {
            std.debug.assert(point.len == self.num_vars);

            var result = F.zero;

            // For each hypercube vertex, compute its Lagrange basis polynomial
            for (self.evals, 0..) |eval, i| {
                var term = eval;

                // Precompute 1-x[j] for all j
                // TODO when num_vars is comptime, can use stack array
                // var one_minus = try std.BoundedArray(F, 32).init(0); // 32 is reasonable upper bound
                // for (point) |x| {
                //     one_minus.appendAssumeCapacity(F.one.sub(x));
                // }

                // Lagrange basis: product of x_j if bit_j=1, else 1-x_j)
                for (0..self.num_vars) |j| {
                    const bit = (i >> @intCast(self.num_vars - 1 - j) & 1);

                    // Multiply by the appropriate factor:
                    // if bit_j=1, x_j, else 1-x_j
                    const factor = if (bit == 1) point[j] else F.one.sub(point[j]); // TODO use @select with simd to make branchless

                    term = term.mul(factor);
                }

                result = result.add(term);
            }

            return result;
        }

        // ========= Sumcheck Operations ========= //

        /// Compute sum over the boolean hypercube
        /// This is what sumcheck protocols prove: sum_{x in {0,1}^n} f(x)
        pub fn sum(self: *const Self) F {
            var total = F.zero;
            for (self.evals) |eval| {
                total = total.add(eval);
            }
            return total;
        }

        /// Compute sum over each half of the hypercube
        /// Returns: [sum where x_0=0, sum where x_0=1]
        pub fn sumHalves(self: *const Self) [2]F {
            const half_size = self.evals.len / 2;

            var sum0 = F.zero;
            var sum1 = F.zero;

            for (0..half_size) |i| {
                sum0 = sum0.add(self.evals[i]);
                sum1 = sum1.add(self.evals[half_size + i]);
            }

            return .{ sum0, sum1 };
        }

        /// Bind the first variable to value r
        /// Produces a polynomial in (num_vars - 1) variables
        pub fn bind(self: *Self, r: F, dst: []F) void {
            const half_size = self.evals.len / 2;
            std.debug.assert(dst.len == half_size);

            for (0..half_size) |i| {
                const eval_0 = self.evals[i];
                const eval_1 = self.evals[half_size + i];

                // Linear interpolation
                const diff = eval_1.sub(eval_0);
                const scaled = r.mul(diff);
                dst[i] = eval_0.add(scaled);
            }
        }
    };
}

// For testing
const M31 = @import("../fields/mersenne31.zig").Mersenne31;

test "multilinear polynomial: sum over hypercube" {
    const allocator = std.testing.allocator;

    var poly = try MultilinearPoly(M31).init(allocator, 2);
    defer poly.deinit(allocator);

    // f(0,0) = 1, f(0,1) = 4, f(1,0) = 3, f(1,1) = 10
    poly.evals[0] = M31.fromU64(1);
    poly.evals[1] = M31.fromU64(4);
    poly.evals[2] = M31.fromU64(3);
    poly.evals[3] = M31.fromU64(10);

    const total = poly.sum();

    // sum = 1 + 4 + 3 + 10 = 18
    try std.testing.expect(M31.eql(total, M31.fromU64(18)));
}

test "multilinear polynomial: sum halves" {
    const allocator = std.testing.allocator;

    var poly = try MultilinearPoly(M31).init(allocator, 2);
    defer poly.deinit(allocator);

    poly.evals[0] = M31.fromU64(1); // f(0,0)
    poly.evals[1] = M31.fromU64(4); // f(0,1)
    poly.evals[2] = M31.fromU64(3); // f(1,0)
    poly.evals[3] = M31.fromU64(10); // f(1,1)

    const halves = poly.sumHalves();

    // halves[0] = f(0,0) + f(0,1) = 1 + 4 = 5
    // halves[1] = f(1,0) + f(1,1) = 3 + 10 = 13
    // try std.testing.expect(M31.eql(halves[0], M31.fromU64(5)));
    try std.testing.expect(halves[0].eql(M31.fromU64(5)));
    try std.testing.expect(halves[1].eql(M31.fromU64(13)));

    // Sanity check: halves should sum to total
    const total = halves[0].add(halves[1]);
    try std.testing.expect(total.eql(poly.sum()));
}

test "multilinear polynomial: bind first variable" {
    const allocator = std.testing.allocator;

    var poly = try MultilinearPoly(M31).init(allocator, 2);
    defer poly.deinit(allocator);

    // f(x₀, x₁) = 1 + 2x₀ + 3x₁ + 4x₀x₁
    poly.evals[0] = M31.fromU64(1); // f(0,0) = 1
    poly.evals[1] = M31.fromU64(4); // f(0,1) = 1 + 3 = 4
    poly.evals[2] = M31.fromU64(3); // f(1,0) = 1 + 2 = 3
    poly.evals[3] = M31.fromU64(10); // f(1,1) = 1 + 2 + 3 + 4 = 10

    const bound = try allocator.alloc(M31, 2);
    defer allocator.free(bound);

    // Bind x₀ = 5
    poly.bind(M31.fromU64(5), bound);

    // g(x₁) = f(5, x₁) = 1 + 2*5 + 3x₁ + 4*5*x₁ = 11 + 23x₁
    // g(0) = 11, g(1) = 34
    try std.testing.expect(M31.eql(bound[0], M31.fromU64(11)));
    try std.testing.expect(M31.eql(bound[1], M31.fromU64(34)));
}

test "multilinear polynomial: evaluate at arbitrary point" {
    const allocator = std.testing.allocator;

    var poly = try MultilinearPoly(M31).init(allocator, 2);
    defer poly.deinit(allocator);

    // f(x₀, x₁) = 1 + 2x₀ + 3x₁ + 4x₀x₁
    poly.evals[0] = M31.fromU64(1); // f(0,0) = 1
    poly.evals[1] = M31.fromU64(4); // f(0,1) = 4
    poly.evals[2] = M31.fromU64(3); // f(1,0) = 3
    poly.evals[3] = M31.fromU64(10); // f(1,1) = 10

    // Evaluate at (2, 3)
    const point = [_]M31{ M31.fromU64(2), M31.fromU64(3) };
    const result = poly.evaluate(&point);

    // f(2,3) = 1 + 2*2 + 3*3 + 4*2*3 = 1 + 4 + 9 + 24 = 38
    try std.testing.expect(result.eql(M31.fromU64(38)));
}

test "multilinear polynomial: evaluate at hypercube vertices" {
    const allocator = std.testing.allocator;

    var poly = try MultilinearPoly(M31).init(allocator, 2);
    defer poly.deinit(allocator);

    poly.evals[0] = M31.fromU64(1);
    poly.evals[1] = M31.fromU64(4);
    poly.evals[2] = M31.fromU64(3);
    poly.evals[3] = M31.fromU64(10);

    // Evaluating at hypercube vertices should match stored values
    const p00 = [_]M31{ M31.zero, M31.zero };
    const p01 = [_]M31{ M31.zero, M31.one };
    const p10 = [_]M31{ M31.one, M31.zero };
    const p11 = [_]M31{ M31.one, M31.one };

    try std.testing.expect(M31.eql(poly.evaluate(&p00), M31.fromU64(1)));
    try std.testing.expect(M31.eql(poly.evaluate(&p01), M31.fromU64(4)));
    try std.testing.expect(M31.eql(poly.evaluate(&p10), M31.fromU64(3)));
    try std.testing.expect(M31.eql(poly.evaluate(&p11), M31.fromU64(10)));
}

test "multilinear polynomial: bind consistency with evaluate" {
    const allocator = std.testing.allocator;

    var poly = try MultilinearPoly(M31).init(allocator, 2);
    defer poly.deinit(allocator);

    poly.evals[0] = M31.fromU64(1);
    poly.evals[1] = M31.fromU64(4);
    poly.evals[2] = M31.fromU64(3);
    poly.evals[3] = M31.fromU64(10);

    // Bind x₀ = 5
    const bound = try allocator.alloc(M31, 2);
    defer allocator.free(bound);
    poly.bind(M31.fromU64(5), bound);

    // Create a 1-variable polynomial from bound values
    var bound_poly = MultilinearPoly(M31){
        .num_vars = 1,
        .evals = bound,
    };

    // Evaluate bound_poly at x₁ = 7
    const point1 = [_]M31{M31.fromU64(7)};
    const result1 = bound_poly.evaluate(&point1);

    // Should match evaluating original poly at (5, 7)
    const point2 = [_]M31{ M31.fromU64(5), M31.fromU64(7) };
    const result2 = poly.evaluate(&point2);

    try std.testing.expect(M31.eql(result1, result2));
}
