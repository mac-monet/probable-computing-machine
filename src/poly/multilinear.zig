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

        // ========= Sumcheck Operations ========= //

        /// Compute sum over the boolean hypercube
        /// This is what sumcheck protocols prove: sum_{x in {0,1}^n} f(x)
        /// Uses delayed reduction for performance.
        pub fn sum(self: *const Self) F {
            return F.sumSlice(self.evals[0..self.len()]);
        }

        /// Compute sum over each half of the hypercube
        /// Returns: [sum where x_0=0, sum where x_0=1]
        /// Uses delayed reduction for performance.
        pub fn sumHalves(self: *const Self) [2]F {
            const half_size = self.len() / 2;
            return .{
                F.sumSlice(self.evals[0..half_size]),
                F.sumSlice(self.evals[half_size .. half_size * 2]),
            };
        }

        /// Bind the first variable to value r in-place.
        /// Decreases num_vars by 1. Logical size is always 1 << num_vars.
        /// The evals slice retains original allocation for proper cleanup.
        pub fn bind(self: *Self, r: F) void {
            const half = self.len() / 2;
            F.linearCombineBatch(self.evals[0..half], self.evals[0..half], self.evals[half .. half * 2], r);
            self.num_vars -= 1;
        }

        /// Returns the logical size of the polynomial (1 << num_vars).
        pub fn len(self: *const Self) usize {
            return @as(usize, 1) << @intCast(self.num_vars);
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

    // Bind x₀ = 5 in-place
    poly.bind(M31.fromU64(5));

    // g(x₁) = f(5, x₁) = 1 + 2*5 + 3x₁ + 4*5*x₁ = 11 + 23x₁
    // g(0) = 11, g(1) = 34
    try std.testing.expectEqual(@as(usize, 1), poly.num_vars);
    try std.testing.expectEqual(@as(usize, 2), poly.len());
    try std.testing.expect(M31.eql(poly.evals[0], M31.fromU64(11)));
    try std.testing.expect(M31.eql(poly.evals[1], M31.fromU64(34)));
}

test "multilinear polynomial: full evaluation via successive binds" {
    const allocator = std.testing.allocator;

    var poly = try MultilinearPoly(M31).init(allocator, 2);
    defer poly.deinit(allocator);

    // f(x₀, x₁) = 1 + 2x₀ + 3x₁ + 4x₀x₁
    poly.evals[0] = M31.fromU64(1); // f(0,0) = 1
    poly.evals[1] = M31.fromU64(4); // f(0,1) = 4
    poly.evals[2] = M31.fromU64(3); // f(1,0) = 3
    poly.evals[3] = M31.fromU64(10); // f(1,1) = 10

    // Evaluate at (2, 3) via successive binds
    poly.bind(M31.fromU64(2));
    poly.bind(M31.fromU64(3));

    // f(2,3) = 1 + 2*2 + 3*3 + 4*2*3 = 1 + 4 + 9 + 24 = 38
    try std.testing.expect(poly.evals[0].eql(M31.fromU64(38)));
}
