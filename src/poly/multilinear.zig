const std = @import("std");

/// Bind first variable to r in-place. Returns the smaller slice (new polynomial view).
/// Input slice must have len >= 2 (at least one variable to bind).
pub fn bind(comptime F: type, evals: []F, r: F) []F {
    std.debug.assert(evals.len >= 2);
    const half = evals.len / 2;
    F.linearCombineBatch(evals[0..half], evals[0..half], evals[half..], r);
    return evals[0..half];
}

/// Sum over the boolean hypercube: sum_{x in {0,1}^n} f(x)
pub fn sum(comptime F: type, evals: []const F) F {
    return F.sumSlice(evals);
}

/// Sum each half of the hypercube.
/// Returns: [sum where x_0=0, sum where x_0=1]
pub fn sumHalves(comptime F: type, evals: []const F) [2]F {
    std.debug.assert(evals.len >= 2);
    const half = evals.len / 2;
    return F.sumSlices(2, .{ evals[0..half], evals[half..] });
}

/// Full evaluation at a point via successive binds.
/// Modifies evals in-place. Returns f(point).
pub fn evaluate(comptime F: type, evals: []F, point: []const F) F {
    std.debug.assert(evals.len == @as(usize, 1) << @intCast(point.len));
    var current = evals;
    for (point) |r| {
        current = bind(F, current, r);
    }
    return current[0];
}

// ============ Tests ============ //

const M31 = @import("../fields/mersenne31.zig").Mersenne31;

test "sum over hypercube" {
    // f(0,0) = 1, f(0,1) = 4, f(1,0) = 3, f(1,1) = 10
    var evals = [_]M31{
        M31.fromU64(1),
        M31.fromU64(4),
        M31.fromU64(3),
        M31.fromU64(10),
    };

    const total = sum(M31, &evals);

    // sum = 1 + 4 + 3 + 10 = 18
    try std.testing.expect(total.eql(M31.fromU64(18)));
}

test "sum halves" {
    var evals = [_]M31{
        M31.fromU64(1), // f(0,0)
        M31.fromU64(4), // f(0,1)
        M31.fromU64(3), // f(1,0)
        M31.fromU64(10), // f(1,1)
    };

    const halves = sumHalves(M31, &evals);

    // halves[0] = f(0,0) + f(0,1) = 1 + 4 = 5
    // halves[1] = f(1,0) + f(1,1) = 3 + 10 = 13
    try std.testing.expect(halves[0].eql(M31.fromU64(5)));
    try std.testing.expect(halves[1].eql(M31.fromU64(13)));

    // Sanity check: halves should sum to total
    const total = halves[0].add(halves[1]);
    try std.testing.expect(total.eql(sum(M31, &evals)));
}

test "bind first variable" {
    // f(x₀, x₁) = 1 + 2x₀ + 3x₁ + 4x₀x₁
    var evals = [_]M31{
        M31.fromU64(1), // f(0,0) = 1
        M31.fromU64(4), // f(0,1) = 1 + 3 = 4
        M31.fromU64(3), // f(1,0) = 1 + 2 = 3
        M31.fromU64(10), // f(1,1) = 1 + 2 + 3 + 4 = 10
    };

    // Bind x₀ = 5
    const bound = bind(M31, &evals, M31.fromU64(5));

    // g(x₁) = f(5, x₁) = 1 + 2*5 + 3x₁ + 4*5*x₁ = 11 + 23x₁
    // g(0) = 11, g(1) = 34
    try std.testing.expectEqual(@as(usize, 2), bound.len);
    try std.testing.expect(bound[0].eql(M31.fromU64(11)));
    try std.testing.expect(bound[1].eql(M31.fromU64(34)));
}

test "full evaluation via successive binds" {
    // f(x₀, x₁) = 1 + 2x₀ + 3x₁ + 4x₀x₁
    var evals = [_]M31{
        M31.fromU64(1), // f(0,0) = 1
        M31.fromU64(4), // f(0,1) = 4
        M31.fromU64(3), // f(1,0) = 3
        M31.fromU64(10), // f(1,1) = 10
    };

    const point = [_]M31{ M31.fromU64(2), M31.fromU64(3) };
    const result = evaluate(M31, &evals, &point);

    // f(2,3) = 1 + 2*2 + 3*3 + 4*2*3 = 1 + 4 + 9 + 24 = 38
    try std.testing.expect(result.eql(M31.fromU64(38)));
}

test "bind returns progressively smaller slices" {
    var evals: [8]M31 = undefined;
    for (&evals, 0..) |*e, i| {
        e.* = M31.fromU64(i);
    }

    var view: []M31 = &evals;
    try std.testing.expectEqual(@as(usize, 8), view.len);

    view = bind(M31, view, M31.fromU64(1));
    try std.testing.expectEqual(@as(usize, 4), view.len);

    view = bind(M31, view, M31.fromU64(2));
    try std.testing.expectEqual(@as(usize, 2), view.len);

    view = bind(M31, view, M31.fromU64(3));
    try std.testing.expectEqual(@as(usize, 1), view.len);
}
