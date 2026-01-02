const std = @import("std");

/// Compute eq(b, r) where b is boolean (index) and r is field elements
/// b is represented as an integer index into the hypercube
pub fn eqEval(comptime F: type, b_index: usize, r: []const F) F {
    var result = F.one;
    var idx = b_index;

    // Iterate r in reverse: MSB of index corresponds to r[0]
    var i = r.len;
    while (i > 0) {
        i -= 1;
        const b_i = idx & 1;
        idx >>= 1;

        const term = if (b_i == 1) r[i] else F.one.sub(r[i]);
        result = result.mul(term);
    }

    return result;
}

/// Build eq(b , r) for all b in {0,1}^n
pub fn eqEvals(comptime F: type, r: []const F, dst: []F) void {
    std.debug.assert(dst.len == @as(usize, 1) << @intCast(r.len));

    // Start with eq = 1 for the empty product
    dst[0] = F.one;
    var size: usize = 1;

    // For each var, double the table
    for (r) |r_i| {
        // Process in reverse to avoid overwriting
        var j = size;
        while (j > 0) {
            j -= 1;
            const prev = dst[j];
            dst[2 * j] = prev.mul(F.one.sub(r_i));
            dst[2 * j + 1] = prev.mul(r_i);
        }
        size *= 2;
    }
}

// Testing

const M31 = @import("fields/mersenne31.zig").Mersenne31;
const multilinear = @import("poly/multilinear.zig");

test "eq at boolean points" {
    // eq(b, b) = 1 for boolean b
    const r_00 = [_]M31{ M31.zero, M31.zero };
    try std.testing.expect(eqEval(M31, 0b00, &r_00).eql(M31.one));

    const r_01 = [_]M31{ M31.zero, M31.one };
    try std.testing.expect(eqEval(M31, 0b01, &r_01).eql(M31.one));

    const r_10 = [_]M31{ M31.one, M31.zero };
    try std.testing.expect(eqEval(M31, 0b10, &r_10).eql(M31.one));

    const r_11 = [_]M31{ M31.one, M31.one };
    try std.testing.expect(eqEval(M31, 0b11, &r_11).eql(M31.one));

    // eq(b, b') = 0 for different boolean points
    try std.testing.expect(eqEval(M31, 0b00, &r_11).eql(M31.zero));
    try std.testing.expect(eqEval(M31, 0b01, &r_10).eql(M31.zero));
}

test "eq sums to one" {
    // Σ_{b ∈ {0,1}^n} eq(b, r) = 1 for any r
    const r = [_]M31{ M31.fromU64(7), M31.fromU64(13) };
    const n: usize = 2;
    const size = @as(usize, 1) << n;

    var sum = M31.zero;
    for (0..size) |b| {
        sum = sum.add(eqEval(M31, b, &r));
    }

    try std.testing.expect(sum.eql(M31.one));
}

test "eq multilinear identity" {
    // f(r) = Σ_b f(b) · eq(b, r)
    const evals = [_]M31{
        M31.fromU64(1), // f(0,0)
        M31.fromU64(4), // f(0,1)
        M31.fromU64(3), // f(1,0)
        M31.fromU64(10), // f(1,1)
    };

    const r = [_]M31{ M31.fromU64(5), M31.fromU64(9) };

    // Compute f(r) via summation with eq
    var sum = M31.zero;
    for (0..4) |b| {
        sum = sum.add(evals[b].mul(eqEval(M31, b, &r)));
    }

    // Compute f(r) via multilinear.evaluate
    var evals_copy = evals;
    const direct = multilinear.evaluate(M31, &evals_copy, &r);

    try std.testing.expect(sum.eql(direct));
}

test "eqEvals matches eqEval" {
    const r = [_]M31{ M31.fromU64(5), M31.fromU64(9), M31.fromU64(13) };
    const n = r.len;
    const size = @as(usize, 1) << n;

    // Build table
    var table: [size]M31 = undefined;
    eqEvals(M31, &r, &table);

    // Check each entry against point evaluation
    for (0..size) |b| {
        const expected = eqEval(M31, b, &r);
        try std.testing.expect(table[b].eql(expected));
    }
}

test "eqEvals sums to one" {
    const r = [_]M31{ M31.fromU64(7), M31.fromU64(11) };
    const size = @as(usize, 1) << r.len;

    var table: [size]M31 = undefined;
    eqEvals(M31, &r, &table);

    const sum = multilinear.sum(M31, &table);
    try std.testing.expect(sum.eql(M31.one));
}
