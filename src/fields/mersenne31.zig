const std = @import("std");
const field = @import("field.zig");

/// Mersenne-31: p = 2^31 - 1
pub const Mersenne31 = struct {
    value: u32,

    // p = 2^31 - 1
    pub const MODULUS: u32 = 0x7FFFFFFF;
    pub const ENCODED_SIZE: usize = 4;

    pub const zero: Mersenne31 = .{ .value = 0 };
    pub const one: Mersenne31 = .{ .value = 1 };

    // ============ Core Arithmetic ============ //

    pub fn add(a: Mersenne31, b: Mersenne31) Mersenne31 {
        // a + b < 2^32, fits in u32 with potential wrap
        var sum: u32 = a.value +% b.value;

        // Mersenne reduction: x mod (2^31 - 1) = (x & mask) + (x >> 31)
        // Because 2^31 ≡ 1 (mod p)
        sum = (sum & MODULUS) +% (sum >> 31);

        // At most one more reduction
        if (sum >= MODULUS) sum -%= MODULUS;

        return .{ .value = sum };
    }

    pub fn sub(a: Mersenne31, b: Mersenne31) Mersenne31 {
        const diff = a.value +% (MODULUS - b.value);
        // diff is in [0, 2*MODULUS), reduce once
        return .{ .value = if (diff >= MODULUS) diff - MODULUS else diff };
    }

    pub fn mul(a: Mersenne31, b: Mersenne31) Mersenne31 {
        // Product fits in u64
        const wide: u64 = @as(u64, a.value) * @as(u64, b.value);
        return reduce64(wide);
    }

    pub fn neg(a: Mersenne31) Mersenne31 {
        return if (a.value == 0) a else .{ .value = MODULUS - a.value };
    }

    // ============ Batch Arithmetic ============ //
    // TODO SIMD optimize later

    pub fn addBatch(dst: []Mersenne31, a: []const Mersenne31, b: []const Mersenne31) void {
        std.debug.assert(dst.len == a.len and a.len == b.len);
        for (dst, a, b) |*d, aa, bb| {
            d.* = add(aa, bb);
        }
    }

    pub fn mulBatch(dst: []Mersenne31, a: []const Mersenne31, b: []const Mersenne31) void {
        std.debug.assert(dst.len == a.len and a.len == b.len);
        for (dst, a, b) |*d, aa, bb| {
            d.* = mul(aa, bb);
        }
    }

    pub fn subBatch(dst: []Mersenne31, a: []const Mersenne31, b: []const Mersenne31) void {
        std.debug.assert(dst.len == a.len and a.len == b.len);
        for (dst, a, b) |*d, aa, bb| {
            d.* = sub(aa, bb);
        }
    }

    // Fused multiply-add: dst[i] = a[i] * b[i] + c[i]
    pub fn mulAddBatch(dst: []Mersenne31, a: []const Mersenne31, b: []const Mersenne31, c: []const Mersenne31) void {
        std.debug.assert(dst.len == a.len and a.len == b.len and b.len == c.len);
        for (dst, a, b, c) |*d, aa, bb, cc| {
            d.* = mul(aa, bb).add(cc);
        }
    }

    // ============ Batch Reductions (Delayed Reduction) ============ //
    // These operations accumulate in u64 and reduce once at the end,
    // avoiding per-element reduction overhead.

    /// Sum all elements in a slice with delayed reduction.
    /// Accumulates in u64, reduces once at the end.
    pub fn sumSlice(values: []const Mersenne31) Mersenne31 {
        var acc: u64 = 0;
        for (values) |v| {
            acc +%= v.value;
        }
        return reduce64(acc);
    }

    /// Compute dot product: sum(a[i] * b[i]) with delayed reduction.
    /// Each product is reduced to u64, then accumulated.
    pub fn dotProduct(a: []const Mersenne31, b: []const Mersenne31) Mersenne31 {
        std.debug.assert(a.len == b.len);
        var acc: u64 = 0;
        for (a, b) |aa, bb| {
            // Product fits in u62 (31 bits * 31 bits), safe to accumulate many
            const prod: u64 = @as(u64, aa.value) * @as(u64, bb.value);
            acc +%= prod;
        }
        return reduce64(acc);
    }

    /// Linear interpolation: dst[i] = a[i] + r * (b[i] - a[i])
    /// Used for binding variables in multilinear polynomials.
    pub fn linearCombineBatch(dst: []Mersenne31, a: []const Mersenne31, b: []const Mersenne31, r: Mersenne31) void {
        std.debug.assert(dst.len == a.len and a.len == b.len);
        const r_val: u64 = r.value;

        for (dst, a, b) |*d, aa, bb| {
            // Branchless modular subtraction: diff = (b - a) mod p
            // Adding MODULUS ensures no underflow; reduce64 handles the extra MODULUS
            const diff: u64 = @as(u64, bb.value) +% MODULUS -% aa.value;
            const prod = r_val *% diff;
            const result = @as(u64, aa.value) +% prod;
            d.* = reduce64(result);
        }
    }

    // ============ Derived Arithmetic ============ //

    pub const square = field.defaults(Mersenne31).square;
    pub const double = field.defaults(Mersenne31).double;
    pub const pow = field.defaults(Mersenne31).pow;

    pub fn inv(a: Mersenne31) Mersenne31 {
        std.debug.assert(!a.isZero());
        // a^(-1) = a^(p-2) by Fermat's little theorem
        return a.pow(MODULUS - 2);
    }

    // ============ Serialization ============ //

    pub fn toBytes(self: Mersenne31) [ENCODED_SIZE]u8 {
        // Normalize first in case greater than modulus
        const normalized = if (self.value >= MODULUS)
            self.value - MODULUS
        else
            self.value;
        return @bitCast(std.mem.nativeToLittle(u32, normalized));
    }

    pub fn fromBytes(bytes: [ENCODED_SIZE]u8) field.FieldError!Mersenne31 {
        const value = std.mem.littleToNative(u32, @bitCast(bytes));
        if (value >= MODULUS) return field.FieldError.InvalidValue;
        return .{ .value = value };
    }

    // ============ Comparison ============ //

    pub fn eql(a: Mersenne31, b: Mersenne31) bool {
        return a.value == b.value;
    }

    pub fn isZero(a: Mersenne31) bool {
        return a.value == 0;
    }

    // ============ Construction ============ //

    pub fn fromU64(x: u64) Mersenne31 {
        return reduce64(x);
    }

    pub fn fromU32(x: u32) Mersenne31 {
        var v = x;
        v = (v & MODULUS) +% (v >> 31);
        if (v >= MODULUS) v -%= MODULUS;
        return .{ .value = v };
    }

    pub fn random(rng: std.Random) Mersenne31 {
        // Sample uniformly from [0, p)
        while (true) {
            const candidate = rng.int(u32) >> 1; // 31 bits
            if (candidate < MODULUS) return .{ .value = candidate };
        }
    }

    // ============ Reduction ============ //

    /// Reduce a u64 to a canonical Mersenne31 element.
    /// Public for use in batch operations with delayed reduction.
    pub fn reduce64(x: u64) Mersenne31 {
        // Reduce x mod (2^31 - 1)
        // Key: 2^31 ≡ 1 (mod p)
        // So: x = lo + hi * 2^31 ≡ lo + hi (mod p)

        // First reduction: 64 bits → ~32 bits
        const lo: u64 = x & MODULUS;
        const hi: u64 = x >> 31;

        var sum = lo +% hi;

        // Second reduction: ~32 bits → 31 bits
        sum = (sum & MODULUS) +% (sum >> 31);

        // Final check
        if (sum >= MODULUS) sum -%= MODULUS;

        return .{ .value = @truncate(sum) };
    }
};

// Interface and generic tests
comptime {
    field.verify(Mersenne31);
    _ = field.tests(Mersenne31);
}

// ============ Mersenne31-Specific Tests ============ //

// ============ Reduction Edge Case Tests ============ //

test "addition near modulus" {
    const p = Mersenne31.MODULUS;
    const pm1 = Mersenne31{ .value = p - 1 };
    const pm2 = Mersenne31{ .value = p - 2 };

    // (p-1) + 1 = 0
    try std.testing.expect(pm1.add(Mersenne31.one).isZero());

    // (p-1) + (p-1) = p-2  (since -1 + -1 = -2)
    try std.testing.expect(pm1.add(pm1).eql(pm2));

    // (p-2) + 2 = 0
    const two = Mersenne31.fromU32(2);
    try std.testing.expect(pm2.add(two).isZero());
}

test "multiplication near modulus" {
    const p = Mersenne31.MODULUS;
    const pm1 = Mersenne31{ .value = p - 1 }; // -1

    // (-1) * (-1) = 1
    try std.testing.expect(pm1.mul(pm1).eql(Mersenne31.one));

    // 2 * (-1) = -2 = p-2
    const two = Mersenne31.fromU32(2);
    const pm2 = Mersenne31{ .value = p - 2 };
    try std.testing.expect(two.mul(pm1).eql(pm2));

    // 0 * (p-1) = 0
    try std.testing.expect(Mersenne31.zero.mul(pm1).isZero());
}

test "reduction half modulus" {
    const p = Mersenne31.MODULUS;
    const half_p = Mersenne31.fromU64(p / 2);
    const sum = half_p.add(half_p).add(half_p);
    const expected = Mersenne31.fromU64(@as(u64, 3) * (p / 2) % p);
    try std.testing.expect(sum.eql(expected));
}

// ============ Subtraction Tests ============ //

test "subtraction boundary values" {
    const p = Mersenne31.MODULUS;
    const vals = [_]u32{ 0, 1, 2, p - 2, p - 1 };

    for (vals) |a_raw| {
        for (vals) |b_raw| {
            const a = Mersenne31{ .value = a_raw };
            const b = Mersenne31{ .value = b_raw };

            const got = a.sub(b);

            // Reference: (a + p - b) mod p
            const expected_val: u32 = @intCast((@as(u64, a_raw) + p - b_raw) % p);
            try std.testing.expectEqual(expected_val, got.value);
        }
    }
}

test "subtraction specific cases" {
    const p = Mersenne31.MODULUS;

    // 0 - 1 = p-1
    try std.testing.expectEqual(p - 1, Mersenne31.zero.sub(Mersenne31.one).value);

    // 0 - (p-1) = 1
    const pm1 = Mersenne31{ .value = p - 1 };
    try std.testing.expectEqual(@as(u32, 1), Mersenne31.zero.sub(pm1).value);

    // 1 - 2 = p-1
    const two = Mersenne31.fromU32(2);
    try std.testing.expectEqual(p - 1, Mersenne31.one.sub(two).value);

    // (p-1) - (p-1) = 0
    try std.testing.expect(pm1.sub(pm1).isZero());
}

// ============ Constructor Tests ============ //

test "fromU32 edge cases" {
    const p = Mersenne31.MODULUS;

    try std.testing.expectEqual(@as(u32, 0), Mersenne31.fromU32(0).value);
    try std.testing.expectEqual(@as(u32, 1), Mersenne31.fromU32(1).value);
    try std.testing.expectEqual(p - 1, Mersenne31.fromU32(p - 1).value);
    try std.testing.expectEqual(@as(u32, 0), Mersenne31.fromU32(p).value);
    try std.testing.expectEqual(@as(u32, 1), Mersenne31.fromU32(p + 1).value);

    // max u32
    const max_u32: u32 = std.math.maxInt(u32);
    const expected: u32 = @intCast(@as(u64, max_u32) % p);
    try std.testing.expectEqual(expected, Mersenne31.fromU32(max_u32).value);
}

test "fromU64 edge cases" {
    const p: u64 = Mersenne31.MODULUS;

    try std.testing.expectEqual(@as(u32, 0), Mersenne31.fromU64(0).value);
    try std.testing.expectEqual(@as(u32, 1), Mersenne31.fromU64(1).value);
    try std.testing.expectEqual(@as(u32, p - 1), Mersenne31.fromU64(p - 1).value);
    try std.testing.expectEqual(@as(u32, 0), Mersenne31.fromU64(p).value);
    try std.testing.expectEqual(@as(u32, 1), Mersenne31.fromU64(p + 1).value);

    // 2^31 ≡ 1 (mod p)
    try std.testing.expectEqual(@as(u32, 1), Mersenne31.fromU64(1 << 31).value);

    // 2^31 - 1 = p, which reduces to 0
    try std.testing.expectEqual(@as(u32, 0), Mersenne31.fromU64((1 << 31) - 1).value);

    // max u64
    const max_u64: u64 = std.math.maxInt(u64);
    const expected: u32 = @intCast(max_u64 % p);
    try std.testing.expectEqual(expected, Mersenne31.fromU64(max_u64).value);

    // Large values
    try std.testing.expectEqual(@as(u32, @intCast(0xDEADBEEF % p)), Mersenne31.fromU64(0xDEADBEEF).value);
    try std.testing.expectEqual(@as(u32, @intCast(0xCAFEBABEDEADBEEF % p)), Mersenne31.fromU64(0xCAFEBABEDEADBEEF).value);
}

// ============ Serialization Tests ============ //

test "fromBytes rejects invalid values" {
    // p itself should be rejected
    const invalid_bytes: [4]u8 = @bitCast(std.mem.nativeToLittle(u32, Mersenne31.MODULUS));
    try std.testing.expectError(field.FieldError.InvalidValue, Mersenne31.fromBytes(invalid_bytes));

    // p+1 should be rejected
    const invalid_bytes2: [4]u8 = @bitCast(std.mem.nativeToLittle(u32, Mersenne31.MODULUS + 1));
    try std.testing.expectError(field.FieldError.InvalidValue, Mersenne31.fromBytes(invalid_bytes2));

    // max u32 should be rejected
    const invalid_bytes3: [4]u8 = @bitCast(std.mem.nativeToLittle(u32, std.math.maxInt(u32)));
    try std.testing.expectError(field.FieldError.InvalidValue, Mersenne31.fromBytes(invalid_bytes3));
}

// ============ Batch Operation Tests ============ //

test "batch ops match scalar ops" {
    var a = [_]Mersenne31{
        Mersenne31.fromU32(0),
        Mersenne31.fromU32(1),
        Mersenne31.fromU32(2),
        Mersenne31.fromU32(1234567),
    };
    var b = [_]Mersenne31{
        Mersenne31.fromU32(3),
        Mersenne31.fromU32(Mersenne31.MODULUS - 1),
        Mersenne31.fromU32(5),
        Mersenne31.fromU32(7654321),
    };

    var add_dst: [4]Mersenne31 = undefined;
    Mersenne31.addBatch(&add_dst, &a, &b);
    for (0..4) |i| {
        try std.testing.expect(add_dst[i].eql(a[i].add(b[i])));
    }

    var mul_dst: [4]Mersenne31 = undefined;
    Mersenne31.mulBatch(&mul_dst, &a, &b);
    for (0..4) |i| {
        try std.testing.expect(mul_dst[i].eql(a[i].mul(b[i])));
    }

    var sub_dst: [4]Mersenne31 = undefined;
    Mersenne31.subBatch(&sub_dst, &a, &b);
    for (0..4) |i| {
        try std.testing.expect(sub_dst[i].eql(a[i].sub(b[i])));
    }

    var mac_dst: [4]Mersenne31 = undefined;
    Mersenne31.mulAddBatch(&mac_dst, &a, &b, &a);
    for (0..4) |i| {
        try std.testing.expect(mac_dst[i].eql(a[i].mul(b[i]).add(a[i])));
    }
}

// ============ Batch Reduction Tests ============ //

test "sumSlice matches iterative add" {
    const values = [_]Mersenne31{
        Mersenne31.fromU32(100),
        Mersenne31.fromU32(200),
        Mersenne31.fromU32(300),
        Mersenne31.fromU32(Mersenne31.MODULUS - 1),
        Mersenne31.fromU32(Mersenne31.MODULUS - 2),
    };

    // Compute using sumSlice
    const fast_sum = Mersenne31.sumSlice(&values);

    // Compute using iterative add
    var slow_sum = Mersenne31.zero;
    for (values) |v| {
        slow_sum = slow_sum.add(v);
    }

    try std.testing.expect(fast_sum.eql(slow_sum));
}

test "sumSlice empty slice" {
    const empty: []const Mersenne31 = &.{};
    try std.testing.expect(Mersenne31.sumSlice(empty).isZero());
}

test "sumSlice large accumulation" {
    // Test that delayed reduction handles large sums correctly
    const p = Mersenne31.MODULUS;
    var values: [1000]Mersenne31 = undefined;
    for (&values) |*v| {
        v.* = Mersenne31{ .value = p - 1 }; // max value
    }

    const result = Mersenne31.sumSlice(&values);

    // Expected: 1000 * (p-1) mod p = 1000 * (-1) mod p = -1000 mod p = p - 1000
    const expected = Mersenne31.fromU64(@as(u64, 1000) * (p - 1));
    try std.testing.expect(result.eql(expected));
}

test "dotProduct matches iterative mul-add" {
    const a = [_]Mersenne31{
        Mersenne31.fromU32(2),
        Mersenne31.fromU32(3),
        Mersenne31.fromU32(4),
    };
    const b = [_]Mersenne31{
        Mersenne31.fromU32(5),
        Mersenne31.fromU32(6),
        Mersenne31.fromU32(7),
    };

    // Fast: dotProduct
    const fast = Mersenne31.dotProduct(&a, &b);

    // Slow: iterative
    var slow = Mersenne31.zero;
    for (a, b) |aa, bb| {
        slow = slow.add(aa.mul(bb));
    }

    // 2*5 + 3*6 + 4*7 = 10 + 18 + 28 = 56
    try std.testing.expect(fast.eql(slow));
    try std.testing.expect(fast.eql(Mersenne31.fromU32(56)));
}

test "linearCombineBatch matches scalar computation" {
    const a = [_]Mersenne31{
        Mersenne31.fromU32(10),
        Mersenne31.fromU32(20),
        Mersenne31.fromU32(30),
    };
    const b = [_]Mersenne31{
        Mersenne31.fromU32(50),
        Mersenne31.fromU32(60),
        Mersenne31.fromU32(70),
    };
    const r = Mersenne31.fromU32(3);

    var dst: [3]Mersenne31 = undefined;
    Mersenne31.linearCombineBatch(&dst, &a, &b, r);

    // Check: dst[i] = a[i] + r * (b[i] - a[i])
    for (dst, a, b) |result, aa, bb| {
        const diff = bb.sub(aa);
        const expected = aa.add(r.mul(diff));
        try std.testing.expect(result.eql(expected));
    }
}

test "linearCombineBatch handles b < a (underflow)" {
    // Test case where b[i] < a[i], requiring underflow handling
    const a = [_]Mersenne31{
        Mersenne31.fromU32(100),
        Mersenne31.fromU32(200),
    };
    const b = [_]Mersenne31{
        Mersenne31.fromU32(50), // b < a
        Mersenne31.fromU32(150), // b < a
    };
    const r = Mersenne31.fromU32(2);

    var dst: [2]Mersenne31 = undefined;
    Mersenne31.linearCombineBatch(&dst, &a, &b, r);

    // Verify against scalar
    for (dst, a, b) |result, aa, bb| {
        const diff = bb.sub(aa);
        const expected = aa.add(r.mul(diff));
        try std.testing.expect(result.eql(expected));
    }
}

// ============ Park-Miller MINSTD Test Vectors ============ //

test "Park-Miller MINSTD sequence" {
    // Classic Park-Miller LCG: x_{n+1} = 16807 * x_n mod (2^31 - 1)
    // These are well-known test vectors
    const multiplier = Mersenne31.fromU32(16807);
    var x = Mersenne31.fromU32(1);

    const expected = [_]u32{
        16807,
        282475249,
        1622650073,
        984943658,
        1144108930,
    };

    for (expected) |exp| {
        x = x.mul(multiplier);
        try std.testing.expectEqual(exp, x.value);
    }
}

test "Park-Miller MINSTD 10000th iteration" {
    // x_10000 = 1043618065 (well-known test vector)
    const multiplier = Mersenne31.fromU32(16807);
    var x = Mersenne31.fromU32(1);

    for (0..10000) |_| {
        x = x.mul(multiplier);
    }
    try std.testing.expectEqual(@as(u32, 1043618065), x.value);
}

// ============ Fermat's Little Theorem ============ //

test "Fermat's little theorem" {
    // a^(p-1) = 1 for non-zero a
    const a = Mersenne31.fromU64(12345);
    try std.testing.expect(a.pow(Mersenne31.MODULUS - 1).eql(Mersenne31.one));

    const b = Mersenne31.fromU64(9999999);
    try std.testing.expect(b.pow(Mersenne31.MODULUS - 1).eql(Mersenne31.one));
}
