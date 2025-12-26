const std = @import("std");
const field = @import("field.zig");

const Goldilocks = struct {
    value: u64,

    // Prime modulus p = 2^64 - 2^32 + 1
    pub const MODULUS: u64 = 0xFFFFFFFF00000001;

    // Size of the serialized encoding
    pub const ENCODED_SIZE: usize = 8;

    // Useful for fast reduction: 2^64 mod p = 2^32 - 1
    const EPSILON: u64 = 0xFFFFFFFF;

    pub const zero: Goldilocks = .{ .value = 0 };
    pub const one: Goldilocks = .{ .value = 1 };

    // TODO this and sub can be made completely branchless
    pub fn add(a: Goldilocks, b: Goldilocks) Goldilocks {
        const result = @addWithOverflow(a.value, b.value);
        const sum = result[0];
        const carry = result[1];

        // If overflow occurred or sum >= p, we need to reduce
        // Subtracting p is equivalent to adding EPSILON with wrapping
        const needs_reduce = (carry == 1) or (sum >= MODULUS);
        const correction: u64 = if (needs_reduce) EPSILON else 0;

        return .{ .value = sum +% correction };
    }

    pub fn sub(a: Goldilocks, b: Goldilocks) Goldilocks {
        const result = @subWithOverflow(a.value, b.value);
        const diff = result[0];
        const borrow = result[1];

        // If borrow, add p back (equivalent to subtractiong EPSILON + 1 with wrapping)
        const correction: u64 = if (borrow == 1) EPSILON else 0;

        return .{ .value = diff -% correction };
    }

    pub fn mul(a: Goldilocks, b: Goldilocks) Goldilocks {
        const wide: u128 = @as(u128, a.value) * @as(u128, b.value);
        return reduce128(wide);
    }

    // TODO impl batch functions
    pub fn addBatch(_: []Goldilocks, _: []const Goldilocks, _: []const Goldilocks) void {
        @panic("TODO");
    }
    pub fn mulBatch(_: []Goldilocks, _: []const Goldilocks, _: []const Goldilocks) void {
        @panic("TODO");
    }
    pub fn subBatch(_: []Goldilocks, _: []const Goldilocks, _: []const Goldilocks) void {
        @panic("TODO");
    }
    pub fn mulAddBatch(_: []Goldilocks, _: []const Goldilocks, _: []const Goldilocks, _: []const Goldilocks) void {
        @panic("TODO");
    }

    pub fn neg(a: Goldilocks) Goldilocks {
        return if (a.value == 0) a else .{ .value = MODULUS - a.value };
    }

    // ========= Derived Arithmetic ========= //

    pub const square = field.defaults(Goldilocks).square;

    pub const double = field.defaults(Goldilocks).double;

    pub fn inv(a: Goldilocks) Goldilocks {
        std.debug.assert(!a.isZero());
        return a.pow(MODULUS - 2);
    }

    pub fn pow(base: Goldilocks, exp: u64) Goldilocks {
        if (exp == 0) return one;

        var result = one;
        var b = base;
        var e = exp;

        while (e > 0) {
            if (e & 1 == 1) result = result.mul(b);
            b = b.square();
            e >>= 1;
        }

        return result;
    }

    // ========= Comparison ========= //

    pub fn eql(a: Goldilocks, b: Goldilocks) bool {
        return a.value == b.value;
    }

    pub fn isZero(self: Goldilocks) bool {
        return self.value == 0;
    }

    // ========= Serialization ========= //

    pub fn toBytes(self: Goldilocks) [ENCODED_SIZE]u8 {
        return @bitCast(std.mem.nativeToLittle(u64, self.value));
    }

    pub fn fromBytes(bytes: [ENCODED_SIZE]u8) field.FieldError!Goldilocks {
        const value = std.mem.littleToNative(u64, @bitCast(bytes));
        if (value >= MODULUS) return field.FieldError.InvalidValue;
        return .{ .value = value };
    }

    // ========= Construction ========= //

    pub fn fromU64(x: u64) Goldilocks {
        return if (x >= MODULUS) .{ .value = x - MODULUS } else .{ .value = x };
    }

    pub fn random(rng: std.Random) Goldilocks {
        while (true) {
            const candidate = rng.int(u64);
            if (candidate < MODULUS) return .{ .value = candidate };
        }
    }

    // ========= Internal ========= //

    fn reduce128(x: u128) Goldilocks {
        // x = x_lo + x_hi * 2^64
        // 2^64 ≡ EPSILON (mod p)
        // So: x ≡ x_lo + x_hi * EPSILON (mod p)

        const x_lo: u64 = @truncate(x);
        const x_hi: u64 = @truncate(x >> 64);

        // x_hi * EPSILON could be up to ~96 bits
        const mid: u128 = @as(u128, x_hi) * EPSILON;
        const mid_lo: u64 = @truncate(mid);
        const mid_hi: u64 = @truncate(mid >> 64);

        const r1 = @addWithOverflow(x_lo, mid_lo);
        const sum1 = r1[0];
        const c1: u64 = r1[1];

        // Add (mid_hi + c1) * EPSILON
        // Note: mid_hi is at most 32 bits, c1 is 0 or 1
        // So (mid_hi + c1) * EPSILON fits in u64
        const correction = (mid_hi +% c1) *% EPSILON;

        const r2 = @addWithOverflow(sum1, correction);
        var result = r2[0];
        const c2 = r2[1];

        if (c2 == 1 or result >= MODULUS) {
            result -%= MODULUS;
        }

        return .{ .value = result };
    }
};

// Interface check
comptime {
    field.verify(Goldilocks);
}

// TODO review, make generic
test "additive identity" {
    const a = Goldilocks.fromU64(12345);
    try std.testing.expect(a.add(Goldilocks.zero).eql(a));
}

test "multiplicative identity" {
    const a = Goldilocks.fromU64(12345);
    try std.testing.expect(a.mul(Goldilocks.one).eql(a));
}

test "additive inverse" {
    const a = Goldilocks.fromU64(12345);
    try std.testing.expect(a.add(a.neg()).isZero());
}

test "multiplicative inverse" {
    const a = Goldilocks.fromU64(12345);
    try std.testing.expect(a.mul(a.inv()).eql(Goldilocks.one));
}

test "commutativity" {
    const a = Goldilocks.fromU64(111);
    const b = Goldilocks.fromU64(222);
    try std.testing.expect(a.add(b).eql(b.add(a)));
    try std.testing.expect(a.mul(b).eql(b.mul(a)));
}

test "distributivity" {
    const a = Goldilocks.fromU64(111);
    const b = Goldilocks.fromU64(222);
    const c = Goldilocks.fromU64(333);

    const lhs = a.mul(b.add(c));
    const rhs = a.mul(b).add(a.mul(c));
    try std.testing.expect(lhs.eql(rhs));
}

test "reduction edge cases" {
    // Near modulus
    const near_p = Goldilocks{ .value = Goldilocks.MODULUS - 1 };
    const result = near_p.add(Goldilocks.one);
    try std.testing.expect(result.isZero());

    // Overflow in addition
    const half_p = Goldilocks.fromU64(Goldilocks.MODULUS / 2);
    const sum = half_p.add(half_p).add(half_p);
    // 3 * floor(p/2) mod p = floor(p/2) - 1 (since p is odd)
    try std.testing.expect(sum.eql(half_p.sub(Goldilocks.one)));
}

test "serialization roundtrip" {
    const original = Goldilocks.fromU64(0xDEADBEEFCAFEBABE % Goldilocks.MODULUS);
    const bytes = original.toBytes();
    const recovered = try Goldilocks.fromBytes(bytes);
    try std.testing.expect(original.eql(recovered));
}
