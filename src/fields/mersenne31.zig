const std = @import("std");
const field = @import("field.zig");

pub const Mersenne31 = struct {
    value: u32,

    pub const MODULUS: u32 = 0x7FFFFFFF; // 2^31 - 1
    pub const ENCODED_SIZE: usize = 4;

    const MODULUS_U64: u64 = MODULUS;

    pub const zero: Mersenne31 = .{ .value = 0 };
    pub const one: Mersenne31 = .{ .value = 1 };

    // ============ Core Arithmetic ============

    pub fn add(a: Mersenne31, b: Mersenne31) Mersenne31 {
        // a, b < 2^31, so sum < 2^32, fits in u32 with possible wrap
        var sum: u32 = a.value +% b.value;

        // Mersenne reduction: x mod (2^31 - 1) = (x & MODULUS) + (x >> 31)
        sum = (sum & MODULUS) + (sum >> 31);

        // One more reduction might be needed
        if (sum >= MODULUS) sum -= MODULUS;

        return .{ .value = sum };
    }

    pub fn sub(a: Mersenne31, b: Mersenne31) Mersenne31 {
        if (a.value >= b.value) {
            return .{ .value = a.value - b.value };
        } else {
            return .{ .value = MODULUS - (b.value - a.value) };
        }
    }

    pub fn mul(a: Mersenne31, b: Mersenne31) Mersenne31 {
        const wide: u64 = @as(u64, a.value) * @as(u64, b.value);
        return reduce64(wide);
    }

    pub fn neg(a: Mersenne31) Mersenne31 {
        return if (a.value == 0) a else .{ .value = MODULUS - a.value };
    }

    // ============ Derived Arithmetic ============

    pub fn square(a: Mersenne31) Mersenne31 {
        return a.mul(a);
    }

    pub fn double(a: Mersenne31) Mersenne31 {
        return a.add(a);
    }

    pub fn inv(a: Mersenne31) Mersenne31 {
        std.debug.assert(!a.isZero());
        return a.pow(MODULUS - 2);
    }

    pub fn pow(base: Mersenne31, exp: u32) Mersenne31 {
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

    // ============ Serialization ============

    pub fn toBytes(self: Mersenne31) [ENCODED_SIZE]u8 {
        return @bitCast(std.mem.nativeToLittle(u32, self.value));
    }

    pub fn fromBytes(bytes: [ENCODED_SIZE]u8) field.FieldError!Mersenne31 {
        const value = std.mem.littleToNative(u32, @bitCast(bytes));
        if (value >= MODULUS) return field.FieldError.InvalidValue;
        return .{ .value = value };
    }

    // ============ Comparison ============

    pub fn eql(a: Mersenne31, b: Mersenne31) bool {
        return a.value == b.value;
    }

    pub fn isZero(a: Mersenne31) bool {
        return a.value == 0;
    }

    // ============ Construction ============

    pub fn fromU64(x: u64) Mersenne31 {
        const reduced = reduce64(x);
        return reduced;
    }

    pub fn fromU32(x: u32) Mersenne31 {
        return if (x >= MODULUS) .{ .value = x - MODULUS } else .{ .value = x };
    }

    pub fn random(rng: std.Random) Mersenne31 {
        while (true) {
            const candidate = rng.int(u32) & MODULUS; // Mask to 31 bits
            if (candidate < MODULUS) return .{ .value = candidate };
        }
    }

    // ============ Internal ============

    fn reduce64(x: u64) Mersenne31 {
        // Reduce x mod (2^31 - 1)
        // x = x_lo + x_hi * 2^31
        // 2^31 ≡ 1 (mod 2^31 - 1)
        // So x ≡ x_lo + x_hi (mod p)

        var lo: u32 = @truncate(x & MODULUS_U64);
        var hi: u32 = @truncate(x >> 31);

        // Could need multiple folds for large x
        while (hi > 0) {
            const sum = @as(u64, lo) + @as(u64, hi);
            lo = @truncate(sum & MODULUS_U64);
            hi = @truncate(sum >> 31);
        }

        if (lo >= MODULUS) lo -= MODULUS;

        return .{ .value = lo };
    }
};

// Interface check
comptime {
    field.verify(Mersenne31);
}

test "additive identity" {
    const a = Mersenne31.fromU64(12345);
    try std.testing.expect(a.add(Mersenne31.zero).eql(a));
}
