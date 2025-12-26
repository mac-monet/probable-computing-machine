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

    pub fn addBatch(dst: []Mersenne31, a: []Mersenne31, b: []Mersenne31) void {
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

    // ============ Internal ============ //

    fn reduce64(x: u64) Mersenne31 {
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

// Interface check
comptime {
    field.verify(Mersenne31);
}

test "additive identity" {
    const a = Mersenne31.fromU64(12345);
    try std.testing.expect(a.add(Mersenne31.zero).eql(a));
}
