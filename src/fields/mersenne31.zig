const std = @import("std");
const Field = @import("field.zig");

/// Mersenne-31: p = 2^31 - 1
pub const Mersenne31 = packed struct(u32) {
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

        // Branchless final reduction: subtract MODULUS if sum >= MODULUS
        sum -%= @as(u32, @intFromBool(sum >= MODULUS)) *% MODULUS;

        return .{ .value = sum };
    }

    pub fn sub(a: Mersenne31, b: Mersenne31) Mersenne31 {
        const diff = a.value +% (MODULUS - b.value);
        // Branchless: diff is in [0, 2*MODULUS), reduce once
        return .{ .value = diff -% @as(u32, @intFromBool(diff >= MODULUS)) *% MODULUS };
    }

    pub fn mul(a: Mersenne31, b: Mersenne31) Mersenne31 {
        // Product fits in u64
        const wide: u64 = @as(u64, a.value) * @as(u64, b.value);
        return reduce64(wide);
    }

    pub fn neg(a: Mersenne31) Mersenne31 {
        return if (a.value == 0) a else .{ .value = MODULUS - a.value };
    }

    // ============ Batch Operations ============ //

    /// Sum N slices with sequential per-slice access.
    /// Each slice is processed entirely before moving to the next.
    /// LLVM auto-vectorizes this well; explicit SIMD adds overhead.
    pub fn sumSlices(comptime N: usize, slices: [N][]const Mersenne31) [N]Mersenne31 {
        const len = slices[0].len;
        comptime var check = 1;
        inline while (check < N) : (check += 1) {
            std.debug.assert(slices[check].len == len);
        }

        var results: [N]Mersenne31 = undefined;
        inline for (0..N) |s| {
            var acc: u64 = 0;
            for (slices[s]) |v| {
                acc +%= v.value;
            }
            results[s] = reduce64(acc);
        }
        return results;
    }

    /// Compute dot product: sum(a[i] * b[i]) with SIMD acceleration.
    /// Uses partial reduction per product to avoid overflow.
    /// For best performance, inputs should be aligned to simd.alignment bytes.
    pub fn dotProduct(a: []const Mersenne31, b: []const Mersenne31) Mersenne31 {
        std.debug.assert(a.len == b.len);

        const simd = @import("../simd.zig");
        const VEC_LEN = simd.u32_len orelse 4;
        const VecU64 = @Vector(VEC_LEN, u64);

        const len = a.len;

        // VEC_LEN parallel accumulators - horizontal sum only at the end
        var acc: VecU64 = @splat(0);

        // SIMD loop: process VEC_LEN elements per iteration
        var i: usize = 0;
        while (i + VEC_LEN <= len) : (i += VEC_LEN) {
            acc +%= dotProductChunk(VEC_LEN, a[i..][0..VEC_LEN], b[i..][0..VEC_LEN]);
        }

        // Horizontal sum of parallel accumulators
        var scalar_acc: u64 = @reduce(.Add, acc);

        // Tail loop: process remaining elements
        while (i < len) : (i += 1) {
            const prod: u64 = @as(u64, a[i].value) * @as(u64, b[i].value);
            // Partial reduction: 62 bits → ~32 bits
            scalar_acc +%= (prod & MODULUS) + (prod >> 31);
        }

        return reduce64(scalar_acc);
    }

    /// Process one chunk of VEC_LEN elements for dotProduct.
    /// Returns vector of partial products (each reduced to ~32 bits).
    inline fn dotProductChunk(
        comptime VEC_LEN: comptime_int,
        a: *const [VEC_LEN]Mersenne31,
        b: *const [VEC_LEN]Mersenne31,
    ) @Vector(VEC_LEN, u64) {
        const VecU32 = @Vector(VEC_LEN, u32);
        const VecU64 = @Vector(VEC_LEN, u64);

        // Load as u32 vectors (safe due to packed struct layout)
        const a_vec: VecU32 = @bitCast(a.*);
        const b_vec: VecU32 = @bitCast(b.*);

        // Widen to u64 for multiply
        const a_wide: VecU64 = a_vec;
        const b_wide: VecU64 = b_vec;

        // Multiply: result is ~62 bits per lane
        const prod: VecU64 = a_wide *% b_wide;

        // Partial reduction per product: 62 bits → ~32 bits
        // This allows accumulating billions of products without overflow.
        // Key: (prod & MODULUS) + (prod >> 31) ≡ prod (mod p)
        const mask: VecU64 = @splat(@as(u64, MODULUS));
        const lo = prod & mask;
        const hi = prod >> @as(@Vector(VEC_LEN, u6), @splat(31));

        return lo +% hi;
    }

    /// Linear interpolation: dst[i] = a[i] + r * (b[i] - a[i])
    /// Used for binding variables in multilinear polynomials.
    /// For best performance, inputs should be aligned to simd.alignment bytes.
    pub fn linearCombineBatch(dst: []Mersenne31, a: []const Mersenne31, b: []const Mersenne31, r: Mersenne31) void {
        std.debug.assert(dst.len == a.len and a.len == b.len);

        // TODO why is this 4 as default?
        const simd = @import("../simd.zig");
        const VEC_LEN = simd.u32_len orelse 4;

        const len = dst.len;
        const r_val: u64 = r.value;

        // SIMD loop: process VEC_LEN elements per iteration
        var i: usize = 0;
        while (i + VEC_LEN <= len) : (i += VEC_LEN) {
            linearCombineChunk(VEC_LEN, dst[i..][0..VEC_LEN], a[i..][0..VEC_LEN], b[i..][0..VEC_LEN], r_val);
        }

        // Tail loop: process remaining elements
        while (i < len) : (i += 1) {
            const diff: u64 = @as(u64, b[i].value) +% MODULUS -% a[i].value;
            const prod = r_val *% diff;
            const result = @as(u64, a[i].value) +% prod;
            dst[i] = reduce64(result);
        }
    }

    /// Process one chunk of VEC_LEN elements for linearCombineBatch.
    /// Extracted to help LLVM optimize the SIMD operations.
    inline fn linearCombineChunk(
        comptime VEC_LEN: comptime_int,
        dst: *[VEC_LEN]Mersenne31,
        a: *const [VEC_LEN]Mersenne31,
        b: *const [VEC_LEN]Mersenne31,
        r_val: u64,
    ) void {
        const VecU32 = @Vector(VEC_LEN, u32);
        const VecU64 = @Vector(VEC_LEN, u64);

        // Load as u32 vectors (safe due to packed struct layout)
        const a_vec: VecU32 = @bitCast(a.*);
        const b_vec: VecU32 = @bitCast(b.*);

        // Compute diff in u32: b + MODULUS - a
        // No overflow: values < MODULUS, so b + MODULUS < 2^32
        const modulus_vec: VecU32 = @splat(MODULUS);
        const diff_vec: VecU32 = b_vec +% modulus_vec -% a_vec;

        // Widen to u64 for multiply
        const diff_wide: VecU64 = diff_vec;
        const a_wide: VecU64 = a_vec;
        const r_vec: VecU64 = @splat(r_val);

        // prod = r * diff, result = a + prod
        const prod: VecU64 = r_vec *% diff_wide;
        const result: VecU64 = a_wide +% prod;

        // Reduce and store
        dst.* = @bitCast(reduce64Vec(VEC_LEN, result));
    }

    // ============ Derived Arithmetic ============ //

    pub const square = Field.defaults(Mersenne31).square;
    pub const double = Field.defaults(Mersenne31).double;
    pub const pow = Field.defaults(Mersenne31).pow;

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

    pub fn fromBytes(bytes: [ENCODED_SIZE]u8) Field.FieldError!Mersenne31 {
        const value = std.mem.littleToNative(u32, @bitCast(bytes));
        if (value >= MODULUS) return Field.FieldError.InvalidValue;
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
        // Branchless final reduction
        v -%= @as(u32, @intFromBool(v >= MODULUS)) *% MODULUS;
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

    /// Reduce a vector of u64 values to canonical Mersenne31 u32 values.
    /// Used by batch operations for SIMD reduction.
    inline fn reduce64Vec(comptime N: comptime_int, x: @Vector(N, u64)) @Vector(N, u32) {
        const VecU64 = @Vector(N, u64);
        const mask: VecU64 = @splat(@as(u64, MODULUS));

        // Step 1: 64 bits → ~32 bits
        const lo1 = x & mask;
        const hi1 = x >> @as(@Vector(N, u6), @splat(31));
        var sum = lo1 +% hi1;

        // Step 2: ~32 bits → 31 bits
        const lo2 = sum & mask;
        const hi2 = sum >> @as(@Vector(N, u6), @splat(31));
        sum = lo2 +% hi2;

        // Step 3: branchless final reduction
        const ge_mask = sum >= mask;
        sum = @select(u64, ge_mask, sum -% mask, sum);

        return @truncate(sum);
    }

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

        // Branchless final reduction
        sum -%= @as(u64, @intFromBool(sum >= MODULUS)) *% MODULUS;

        return .{ .value = @truncate(sum) };
    }
};

// ============ Quadratic Extension Field ============ //

/// Configuration for Mersenne31 quadratic extension.
/// F_p^2 = F_p[x] / (x^2 - 11), where 11 is a quadratic non-residue.
const ExtConfig = struct {
    pub const Base = Mersenne31;
    pub const degree = 2;

    /// The non-residue W such that x^2 = W defines the extension.
    /// 11 is a quadratic non-residue mod 2^31 - 1.
    pub const W: u64 = 11;

    /// Reduce polynomial of degree < 2*degree to degree < degree.
    /// c0 + c1*x + c2*x^2 -> (c0 + c2*W) + c1*x
    pub fn reduce(product: *const [3]Mersenne31) [2]Mersenne31 {
        return .{
            product[0].add(product[2].mul(Mersenne31.fromU64(W))),
            product[1],
        };
    }
};

/// Quadratic extension of Mersenne31: elements of F_p^2.
pub const Ext = Field.ExtField.init(ExtConfig);

// Interface and generic tests
comptime {
    Field.verify(Mersenne31);
    _ = Field.tests(Mersenne31);

    // Verify packed struct layout for SIMD compatibility
    std.debug.assert(@sizeOf(Mersenne31) == 4);
    std.debug.assert(@bitSizeOf(Mersenne31) == 32);

    // Extension field verification
    Field.ExtField.verify(Ext);
    _ = Field.ExtField.tests(Ext);
}

// ============ Mersenne31-Specific Tests ============ //

// ============ Extension Field Tests ============ //

test "Ext: multiplication uses irreducible correctly" {
    // (0 + 1*x) * (0 + 1*x) = x^2 = 11 (since x^2 = W = 11)
    const x = Ext{ .coeffs = .{ Mersenne31.zero, Mersenne31.one } };
    const x_squared = x.mul(x);

    // x^2 should equal 11 (embedded in extension)
    const expected = Ext.fromBase(Mersenne31.fromU64(11));
    try std.testing.expect(x_squared.eql(expected));
}

test "Ext: inversion" {
    // Test that a * a^(-1) = 1 for various elements
    const test_vals = [_][2]u64{
        .{ 1, 0 }, // base field element
        .{ 0, 1 }, // x
        .{ 1, 1 }, // 1 + x
        .{ 123, 456 }, // arbitrary
    };

    for (test_vals) |v| {
        const a = Ext{ .coeffs = .{
            Mersenne31.fromU64(v[0]),
            Mersenne31.fromU64(v[1]),
        } };
        if (a.isZero()) continue;

        const a_inv = a.inv();
        const product = a.mul(a_inv);
        try std.testing.expect(product.eql(Ext.one));
    }
}

test "Ext: mulBase optimization" {
    // mulBase should be equivalent to mul(fromBase) but faster
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    for (0..10) |_| {
        const ext_a = Ext.random(rng);
        const base_b = Mersenne31.random(rng);

        const via_mulBase = ext_a.mulBase(base_b);
        const via_mul = ext_a.mul(Ext.fromBase(base_b));

        try std.testing.expect(via_mulBase.eql(via_mul));
    }
}

test "Ext.Batch: dotProductMixed" {
    // dotProductMixed(bases, exts) = sum(bases[i] * exts[i])
    const bases = [_]Mersenne31{
        Mersenne31.fromU64(2),
        Mersenne31.fromU64(3),
        Mersenne31.fromU64(5),
    };

    // SoA storage: c0s and c1s separate
    var c0s = [_]Mersenne31{ Mersenne31.fromU64(1), Mersenne31.fromU64(3), Mersenne31.fromU64(5) };
    var c1s = [_]Mersenne31{ Mersenne31.fromU64(2), Mersenne31.fromU64(4), Mersenne31.fromU64(6) };
    const batch = Ext.Batch{ .coeffs = .{ &c0s, &c1s } };

    const result = batch.dotProductMixed(&bases);

    // Manual: 2*(1+2x) + 3*(3+4x) + 5*(5+6x) = (2+9+25) + (4+12+30)x = 36 + 46x
    const expected = Ext{ .coeffs = .{ Mersenne31.fromU64(36), Mersenne31.fromU64(46) } };
    try std.testing.expect(result.eql(expected));
}

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
    try std.testing.expectError(Field.FieldError.InvalidValue, Mersenne31.fromBytes(invalid_bytes));

    // p+1 should be rejected
    const invalid_bytes2: [4]u8 = @bitCast(std.mem.nativeToLittle(u32, Mersenne31.MODULUS + 1));
    try std.testing.expectError(Field.FieldError.InvalidValue, Mersenne31.fromBytes(invalid_bytes2));

    // max u32 should be rejected
    const invalid_bytes3: [4]u8 = @bitCast(std.mem.nativeToLittle(u32, std.math.maxInt(u32)));
    try std.testing.expectError(Field.FieldError.InvalidValue, Mersenne31.fromBytes(invalid_bytes3));
}

// ============ Batch Operation Tests ============ //

test "sumSlices matches iterative add" {
    const values = [_]Mersenne31{
        Mersenne31.fromU32(100),
        Mersenne31.fromU32(200),
        Mersenne31.fromU32(300),
        Mersenne31.fromU32(Mersenne31.MODULUS - 1),
        Mersenne31.fromU32(Mersenne31.MODULUS - 2),
    };

    // Compute using sumSlices with N=1
    const fast_sum = Mersenne31.sumSlices(1, .{&values})[0];

    // Compute using iterative add
    var slow_sum = Mersenne31.zero;
    for (values) |v| {
        slow_sum = slow_sum.add(v);
    }

    try std.testing.expect(fast_sum.eql(slow_sum));
}

test "sumSlices empty slice" {
    const empty: []const Mersenne31 = &.{};
    try std.testing.expect(Mersenne31.sumSlices(1, .{empty})[0].isZero());
}

test "sumSlices large accumulation" {
    // Test that delayed reduction handles large sums correctly
    const p = Mersenne31.MODULUS;
    var values: [1000]Mersenne31 = undefined;
    for (&values) |*v| {
        v.* = Mersenne31{ .value = p - 1 }; // max value
    }

    const result = Mersenne31.sumSlices(1, .{&values})[0];

    // Expected: 1000 * (p-1) mod p = 1000 * (-1) mod p = -1000 mod p = p - 1000
    const expected = Mersenne31.fromU64(@as(u64, 1000) * (p - 1));
    try std.testing.expect(result.eql(expected));
}

test "sumSlices multiple slices (sumcheck pattern)" {
    // Simulates sumcheck: sum left half and right half of evaluations
    const evals = [_]Mersenne31{
        // Left half (x_0 = 0)
        Mersenne31.fromU32(10),
        Mersenne31.fromU32(20),
        Mersenne31.fromU32(30),
        Mersenne31.fromU32(40),
        // Right half (x_0 = 1)
        Mersenne31.fromU32(15),
        Mersenne31.fromU32(25),
        Mersenne31.fromU32(35),
        Mersenne31.fromU32(45),
    };

    const half = evals.len / 2;
    const sums = Mersenne31.sumSlices(2, .{ evals[0..half], evals[half..] });

    // Verify against manual sums
    // Left: 10 + 20 + 30 + 40 = 100
    // Right: 15 + 25 + 35 + 45 = 120
    try std.testing.expect(sums[0].eql(Mersenne31.fromU32(100)));
    try std.testing.expect(sums[1].eql(Mersenne31.fromU32(120)));
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

test "dotProduct large accumulation (overflow)" {
    // Test that dotProduct handles large accumulations correctly.
    // Each product is ~62 bits: (2^31-2)^2 ≈ 2^62
    // A u64 accumulator overflows after just 4 products.
    // Wrapping arithmetic causes error because 2^64 ≡ 4 (mod p), not 0.
    const p = Mersenne31.MODULUS;
    const max_val = Mersenne31{ .value = p - 1 }; // 2^31 - 2

    // Use 16 elements - enough to trigger multiple overflows
    var a: [16]Mersenne31 = undefined;
    var b: [16]Mersenne31 = undefined;
    for (&a, &b) |*aa, *bb| {
        aa.* = max_val;
        bb.* = max_val;
    }

    // Fast: dotProduct
    const fast = Mersenne31.dotProduct(&a, &b);

    // Slow: iterative with proper reduction each step
    var slow = Mersenne31.zero;
    for (a, b) |aa, bb| {
        slow = slow.add(aa.mul(bb));
    }

    // If dotProduct uses wrapping arithmetic incorrectly,
    // each overflow adds an error of 4 to the result.
    try std.testing.expect(fast.eql(slow));
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

// ============ Aligned Allocation Tests ============ //

test "dotProduct with aligned allocation" {
    const simd = @import("../simd.zig");
    const allocator = std.testing.allocator;

    // Test with exact SIMD multiple (32 elements)
    const n_exact = 32;
    const a_exact = try allocator.alignedAlloc(Mersenne31, simd.alignment, n_exact);
    defer allocator.free(a_exact);
    const b_exact = try allocator.alignedAlloc(Mersenne31, simd.alignment, n_exact);
    defer allocator.free(b_exact);

    for (a_exact, b_exact, 0..) |*aa, *bb, i| {
        aa.* = Mersenne31.fromU32(@intCast(i + 1));
        bb.* = Mersenne31.fromU32(@intCast(i + 100));
    }

    const fast_exact = Mersenne31.dotProduct(a_exact, b_exact);
    var slow_exact = Mersenne31.zero;
    for (a_exact, b_exact) |aa, bb| {
        slow_exact = slow_exact.add(aa.mul(bb));
    }
    try std.testing.expect(fast_exact.eql(slow_exact));

    // Test with tail (37 elements = 32 + 5 tail on AVX2, or other combos)
    const n_tail = 37;
    const a_tail = try allocator.alignedAlloc(Mersenne31, simd.alignment, n_tail);
    defer allocator.free(a_tail);
    const b_tail = try allocator.alignedAlloc(Mersenne31, simd.alignment, n_tail);
    defer allocator.free(b_tail);

    for (a_tail, b_tail, 0..) |*aa, *bb, i| {
        aa.* = Mersenne31.fromU32(@intCast(i * 7 + 3));
        bb.* = Mersenne31.fromU32(@intCast(i * 11 + 5));
    }

    const fast_tail = Mersenne31.dotProduct(a_tail, b_tail);
    var slow_tail = Mersenne31.zero;
    for (a_tail, b_tail) |aa, bb| {
        slow_tail = slow_tail.add(aa.mul(bb));
    }
    try std.testing.expect(fast_tail.eql(slow_tail));
}

test "linearCombineBatch with aligned allocation" {
    const simd = @import("../simd.zig");
    const allocator = std.testing.allocator;

    // Test with exact SIMD multiple (32 elements)
    const n_exact = 32;
    const a = try allocator.alignedAlloc(Mersenne31, simd.alignment, n_exact);
    defer allocator.free(a);
    const b = try allocator.alignedAlloc(Mersenne31, simd.alignment, n_exact);
    defer allocator.free(b);
    const dst = try allocator.alignedAlloc(Mersenne31, simd.alignment, n_exact);
    defer allocator.free(dst);

    for (a, b, 0..) |*aa, *bb, i| {
        aa.* = Mersenne31.fromU32(@intCast(i * 3 + 10));
        bb.* = Mersenne31.fromU32(@intCast(i * 5 + 50));
    }
    const r = Mersenne31.fromU32(7);

    Mersenne31.linearCombineBatch(dst, a, b, r);

    // Verify against scalar
    for (dst, a, b) |result, aa, bb| {
        const diff = bb.sub(aa);
        const expected = aa.add(r.mul(diff));
        try std.testing.expect(result.eql(expected));
    }

    // Test with tail (37 elements)
    const n_tail = 37;
    const a2 = try allocator.alignedAlloc(Mersenne31, simd.alignment, n_tail);
    defer allocator.free(a2);
    const b2 = try allocator.alignedAlloc(Mersenne31, simd.alignment, n_tail);
    defer allocator.free(b2);
    const dst2 = try allocator.alignedAlloc(Mersenne31, simd.alignment, n_tail);
    defer allocator.free(dst2);

    for (a2, b2, 0..) |*aa, *bb, i| {
        aa.* = Mersenne31.fromU32(@intCast(i * 13 + 1));
        bb.* = Mersenne31.fromU32(@intCast(i * 17 + 2));
    }

    Mersenne31.linearCombineBatch(dst2, a2, b2, r);

    for (dst2, a2, b2) |result, aa, bb| {
        const diff = bb.sub(aa);
        const expected = aa.add(r.mul(diff));
        try std.testing.expect(result.eql(expected));
    }
}

test "dotProduct large aligned (stress SIMD accumulation)" {
    const simd = @import("../simd.zig");
    const allocator = std.testing.allocator;

    // Large array to stress SIMD accumulation over many iterations
    const n = 1024;
    const a = try allocator.alignedAlloc(Mersenne31, simd.alignment, n);
    defer allocator.free(a);
    const b = try allocator.alignedAlloc(Mersenne31, simd.alignment, n);
    defer allocator.free(b);

    // Use near-max values to stress overflow handling
    const max_val = Mersenne31{ .value = Mersenne31.MODULUS - 1 };
    for (a, b) |*aa, *bb| {
        aa.* = max_val;
        bb.* = max_val;
    }

    const fast = Mersenne31.dotProduct(a, b);

    // Scalar reference
    var slow = Mersenne31.zero;
    for (a, b) |aa, bb| {
        slow = slow.add(aa.mul(bb));
    }

    try std.testing.expect(fast.eql(slow));
}

test "linearCombineBatch large aligned" {
    const simd = @import("../simd.zig");
    const allocator = std.testing.allocator;

    const n = 1024;
    const a = try allocator.alignedAlloc(Mersenne31, simd.alignment, n);
    defer allocator.free(a);
    const b = try allocator.alignedAlloc(Mersenne31, simd.alignment, n);
    defer allocator.free(b);
    const dst = try allocator.alignedAlloc(Mersenne31, simd.alignment, n);
    defer allocator.free(dst);

    // Mix of values including edge cases
    for (a, b, 0..) |*aa, *bb, i| {
        aa.* = Mersenne31.fromU32(@intCast(i * 1000 % Mersenne31.MODULUS));
        bb.* = Mersenne31.fromU32(@intCast((i * 7777 + 123) % Mersenne31.MODULUS));
    }
    const r = Mersenne31.fromU32(999999);

    Mersenne31.linearCombineBatch(dst, a, b, r);

    // Verify against scalar
    for (dst, a, b) |result, aa, bb| {
        const diff = bb.sub(aa);
        const expected = aa.add(r.mul(diff));
        try std.testing.expect(result.eql(expected));
    }
}
