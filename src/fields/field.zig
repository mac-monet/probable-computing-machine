const std = @import("std");
const testing = std.testing;

// Finite Field interface. Loosely follows std.crypto.ff.

/// Common error type for field operations
pub const FieldError = error{
    InvalidValue,
};

/// Documents the required field interface.
/// Use: `comptime { verify(MyField); }` at end of implementation file.
pub fn verify(comptime F: type) void {
    // Type info
    _ = F.MODULUS;
    _ = F.ENCODED_SIZE;

    // Constants
    _ = F.zero;
    _ = F.one;

    // Core arithmetic (scalar)
    _ = @as(fn (F, F) F, F.add);
    _ = @as(fn (F, F) F, F.sub);
    _ = @as(fn (F, F) F, F.mul);
    _ = @as(fn (F) F, F.neg);

    // Batch arithmetic
    _ = @as(fn ([]F, []const F, []const F) void, F.addBatch);
    _ = @as(fn ([]F, []const F, []const F) void, F.mulBatch);
    _ = @as(fn ([]F, []const F, []const F) void, F.subBatch);
    _ = @as(fn ([]F, []const F, []const F, []const F) void, F.mulAddBatch);
    _ = @as(fn ([]F, []const F, []const F, F) void, F.linearCombineBatch);

    // Batch reductions
    _ = @as(fn ([]const F) F, F.sumSlice);
    _ = @as(fn ([]const F, []const F) F, F.dotProduct);

    // Extended arithmetic
    _ = @as(fn (F) F, F.square);
    _ = @as(fn (F) F, F.inv);

    // Random sampling
    _ = @as(fn (std.Random) F, F.random);

    // Comparison
    _ = @as(fn (F, F) bool, F.eql);
    _ = @as(fn (F) bool, F.isZero);

    // Serialization
    _ = @as(fn (F) [F.ENCODED_SIZE]u8, F.toBytes);
    _ = @as(fn ([F.ENCODED_SIZE]u8) error{InvalidValue}!F, F.fromBytes);
    // TODO fromBytesBatch, toBytesBatch, fromBytesUnchecked (for scribe streaming trusted data)

    // TODO from/to U64, from/to U32? not sure
}

/// Provides default implementations for derived operations.
/// Usage: `pub const square = FieldDefaults(Self).square;`
pub fn defaults(comptime Self: type) type {
    return struct {
        pub fn square(a: Self) Self {
            return a.mul(a);
        }

        pub fn double(a: Self) Self {
            return a.add(a);
        }

        pub fn pow(base: Self, exp: u64) Self {
            if (exp == 0) return Self.one;

            var result = Self.one;
            var b = base;
            var e = exp;

            while (e > 0) {
                if (e & 1 == 1) result = result.mul(b);
                b = b.square();
                e >>= 1;
            }
            return result;
        }
    };
}

/// Generic field tests that verify field axioms.
/// Usage: `comptime { _ = field.tests(MyField); }`
pub fn tests(comptime F: type) type {
    return struct {
        // ============ Field Axiom Tests ============ //

        test "additive identity" {
            const a = F.fromU64(12345);
            try testing.expect(a.add(F.zero).eql(a));
            try testing.expect(F.zero.add(a).eql(a));
        }

        test "multiplicative identity" {
            const a = F.fromU64(12345);
            try testing.expect(a.mul(F.one).eql(a));
            try testing.expect(F.one.mul(a).eql(a));
        }

        test "additive inverse" {
            const a = F.fromU64(12345);
            try testing.expect(a.add(a.neg()).isZero());
            try testing.expect(a.neg().add(a).isZero());
        }

        test "multiplicative inverse" {
            const a = F.fromU64(12345);
            try testing.expect(a.mul(a.inv()).eql(F.one));
            try testing.expect(a.inv().mul(a).eql(F.one));
        }

        test "commutativity" {
            const a = F.fromU64(111);
            const b = F.fromU64(222);
            try testing.expect(a.add(b).eql(b.add(a)));
            try testing.expect(a.mul(b).eql(b.mul(a)));
        }

        test "associativity" {
            const a = F.fromU64(111);
            const b = F.fromU64(222);
            const c = F.fromU64(333);
            try testing.expect(a.add(b).add(c).eql(a.add(b.add(c))));
            try testing.expect(a.mul(b).mul(c).eql(a.mul(b.mul(c))));
        }

        test "distributivity" {
            const a = F.fromU64(111);
            const b = F.fromU64(222);
            const c = F.fromU64(333);
            const lhs = a.mul(b.add(c));
            const rhs = a.mul(b).add(a.mul(c));
            try testing.expect(lhs.eql(rhs));
        }

        // ============ Subtraction Consistency ============ //

        test "subtraction equals add neg" {
            const vals = [_]u64{ 0, 1, 2, 12345, 9999999 };
            for (vals) |a_raw| {
                for (vals) |b_raw| {
                    const a = F.fromU64(a_raw);
                    const b = F.fromU64(b_raw);
                    try testing.expect(a.sub(b).eql(a.add(b.neg())));
                }
            }
        }

        // ============ Serialization Tests ============ //

        test "serialization roundtrip" {
            const vals = [_]u64{ 0, 1, 12345, 9999999 };
            for (vals) |v| {
                const original = F.fromU64(v);
                const bytes = original.toBytes();
                const recovered = try F.fromBytes(bytes);
                try testing.expect(original.eql(recovered));
            }
        }

        // ============ Derived Operation Tests ============ //

        test "square matches mul" {
            const a = F.fromU64(12345);
            try testing.expect(a.square().eql(a.mul(a)));

            const b = F.fromU64(9999999);
            try testing.expect(b.square().eql(b.mul(b)));
        }

        test "double matches add" {
            const a = F.fromU64(12345);
            try testing.expect(a.double().eql(a.add(a)));

            const b = F.fromU64(9999999);
            try testing.expect(b.double().eql(b.add(b)));
        }

        test "pow edge cases" {
            const a = F.fromU64(12345);

            // a^0 = 1
            try testing.expect(a.pow(0).eql(F.one));

            // a^1 = a
            try testing.expect(a.pow(1).eql(a));

            // a^2 = a*a
            try testing.expect(a.pow(2).eql(a.mul(a)));

            // 0^n = 0 for n > 0
            try testing.expect(F.zero.pow(1).isZero());
            try testing.expect(F.zero.pow(100).isZero());

            // 1^n = 1
            try testing.expect(F.one.pow(0).eql(F.one));
            try testing.expect(F.one.pow(1).eql(F.one));
            try testing.expect(F.one.pow(1000000).eql(F.one));
        }
    };
}
