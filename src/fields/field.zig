const std = @import("std");

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

    // Core arithmetic
    _ = @as(fn (F, F) F, F.add);
    _ = @as(fn (F, F) F, F.sub);
    _ = @as(fn (F, F) F, F.mul);
    _ = @as(fn (F) F, F.neg);
    _ = @as(fn ([]F, []F, []F) void, F.addBatch);
    _ = @as(fn ([]F, []F, []F) void, F.mulBatch);
    _ = @as(fn ([]F, []F, []F) void, F.subBatch);
    _ = @as(fn ([]F, []F, []F, []F) void, F.mulAddBatch);

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
