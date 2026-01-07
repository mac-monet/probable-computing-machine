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

    // Batch operations
    _ = @as(fn ([]F, []const F, []const F, F) void, F.linearCombineBatch);
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

// ============ Extension Field Interface ============ //

/// Extension field interface and generic constructor.
pub const ExtField = struct {
    /// Verify that a type implements the extension field interface.
    /// Use: `comptime { ExtField.verify(MyExt); }` at end of implementation.
    pub fn verify(comptime E: type) void {
        // Must have base field reference and degree
        _ = E.BaseField;
        _ = E.DEGREE;

        // Extension-specific operations
        _ = @as(fn (E.BaseField) E, E.fromBase);
        _ = @as(fn (E, E.BaseField) E, E.mulBase);

        // Standard field operations
        _ = E.zero;
        _ = E.one;
        _ = @as(fn (E, E) E, E.add);
        _ = @as(fn (E, E) E, E.sub);
        _ = @as(fn (E, E) E, E.mul);
        _ = @as(fn (E) E, E.neg);
        _ = @as(fn (E, E) bool, E.eql);
        _ = @as(fn (E) bool, E.isZero);
    }

    /// Construct an extension field from a configuration type.
    ///
    /// Config must provide:
    /// - `Base`: the base field type
    /// - `degree`: extension degree (comptime_int)
    /// - `reduce(*const [2*degree-1]Base) [degree]Base`: reduction mod irreducible
    pub fn init(comptime Config: type) type {
        const Base = Config.Base;
        const degree = Config.degree;

        return struct {
            const Self = @This();

            coeffs: [degree]Base,

            pub const BaseField = Base;
            pub const DEGREE = degree;

            pub const zero: Self = .{ .coeffs = [_]Base{Base.zero} ** degree };
            pub const one: Self = blk: {
                var c = [_]Base{Base.zero} ** degree;
                c[0] = Base.one;
                break :blk .{ .coeffs = c };
            };

            /// Embed base field element: a -> (a, 0, 0, ...)
            pub fn fromBase(a: Base) Self {
                var c = [_]Base{Base.zero} ** degree;
                c[0] = a;
                return .{ .coeffs = c };
            }

            /// Multiply extension element by base field element.
            /// O(degree) base field multiplications.
            pub fn mulBase(self: Self, b: Base) Self {
                var result: [degree]Base = undefined;
                inline for (0..degree) |i| {
                    result[i] = self.coeffs[i].mul(b);
                }
                return .{ .coeffs = result };
            }

            pub fn add(a: Self, b: Self) Self {
                var result: [degree]Base = undefined;
                inline for (0..degree) |i| {
                    result[i] = a.coeffs[i].add(b.coeffs[i]);
                }
                return .{ .coeffs = result };
            }

            pub fn sub(a: Self, b: Self) Self {
                var result: [degree]Base = undefined;
                inline for (0..degree) |i| {
                    result[i] = a.coeffs[i].sub(b.coeffs[i]);
                }
                return .{ .coeffs = result };
            }

            /// Polynomial multiplication with reduction.
            /// O(degree²) base field multiplications.
            pub fn mul(a: Self, b: Self) Self {
                // Schoolbook polynomial multiplication
                var product: [2 * degree - 1]Base = [_]Base{Base.zero} ** (2 * degree - 1);

                inline for (0..degree) |i| {
                    inline for (0..degree) |j| {
                        product[i + j] = product[i + j].add(a.coeffs[i].mul(b.coeffs[j]));
                    }
                }

                // Reduce mod irreducible polynomial
                return .{ .coeffs = Config.reduce(&product) };
            }

            pub fn neg(a: Self) Self {
                var result: [degree]Base = undefined;
                inline for (0..degree) |i| {
                    result[i] = a.coeffs[i].neg();
                }
                return .{ .coeffs = result };
            }

            pub fn eql(a: Self, b: Self) bool {
                inline for (0..degree) |i| {
                    if (!a.coeffs[i].eql(b.coeffs[i])) return false;
                }
                return true;
            }

            pub fn isZero(a: Self) bool {
                inline for (0..degree) |i| {
                    if (!a.coeffs[i].isZero()) return false;
                }
                return true;
            }

            pub fn square(a: Self) Self {
                return a.mul(a);
            }

            /// Multiplicative inverse using norm-based approach.
            /// For degree-2: a^(-1) = conjugate(a) / norm(a)
            pub fn inv(a: Self) Self {
                std.debug.assert(!a.isZero());
                if (degree == 2) {
                    // For x^2 = W: (c0 + c1*x)^(-1) = (c0 - c1*x) / (c0^2 - W*c1^2)
                    const c0 = a.coeffs[0];
                    const c1 = a.coeffs[1];
                    const w = Base.fromU64(Config.W);
                    const norm = c0.mul(c0).sub(w.mul(c1.mul(c1)));
                    const norm_inv = norm.inv();
                    return .{ .coeffs = .{
                        c0.mul(norm_inv),
                        c1.neg().mul(norm_inv),
                    } };
                } else {
                    // General case: use Fermat's little theorem
                    // a^(-1) = a^(p^degree - 2) - expensive but correct
                    @compileError("inv() for degree > 2 not yet implemented");
                }
            }

            /// Random sampling from the extension field.
            pub fn random(rng: std.Random) Self {
                var result: [degree]Base = undefined;
                inline for (0..degree) |i| {
                    result[i] = Base.random(rng);
                }
                return .{ .coeffs = result };
            }

            /// SoA (Struct of Arrays) layout for SIMD-friendly batch operations.
            /// Each coefficient stream is stored separately for cache efficiency.
            pub const Batch = struct {
                /// coeffs[i] contains all i-th coefficients across elements.
                /// Length of each slice is the batch size.
                coeffs: [degree][]Base,

                /// Transpose from AoS ([]Ext) to SoA (Batch).
                /// Caller provides storage for coefficient slices.
                pub fn fromSlice(exts: []const Self, storage: *[degree][]Base) Batch {
                    const len = exts.len;
                    inline for (0..degree) |d| {
                        for (0..len) |i| {
                            storage[d][i] = exts[i].coeffs[d];
                        }
                    }
                    return .{ .coeffs = storage.* };
                }

                /// Dot product with base field using SIMD-friendly layout.
                /// Reuses Base.dotProduct for each coefficient stream.
                pub fn dotProductMixed(self: Batch, bases: []const Base) Self {
                    std.debug.assert(self.coeffs[0].len == bases.len);
                    var result: [degree]Base = undefined;
                    inline for (0..degree) |d| {
                        result[d] = Base.dotProduct(bases, self.coeffs[d]);
                    }
                    return .{ .coeffs = result };
                }
            };
        };
    }

    /// Default implementations for extension field derived operations.
    pub fn defaults(comptime Self: type) type {
        return struct {
            pub fn square(a: Self) Self {
                return a.mul(a);
            }

            pub fn double(a: Self) Self {
                return a.add(a);
            }
        };
    }

    /// Generic extension field tests.
    pub fn tests(comptime E: type) type {
        return struct {
            test "extension: additive identity" {
                const a = E.fromBase(E.BaseField.fromU64(12345));
                try testing.expect(a.add(E.zero).eql(a));
                try testing.expect(E.zero.add(a).eql(a));
            }

            test "extension: multiplicative identity" {
                const a = E.fromBase(E.BaseField.fromU64(12345));
                try testing.expect(a.mul(E.one).eql(a));
                try testing.expect(E.one.mul(a).eql(a));
            }

            test "extension: additive inverse" {
                const a = E.fromBase(E.BaseField.fromU64(12345));
                try testing.expect(a.add(a.neg()).isZero());
            }

            test "extension: multiplicative inverse" {
                const a = E.fromBase(E.BaseField.fromU64(12345));
                try testing.expect(a.mul(a.inv()).eql(E.one));
            }

            test "extension: commutativity" {
                const a = E.fromBase(E.BaseField.fromU64(111));
                const b = E.fromBase(E.BaseField.fromU64(222));
                try testing.expect(a.add(b).eql(b.add(a)));
                try testing.expect(a.mul(b).eql(b.mul(a)));
            }

            test "extension: associativity" {
                const a = E.fromBase(E.BaseField.fromU64(111));
                const b = E.fromBase(E.BaseField.fromU64(222));
                const c = E.fromBase(E.BaseField.fromU64(333));
                try testing.expect(a.add(b).add(c).eql(a.add(b.add(c))));
                try testing.expect(a.mul(b).mul(c).eql(a.mul(b.mul(c))));
            }

            test "extension: distributivity" {
                const a = E.fromBase(E.BaseField.fromU64(111));
                const b = E.fromBase(E.BaseField.fromU64(222));
                const c = E.fromBase(E.BaseField.fromU64(333));
                const lhs = a.mul(b.add(c));
                const rhs = a.mul(b).add(a.mul(c));
                try testing.expect(lhs.eql(rhs));
            }

            test "extension: fromBase embedding" {
                const base_val = E.BaseField.fromU64(42);
                const ext_val = E.fromBase(base_val);
                // First coefficient should equal base value
                try testing.expect(ext_val.coeffs[0].eql(base_val));
                // Rest should be zero
                inline for (1..E.DEGREE) |i| {
                    try testing.expect(ext_val.coeffs[i].isZero());
                }
            }

            test "extension: mulBase matches mul(fromBase)" {
                const base_a = E.BaseField.fromU64(123);
                const base_b = E.BaseField.fromU64(456);
                const ext_a = E.fromBase(base_a);

                const via_mulBase = ext_a.mulBase(base_b);
                const via_mul = ext_a.mul(E.fromBase(base_b));

                try testing.expect(via_mulBase.eql(via_mul));
            }
        };
    }
};

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
