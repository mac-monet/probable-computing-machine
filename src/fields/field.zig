const std = @import("std");

// Finite Field interface

/// Common error type for field operations
pub const FieldError = error{
    InvalidValue,
};

/// Documents the required field interface.
/// Use: `comptime { verify(MyField); }` at end of implementation file.
pub fn verify(comptime F: type) void {
    const err = "Field '" ++ @typeName(F) ++ "' missing required ";

    // Type info
    if (!@hasDecl(F, "MODULUS")) @compileError(err ++ "MODULUS");
    if (!@hasDecl(F, "ENCODED_SIZE")) @compileError(err ++ "ENCODED_SIZE");

    // Constants
    if (!@hasDecl(F, "zero")) @compileError(err ++ "zero");
    if (!@hasDecl(F, "one")) @compileError(err ++ "one");

    // Core arithmetic
    if (!@hasDecl(F, "add")) @compileError(err ++ "add");
    if (!@hasDecl(F, "sub")) @compileError(err ++ "sub");
    if (!@hasDecl(F, "mul")) @compileError(err ++ "mul");
    if (!@hasDecl(F, "neg")) @compileError(err ++ "neg");

    // Extended arithmetic
    if (!@hasDecl(F, "square")) @compileError(err ++ "square");
    if (!@hasDecl(F, "inv")) @compileError(err ++ "inv");

    // Comparison
    if (!@hasDecl(F, "eql")) @compileError(err ++ "eql");
    if (!@hasDecl(F, "isZero")) @compileError(err ++ "isZero");

    // Serialization
    if (!@hasDecl(F, "toBytes")) @compileError(err ++ "toBytes");
    if (!@hasDecl(F, "fromBytes")) @compileError(err ++ "fromBytes");
    if (!@hasDecl(F, "fromU64")) @compileError(err ++ "fromU64");
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
                b = b.mul(b);
                e >>= 1;
            }
            return result;
        }
    };
}
