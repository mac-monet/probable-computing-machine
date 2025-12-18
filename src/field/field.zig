const std = @import("std");

// Finite Field interface

/// Documents the required field interface.
/// Use: `comptime { verify(MyField); }` at end of implementation file.
pub fn verify(comptime F: type) void {
    const errors = comptime blk: {
        var errs: []const u8 = "";

        if (!@hasDecl(F, "MODULUS")) errs = errs ++ "missing MODULUS\n";
        if (!@hasDecl(F, "zero")) errs = errs ++ "missing zero\n";
        if (!@hasDecl(F, "one")) errs = errs ++ "missing one\n";
        if (!@hasDecl(F, "add")) errs = errs ++ "missing add\n";
        if (!@hasDecl(F, "sub")) errs = errs ++ "missing sub\n";
        if (!@hasDecl(F, "mul")) errs = errs ++ "missing mul\n";
        if (!@hasDecl(F, "neg")) errs = errs ++ "missing neg\n";
        if (!@hasDecl(F, "inv")) errs = errs ++ "missing inv\n";
        if (!@hasDecl(F, "eql")) errs = errs ++ "missing eql\n";
        if (!@hasDecl(F, "isZero")) errs = errs ++ "missing isZero\n";
        if (!@hasDecl(F, "toBytes")) errs = errs ++ "missing toBytes\n";
        if (!@hasDecl(F, "fromBytes")) errs = errs ++ "missing fromBytes\n";
        if (!@hasDecl(F, "fromU64")) errs = errs ++ "missing fromU64\n";

        break :blk errs;
    };

    if (errors.len > 0) {
        @compileError("Field interface not satisfied:\n" ++ errors);
    }
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
