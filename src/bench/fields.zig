const std = @import("std");
const bench = @import("main.zig");
const libzero = @import("libzero");
const Mersenne31 = libzero.Mersenne31;

const WARMUP = 100;
const ITERS = 100_000;

pub fn run(writer: anytype) !void {
    try writer.print("--- Field Arithmetic ---\n", .{});

    try benchField(Mersenne31, "Mersenne31", writer);
}

fn benchField(comptime F: type, comptime name: []const u8, writer: anytype) !void {
    try writer.print("\n{s}:\n", .{name});

    // Setup: create some random-ish values (use smaller constants that fit in all field types)
    const a = F.fromU64(0x12345678 % (F.MODULUS - 1));
    const b = F.fromU64(0x87654321 % (F.MODULUS - 1));

    try bench.benchmark(name ++ " add", WARMUP, ITERS, F.add, .{ a, b }, writer);
    try bench.benchmark(name ++ " sub", WARMUP, ITERS, F.sub, .{ a, b }, writer);
    try bench.benchmark(name ++ " mul", WARMUP, ITERS, F.mul, .{ a, b }, writer);
    try bench.benchmark(name ++ " square", WARMUP, ITERS, F.square, .{a}, writer);

    // Fewer iterations for expensive ops
    try bench.benchmark(name ++ " inv", WARMUP, 1_000, F.inv, .{a}, writer);
}
