const std = @import("std");
const bench = @import("main.zig");
const libzero = @import("libzero");
const MultilinearPoly = libzero.MultilinearPoly;
const Mersenne31 = libzero.Mersenne31;

const WARMUP = 10;
const ITERS = 1_000;

pub fn run(writer: anytype) !void {
    try writer.print("\n--- Multilinear Polynomial ---\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Benchmark different sizes
    inline for ([_]usize{ 16, 20 }) |num_vars| {
        try benchSize(Mersenne31, num_vars, allocator, writer);
    }
}

fn benchSize(comptime F: type, comptime num_vars: usize, allocator: std.mem.Allocator, writer: anytype) !void {
    const size = @as(usize, 1) << num_vars;
    try writer.print("\nMultilinear n={d} (2^{d} = {d} evals):\n", .{ num_vars, num_vars, size });

    var poly = try MultilinearPoly(F).init(allocator, num_vars);
    defer poly.deinit(allocator);

    // Fill with some values
    for (poly.evals, 0..) |*e, i| {
        e.* = F.fromU64(@truncate(i));
    }

    // Create a random-ish point for evaluation
    var point: [num_vars]F = undefined;
    for (&point, 0..) |*p, i| {
        p.* = F.fromU64(@truncate(i * 7 + 13));
    }

    // Allocate destination for bind
    const bind_dst = try allocator.alloc(F, size / 2);
    defer allocator.free(bind_dst);

    const label_prefix = std.fmt.comptimePrint("n={d} ", .{num_vars});

    try bench.benchmark(
        label_prefix ++ "sum",
        WARMUP,
        ITERS,
        MultilinearPoly(F).sum,
        .{&poly},
        writer,
    );

    try bench.benchmark(
        label_prefix ++ "sumHalves",
        WARMUP,
        ITERS,
        MultilinearPoly(F).sumHalves,
        .{&poly},
        writer,
    );

    try bench.benchmark(
        label_prefix ++ "evaluate",
        WARMUP,
        ITERS / 10,
        MultilinearPoly(F).evaluate,
        .{ &poly, &point },
        writer,
    );

    // Wrap bind to match benchmark signature (returns void)
    const BindWrapper = struct {
        fn call(p: *MultilinearPoly(F), r: F, dst: []F) void {
            p.bind(r, dst);
        }
    };

    try bench.benchmark(
        label_prefix ++ "bind",
        WARMUP,
        ITERS,
        BindWrapper.call,
        .{ &poly, F.fromU64(42), bind_dst },
        writer,
    );
}
