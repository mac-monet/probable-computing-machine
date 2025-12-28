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
    inline for ([_]usize{ 16, 20, 24 }) |num_vars| {
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

    // Benchmark full evaluation: clone + bind all variables
    const EvalWrapper = struct {
        fn call(original_evals: []const F, pt: *const [num_vars]F, scratch: []F) F {
            @memcpy(scratch, original_evals);
            var current_size = original_evals.len;
            for (pt) |r| {
                const half = current_size / 2;
                F.linearCombineBatch(scratch[0..half], scratch[0..half], scratch[half..current_size], r);
                current_size = half;
            }
            return scratch[0];
        }
    };

    // Allocate scratch buffer for evaluate benchmark
    const scratch = try allocator.alloc(F, size);
    defer allocator.free(scratch);

    try bench.benchmark(
        label_prefix ++ "evaluate",
        WARMUP,
        ITERS / 10,
        EvalWrapper.call,
        .{ poly.evals[0..size], &point, scratch },
        writer,
    );

    // Benchmark bind (in-place, so we need to reset poly each iteration)
    const BindWrapper = struct {
        fn call(p: *MultilinearPoly(F), r: F, original_evals: []F) void {
            // Reset poly to original state before bind
            p.evals = original_evals;
            p.num_vars = num_vars;
            p.bind(r);
        }
    };

    try bench.benchmark(
        label_prefix ++ "bind",
        WARMUP,
        ITERS,
        BindWrapper.call,
        .{ &poly, F.fromU64(42), poly.evals },
        writer,
    );
}
