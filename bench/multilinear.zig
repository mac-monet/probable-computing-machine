const std = @import("std");
const bench = @import("main.zig");
const libzero = @import("libzero");
const multilinear = libzero.multilinear;
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

    // Allocate eval buffer (caller manages storage)
    const evals = try allocator.alloc(F, size);
    defer allocator.free(evals);

    // Fill with some values
    for (evals, 0..) |*e, i| {
        e.* = F.fromU64(@truncate(i));
    }

    // Create a random-ish point for evaluation
    var point: [num_vars]F = undefined;
    for (&point, 0..) |*p, i| {
        p.* = F.fromU64(@truncate(i * 7 + 13));
    }

    const label_prefix = std.fmt.comptimePrint("n={d} ", .{num_vars});

    // Sum - single slice (no interleaving difference)
    const SumWrapper = struct {
        fn call(e: []const F) F {
            return multilinear.sum(F, e);
        }
    };

    try bench.benchmark(
        label_prefix ++ "sum",
        WARMUP,
        ITERS,
        SumWrapper.call,
        .{evals},
        writer,
    );

    // sumHalves
    const SumHalvesWrapper = struct {
        fn call(e: []const F) [2]F {
            return multilinear.sumHalves(F, e);
        }
    };

    try bench.benchmark(
        label_prefix ++ "sumHalves",
        WARMUP,
        ITERS,
        SumHalvesWrapper.call,
        .{evals},
        writer,
    );

    // Benchmark full evaluation: copy + bind all variables
    const EvalWrapper = struct {
        fn call(original_evals: []const F, pt: *const [num_vars]F, scratch: []F) F {
            @memcpy(scratch, original_evals);
            return multilinear.evaluate(F, scratch, pt);
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
        .{ evals, &point, scratch },
        writer,
    );

    // Benchmark single bind (need to reset each iteration)
    const BindWrapper = struct {
        fn call(original_evals: []const F, r: F, scratch_buf: []F) []F {
            @memcpy(scratch_buf, original_evals);
            return multilinear.bind(F, scratch_buf, r);
        }
    };

    try bench.benchmark(
        label_prefix ++ "bind",
        WARMUP,
        ITERS,
        BindWrapper.call,
        .{ evals, F.fromU64(42), scratch },
        writer,
    );
}
