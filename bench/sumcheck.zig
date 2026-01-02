const std = @import("std");
const bench = @import("main.zig");
const libzero = @import("libzero");
const Sumcheck = libzero.Sumcheck;
const Transcript = libzero.Transcript;
const Mersenne31 = libzero.Mersenne31;

const WARMUP = 10;
const ITERS = 1_000;

pub fn run(writer: anytype) !void {
    try writer.print("\n--- Sumcheck Protocol (memory-aware) ---\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    inline for ([_]usize{ 16, 20, 24 }) |num_vars| {
        try benchSize(Mersenne31, num_vars, allocator, writer);
    }
}

fn benchSize(comptime F: type, comptime num_vars: usize, allocator: std.mem.Allocator, writer: anytype) !void {
    const size = @as(usize, 1) << num_vars;
    const elem_bytes = @sizeOf(F);
    const data_bytes = size * elem_bytes;

    try writer.print("\nSumcheck n={d} (2^{d} = {d} evals):\n", .{ num_vars, num_vars, size });
    try bench.printCacheContext(data_bytes, writer);

    // Allocate eval buffer
    const evals = try allocator.alloc(F, size);
    defer allocator.free(evals);

    // Scratch buffer for proving (gets mutated)
    const scratch = try allocator.alloc(F, size);
    defer allocator.free(scratch);

    // Fill with deterministic values
    for (evals, 0..) |*e, i| {
        e.* = F.fromU64(@truncate(i *% 17 +% 3));
    }

    // Round polynomials output
    var rounds: [num_vars]Sumcheck(F).RoundPoly = undefined;

    const label_prefix = std.fmt.comptimePrint("n={d} ", .{num_vars});

    // Benchmark prover
    const ProveWrapper = struct {
        fn call(
            original_evals: []const F,
            scratch_buf: []F,
            rnd: *[num_vars]Sumcheck(F).RoundPoly,
        ) F {
            @memcpy(scratch_buf, original_evals);
            var transcript = Transcript(F).init("bench");
            return Sumcheck(F).prove(scratch_buf, rnd, &transcript);
        }
    };

    // Fewer iterations for larger sizes
    const iters = if (num_vars >= 24) ITERS / 10 else ITERS;

    // Memory touched per prove:
    // - memcpy: read + write full array = 2 * size
    // - sumHalves across all rounds: reads ~2 * size total (geometric series)
    // - bind across all rounds: reads + writes ~3 * size total
    // Conservative estimate: ~4 * size * elem_bytes
    const prove_bytes = data_bytes * 4;

    try bench.benchmarkMem(
        label_prefix ++ "prove",
        WARMUP,
        iters,
        ProveWrapper.call,
        .{ evals, scratch, &rounds },
        prove_bytes,
        writer,
    );

    // Run one prove to get valid rounds for verify benchmark
    @memcpy(scratch, evals);
    var prover_transcript = Transcript(F).init("bench");
    _ = Sumcheck(F).prove(scratch, &rounds, &prover_transcript);

    // Compute claimed sum for verification
    const claimed_sum = blk: {
        var sum = F.zero;
        for (evals) |e| {
            sum = sum.add(e);
        }
        break :blk sum;
    };

    // Benchmark verifier
    const VerifyWrapper = struct {
        fn call(
            claim: F,
            rnd: *const [num_vars]Sumcheck(F).RoundPoly,
        ) F {
            var transcript = Transcript(F).init("bench");
            var challenges: [num_vars]F = undefined;
            return Sumcheck(F).verify(claim, rnd, &transcript, &challenges) catch unreachable;
        }
    };

    // Verifier memory: just reads rounds array (small)
    const verify_bytes = num_vars * @sizeOf(Sumcheck(F).RoundPoly);

    try bench.benchmarkMem(
        label_prefix ++ "verify",
        WARMUP,
        ITERS * 10,
        VerifyWrapper.call,
        .{ claimed_sum, &rounds },
        verify_bytes,
        writer,
    );
}
