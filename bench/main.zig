const std = @import("std");
const multilinear = @import("multilinear.zig");
const fields = @import("fields.zig");
const sumcheck = @import("sumcheck.zig");

pub fn main() !void {
    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
    const stdout = &stdout_writer.interface;

    try stdout.print("\n=== libzero benchmarks ===\n\n", .{});

    try fields.run(stdout);
    try multilinear.run(stdout);
    try sumcheck.run(stdout);

    try stdout.print("\n=== done ===\n", .{});
    try stdout.flush();
}

// Benchmark utilities
pub const Timer = struct {
    start: std.time.Instant,

    pub fn begin() Timer {
        return .{ .start = std.time.Instant.now() catch unreachable };
    }

    pub fn elapsed(self: Timer) u64 {
        const now = std.time.Instant.now() catch unreachable;
        return now.since(self.start);
    }

    pub fn elapsedMs(self: Timer) f64 {
        return @as(f64, @floatFromInt(self.elapsed())) / 1_000_000.0;
    }
};

/// Sink to prevent optimizing away benchmark results
var sink: usize = 0;

/// Prevent the compiler from optimizing away a value
pub noinline fn doNotOptimize(val: anytype) void {
    sink +%= @intFromPtr(&val);
}

pub fn benchmark(
    comptime name: []const u8,
    comptime warmup_iters: usize,
    comptime bench_iters: usize,
    comptime func: anytype,
    args: anytype,
    writer: anytype,
) !void {
    // Warmup
    for (0..warmup_iters) |_| {
        const result = @call(.never_inline, func, args);
        doNotOptimize(result);
    }

    // Benchmark
    const timer = Timer.begin();
    for (0..bench_iters) |_| {
        const result = @call(.never_inline, func, args);
        doNotOptimize(result);
    }
    const total_ns = timer.elapsed();

    const ns_per_op = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(bench_iters));
    const ops_per_sec = 1_000_000_000.0 / ns_per_op;

    // Auto-scale units for readability
    const scaled = scaleTime(ns_per_op);

    try writer.print("  {s:<40} {d:>10.2} {s}  ({d:>10.0} ops/sec)\n", .{
        name,
        scaled.value,
        scaled.unit,
        ops_per_sec,
    });
}

const ScaledTime = struct {
    value: f64,
    unit: []const u8,
};

fn scaleTime(ns: f64) ScaledTime {
    if (ns >= 1_000_000_000.0) {
        return .{ .value = ns / 1_000_000_000.0, .unit = "s/op " };
    } else if (ns >= 1_000_000.0) {
        return .{ .value = ns / 1_000_000.0, .unit = "ms/op" };
    } else if (ns >= 1_000.0) {
        return .{ .value = ns / 1_000.0, .unit = "µs/op" };
    } else {
        return .{ .value = ns, .unit = "ns/op" };
    }
}

// Assumed CPU frequency for cycle estimation (Apple Silicon M1/M2/M3 perf cores)
const CPU_GHZ: f64 = 3.5;

/// Memory-aware benchmark with bandwidth and cache metrics
pub fn benchmarkMem(
    comptime name: []const u8,
    comptime warmup_iters: usize,
    comptime bench_iters: usize,
    comptime func: anytype,
    args: anytype,
    bytes_per_op: usize,
    writer: anytype,
) !void {
    // Warmup
    for (0..warmup_iters) |_| {
        const result = @call(.never_inline, func, args);
        doNotOptimize(result);
    }

    // Benchmark
    const timer = Timer.begin();
    for (0..bench_iters) |_| {
        const result = @call(.never_inline, func, args);
        doNotOptimize(result);
    }
    const total_ns = timer.elapsed();

    const ns_per_op = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(bench_iters));
    const bytes_f = @as(f64, @floatFromInt(bytes_per_op));

    // Memory metrics
    const bandwidth_gbs = bytes_f / ns_per_op; // GB/s (bytes/ns = GB/s)
    const cycles_per_op = ns_per_op * CPU_GHZ;
    const cycles_per_byte = cycles_per_op / bytes_f;

    // Auto-scale time units
    const scaled = scaleTime(ns_per_op);

    try writer.print("  {s:<32} {d:>8.2} {s}  {d:>6.1} GB/s  {d:>5.2} cyc/B\n", .{
        name,
        scaled.value,
        scaled.unit,
        bandwidth_gbs,
        cycles_per_byte,
    });
}

/// Print cache size context for a given data size
pub fn printCacheContext(bytes: usize, writer: anytype) !void {
    const kb = bytes / 1024;
    const mb = bytes / (1024 * 1024);

    // Typical Apple Silicon cache sizes:
    // L1: 128KB per perf core (64KB I + 64KB D)
    // L2: 12-16MB shared
    const cache_level = if (bytes <= 64 * 1024)
        "L1"
    else if (bytes <= 4 * 1024 * 1024)
        "L2"
    else
        "RAM";

    if (mb > 0) {
        try writer.print("  Data size: {d} MB ({s}-resident)\n", .{ mb, cache_level });
    } else {
        try writer.print("  Data size: {d} KB ({s}-resident)\n", .{ kb, cache_level });
    }
}
