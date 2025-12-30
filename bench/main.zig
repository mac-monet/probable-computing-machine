const std = @import("std");
const multilinear = @import("multilinear.zig");
const fields = @import("fields.zig");

pub fn main() !void {
    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
    const stdout = &stdout_writer.interface;

    try stdout.print("\n=== libzero benchmarks ===\n\n", .{});

    try fields.run(stdout);
    try multilinear.run(stdout);

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
