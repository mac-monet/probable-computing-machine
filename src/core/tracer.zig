const std = @import("std");
const Allocator = std.mem.Allocator;

/// Tracer for recording span timing data.
/// Stores events in a ring buffer for later analysis.
pub const Tracer = struct {
    allocator: Allocator,
    events: std.ArrayListUnmanaged(Event),
    start_time: u64,

    pub const Event = struct {
        name: []const u8,
        start: u64,
        end: u64,
        bytes: ?usize,
    };

    pub fn init(allocator: Allocator) Tracer {
        return .{
            .allocator = allocator,
            .events = .{},
            .start_time = now_ns(),
        };
    }

    pub fn deinit(self: *Tracer) void {
        self.events.deinit(self.allocator);
    }

    pub fn now(self: *Tracer) u64 {
        _ = self;
        return now_ns();
    }

    pub fn record(self: *Tracer, name: []const u8, start: u64) void {
        self.recordWithSize(name, start, null);
    }

    pub fn recordWithSize(self: *Tracer, name: []const u8, start: u64, bytes: ?usize) void {
        self.events.append(self.allocator, .{
            .name = name,
            .start = start,
            .end = now_ns(),
            .bytes = bytes,
        }) catch {};
    }

    /// Reset for next run (keeps capacity)
    pub fn reset(self: *Tracer) void {
        self.events.clearRetainingCapacity();
        self.start_time = now_ns();
    }

    // TODO: Add output formats:
    // pub fn writeChrome(self: *Tracer, writer: anytype) !void { ... }
    // pub fn writeSummary(self: *Tracer, writer: anytype) !void { ... }
};

fn now_ns() u64 {
    return @intCast(std.time.nanoTimestamp());
}

// ============ Tests ============ //

test "Tracer basic usage" {
    var tracer = Tracer.init(std.testing.allocator);
    defer tracer.deinit();

    const start = tracer.now();
    // Small busy loop instead of sleep
    var x: u64 = 0;
    for (0..1000) |i| x += i;
    std.mem.doNotOptimizeAway(&x);
    tracer.record("test_span", start);

    try std.testing.expectEqual(@as(usize, 1), tracer.events.items.len);
    try std.testing.expectEqualStrings("test_span", tracer.events.items[0].name);
    try std.testing.expect(tracer.events.items[0].end >= tracer.events.items[0].start);
}

test "Tracer with size" {
    var tracer = Tracer.init(std.testing.allocator);
    defer tracer.deinit();

    const start = tracer.now();
    tracer.recordWithSize("io_read", start, 1024 * 1024);

    try std.testing.expectEqual(@as(usize, 1), tracer.events.items.len);
    try std.testing.expectEqual(@as(?usize, 1024 * 1024), tracer.events.items[0].bytes);
}

test "Tracer reset" {
    var tracer = Tracer.init(std.testing.allocator);
    defer tracer.deinit();

    tracer.record("span1", tracer.now());
    tracer.record("span2", tracer.now());
    try std.testing.expectEqual(@as(usize, 2), tracer.events.items.len);

    tracer.reset();
    try std.testing.expectEqual(@as(usize, 0), tracer.events.items.len);
}
