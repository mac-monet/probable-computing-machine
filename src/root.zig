//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

// Re-export field types
pub const Goldilocks = @import("fields/goldilocks.zig").Goldilocks;
pub const Mersenne31 = @import("fields/mersenne31.zig").Mersenne31;

// Re-export polynomial types
pub const MultilinearPoly = @import("poly/multilinear.zig").MultilinearPoly;

pub fn bufferedPrint() !void {
    // Stdout is for the actual output of your application, for example if you
    // are implementing gzip, then only the compressed bytes should be sent to
    // stdout, not any debugging messages.
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("Run `zig build test` to run the tests.\n", .{});

    try stdout.flush(); // Don't forget to flush!
}

test {
    // Import modules to include their tests
    _ = @import("fields/goldilocks.zig");
    _ = @import("fields/mersenne31.zig");
    _ = @import("poly/multilinear.zig");
}
