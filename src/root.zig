//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

// Re-export field types
pub const Mersenne31 = @import("fields/mersenne31.zig").Mersenne31;

// Re-export polynomial modules
pub const multilinear = @import("poly/multilinear.zig");

// Re-export protocols
pub const sumcheck = @import("iop/sumcheck.zig");
pub const Transcript = @import("core/transcript.zig").Transcript;

// Re-export merkle tree
pub const merkle = @import("merkle/root.zig");

// Re-export constraint system
pub const constraint = @import("constraint/constraint.zig");

// Re-export trace system
pub const trace = @import("trace/trace.zig");

// Re-export VM modules
pub const vm = struct {
    pub const stack = @import("vm/stack.zig");
    pub const trace = @import("vm/trace.zig");
    pub const constraints = @import("vm/constraints.zig");
    pub const prover = @import("vm/prover.zig");
    pub const verifier = @import("vm/verifier.zig");
};

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
    _ = @import("fields/mersenne31.zig");
    _ = @import("poly/multilinear.zig");
    _ = @import("poly/eq.zig");
    _ = @import("core/transcript.zig");
    _ = @import("merkle/root.zig");
    _ = @import("pcs/basefold.zig");
    _ = @import("vm/stack.zig");
    _ = @import("vm/trace.zig");
    _ = @import("vm/constraints.zig");
    _ = @import("vm/prover.zig");
    _ = @import("vm/verifier.zig");
    _ = @import("vm/proof_test.zig");

    // New architecture
    _ = @import("core/tracer.zig");
    _ = @import("core/context.zig");
    _ = @import("iop/sumcheck.zig");
    _ = @import("pcs/pcs.zig");

    // Constraint system
    _ = @import("constraint/constraint.zig");

    // Trace system
    _ = @import("trace/trace.zig");
}
