//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

pub const Mersenne31 = @import("fields/mersenne31.zig").Mersenne31;
pub const multilinear = @import("poly/multilinear.zig");
pub const sumcheck = @import("iop/sumcheck.zig");
pub const zerocheck = @import("iop/zerocheck.zig");
pub const Transcript = @import("core/transcript.zig").Transcript;
pub const merkle = @import("merkle/root.zig");
pub const constraint = @import("constraint/constraint.zig");
pub const trace = @import("constraint/trace.zig");

test {
    // Import modules to include their tests
    _ = @import("fields/mersenne31.zig");
    _ = @import("poly/multilinear.zig");
    _ = @import("poly/eq.zig");
    _ = @import("core/transcript.zig");
    _ = @import("merkle/root.zig");
    _ = @import("pcs/basefold.zig");

    // Core architecture
    _ = @import("core/tracer.zig");
    _ = @import("core/context.zig");
    _ = @import("iop/sumcheck.zig");
    _ = @import("iop/zerocheck.zig");
    _ = @import("pcs/pcs.zig");

    // Constraint system
    _ = @import("constraint/constraint.zig");
    _ = @import("constraint/trace.zig");
    _ = @import("constraint/builder.zig");
}
