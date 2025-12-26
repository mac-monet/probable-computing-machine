const std = @import("std");
const field = @import("fields/field.zig");

pub fn MultilinearPoly(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Evaluations over the boolean hypercube
        /// evals[i] = P(b₀, b₁, ..., bₙ₋₁) where i = Σ bⱼ·2ʲ
        evals: []F,

        num_vars: usize,

        /// Evaluate at an arbitrary given point
        pub fn evaluate(self: *const Self, point: []const F) F {
            // TODO
            self;
            point;
        }

        pub fn bind(self: *Self, r: F) void {}
    };
}
