const std = @import("std");

const Blake3 = std.crypto.hash.Blake3;

pub fn Transcript(comptime F: type) type {
    return struct {
        const Self = @This();

        state: Blake3,

        /// Initialize with domain separator
        pub fn init(domain: []const u8) Self {
            var t = Self{ .state = Blake3.init(.{}) };
            t.state.update(domain);
            return t;
        }

        /// Add a field element to the transcript
        pub fn absorb(self: *Self, elem: F) void {
            self.state.update(&elem.toBytes());
        }

        /// Add raw bytes (for domain separation, commitments, etc)
        pub fn absorbBytes(self: *Self, bytes: []const u8) void {
            self.state.update(bytes);
        }

        /// Extract a challenge
        pub fn squeeze(self: *Self) F {
            while (true) {
                // Copy state so we can continue
                var output: [4]u8 = undefined;
                var state_copy = self.state;
                state_copy.final(&output);

                // Absorb output to chain state (so next squeeze differs)
                self.state.update(&output);

                // Use first ENCODED_SIZE bytes for field element
                const field_bytes: [F.ENCODED_SIZE]u8 = output[0..F.ENCODED_SIZE].*;

                // Try to construct field element (rejects if >= MODULUS)
                if (F.fromBytes(field_bytes)) |elem| {
                    return elem;
                } else |_| {
                    // Value was invalid, loop with updated state
                    continue;
                }
            }
        }
    };
}

const M31 = @import("fields/mersenne31.zig").Mersenne31;

test "transcript squeeze determinism" {
    const T = Transcript(M31);
    var t1 = T.init("test");
    var t2 = T.init("test");

    // Same seed, same sequence
    try std.testing.expect(t1.squeeze().eql(t2.squeeze()));
    try std.testing.expect(t1.squeeze().eql(t2.squeeze()));

    // Different domain, different output
    var t3 = T.init("other");
    try std.testing.expect(!t1.squeeze().eql(t3.squeeze()));
}
