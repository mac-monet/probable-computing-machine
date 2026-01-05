const std = @import("std");

const Blake3 = std.crypto.hash.Blake3;

/// Fiat-Shamir transcript using Blake3 in XOF mode.
/// Optimized to extract multiple challenges per hash invocation.
pub fn Transcript(comptime F: type) type {
    // Buffer holds 8 challenges worth of bytes, regardless of field size
    // This gives ~8x reduction in hash operations for any field
    const CHALLENGES_PER_BUFFER = 8;
    const BUFFER_SIZE = F.ENCODED_SIZE * CHALLENGES_PER_BUFFER;

    return struct {
        const Self = @This();

        state: Blake3,
        buffer: [BUFFER_SIZE]u8 = undefined,
        buffer_pos: usize = BUFFER_SIZE,

        /// Initialize with domain separator
        pub fn init(domain: []const u8) Self {
            var t = Self{ .state = Blake3.init(.{}) };
            t.state.update(domain);
            return t;
        }

        /// Add a field element to the transcript
        pub fn absorb(self: *Self, elem: F) void {
            // Absorbing invalidates the buffer
            self.buffer_pos = BUFFER_SIZE;
            self.state.update(&elem.toBytes());
        }

        /// Add raw bytes (for domain separation, commitments, etc)
        pub fn absorbBytes(self: *Self, bytes: []const u8) void {
            self.buffer_pos = BUFFER_SIZE;
            self.state.update(bytes);
        }

        /// Extract a challenge using XOF mode with buffering.
        /// ~8x fewer hash operations than naive approach.
        pub fn squeeze(self: *Self) F {
            if (self.buffer_pos >= BUFFER_SIZE) {
                self.refillBuffer();
            }

            // Read field element bytes from buffer
            const bytes: *const [F.ENCODED_SIZE]u8 = @ptrCast(self.buffer[self.buffer_pos..][0..F.ENCODED_SIZE]);
            self.buffer_pos += F.ENCODED_SIZE;

            // Convert to field element using modular reduction (no rejection sampling)
            // Slight bias (~2^-field_bits), acceptable for cryptographic applications
            return F.fromU32(std.mem.readInt(u32, bytes, .little));
        }

        /// Squeeze multiple challenges at once (more efficient for batched use)
        pub fn squeezeN(self: *Self, comptime N: usize) [N]F {
            var result: [N]F = undefined;
            for (&result) |*r| {
                r.* = self.squeeze();
            }
            return result;
        }

        /// Squeeze challenges into a slice (for runtime-sized output)
        pub fn squeezeChallenges(self: *Self, out: []F) void {
            for (out) |*r| {
                r.* = self.squeeze();
            }
        }

        /// Derive a random evaluation point from a fresh transcript.
        /// Creates new transcript with domain, absorbs num_vars, squeezes challenges.
        pub fn derivePoint(domain: []const u8, out: []F) void {
            var t = Self.init(domain);
            t.absorb(F.fromU32(@intCast(out.len)));
            t.squeezeChallenges(out);
        }

        /// Refill the buffer using Blake3 XOF
        fn refillBuffer(self: *Self) void {
            // Copy state to preserve it for potential future absorbs
            var state_copy = self.state;
            state_copy.final(&self.buffer);

            // Chain: absorb the output so next refill differs
            self.state.update(&self.buffer);
            self.buffer_pos = 0;
        }
    };
}

const M31 = @import("../fields/mersenne31.zig").Mersenne31;

test "transcript squeeze determinism" {
    const T = Transcript(M31);
    var t1 = T.init("test");
    var t2 = T.init("test");

    // Same seed, same sequence
    try std.testing.expect(t1.squeeze().eql(t2.squeeze()));
    try std.testing.expect(t1.squeeze().eql(t2.squeeze()));

    // Different domain, different output
    var t3 = T.init("other");
    var t1_new = T.init("test");
    try std.testing.expect(!t1_new.squeeze().eql(t3.squeeze()));
}

test "transcript absorb resets buffer" {
    const T = Transcript(M31);
    var t1 = T.init("test");
    var t2 = T.init("test");

    // Squeeze some challenges
    _ = t1.squeeze();
    _ = t1.squeeze();
    _ = t2.squeeze();
    _ = t2.squeeze();

    // Absorb same value
    t1.absorb(M31.fromU64(42));
    t2.absorb(M31.fromU64(42));

    // Should produce same output after absorb
    try std.testing.expect(t1.squeeze().eql(t2.squeeze()));
}

test "squeezeN matches sequential squeeze" {
    const T = Transcript(M31);
    var t1 = T.init("batch-test");
    var t2 = T.init("batch-test");

    const batch = t1.squeezeN(5);
    for (batch) |b| {
        try std.testing.expect(b.eql(t2.squeeze()));
    }
}

test "many squeezes use buffer efficiently" {
    const T = Transcript(M31);
    var t = T.init("efficiency-test");

    // Squeeze more than one buffer's worth
    var challenges: [20]M31 = undefined;
    for (&challenges) |*c| {
        c.* = t.squeeze();
    }

    // Verify all are different (statistical test)
    for (challenges[0 .. challenges.len - 1], 0..) |c, i| {
        for (challenges[i + 1 ..]) |other| {
            try std.testing.expect(!c.eql(other));
        }
    }
}
