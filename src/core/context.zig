const std = @import("std");
const Allocator = std.mem.Allocator;
const Transcript = @import("transcript.zig").Transcript;

/// ProverContext owns memory for proving operations.
/// Uses arena allocation for proof data and fixed scratch buffers for folding.
pub fn ProverContext(comptime F: type, comptime max_vars: usize) type {
    const max_size = 1 << max_vars;

    return struct {
        const Self = @This();

        /// Backing allocator for arena
        backing: Allocator,

        /// Arena for proof data - freed all at once
        arena: std.heap.ArenaAllocator,

        /// Primary scratch buffer for in-place folding
        /// Reused every round - size: max_size
        scratch: []F,

        /// Secondary scratch for ops needing two buffers
        scratch_aux: []F,

        /// Transcript for Fiat-Shamir challenges
        transcript: Transcript(F),

        pub fn init(backing: Allocator) !Self {
            var arena = std.heap.ArenaAllocator.init(backing);
            errdefer arena.deinit();

            const scratch = try backing.alloc(F, max_size);
            errdefer backing.free(scratch);

            const scratch_aux = try backing.alloc(F, max_size);
            errdefer backing.free(scratch_aux);

            return .{
                .backing = backing,
                .arena = arena,
                .scratch = scratch,
                .scratch_aux = scratch_aux,
                .transcript = Transcript(F).init(""),
            };
        }

        pub fn deinit(self: *Self) void {
            self.arena.deinit();
            self.backing.free(self.scratch);
            self.backing.free(self.scratch_aux);
        }

        /// Reset for next proof - keeps allocations, clears state
        pub fn reset(self: *Self) void {
            _ = self.arena.reset(.retain_capacity);
            self.transcript = Transcript(F).init("");
        }

        /// Reset with a new domain separator
        pub fn resetWithDomain(self: *Self, domain: []const u8) void {
            _ = self.arena.reset(.retain_capacity);
            self.transcript = Transcript(F).init(domain);
        }

        /// Allocate from arena (lives until reset/deinit)
        pub fn alloc(self: *Self, comptime T: type, n: usize) ![]T {
            return self.arena.allocator().alloc(T, n);
        }

        /// Copy data into scratch buffer, returns view
        pub fn copyToScratch(self: *Self, data: []const F) []F {
            std.debug.assert(data.len <= max_size);
            @memcpy(self.scratch[0..data.len], data);
            return self.scratch[0..data.len];
        }

        /// Copy data into auxiliary scratch buffer, returns view
        pub fn copyToScratchAux(self: *Self, data: []const F) []F {
            std.debug.assert(data.len <= max_size);
            @memcpy(self.scratch_aux[0..data.len], data);
            return self.scratch_aux[0..data.len];
        }

        /// Get scratch slice of given size
        pub fn getScratch(self: *Self, size: usize) []F {
            std.debug.assert(size <= max_size);
            return self.scratch[0..size];
        }

        /// Get auxiliary scratch slice of given size
        pub fn getScratchAux(self: *Self, size: usize) []F {
            std.debug.assert(size <= max_size);
            return self.scratch_aux[0..size];
        }
    };
}

/// VerifierContext for verification operations.
/// Lightweight - mostly just transcript state.
pub fn VerifierContext(comptime F: type) type {
    return struct {
        const Self = @This();

        transcript: Transcript(F),

        pub fn init(domain: []const u8) Self {
            return .{ .transcript = Transcript(F).init(domain) };
        }

        /// Reset with new domain separator
        pub fn reset(self: *Self, domain: []const u8) void {
            self.transcript = Transcript(F).init(domain);
        }
    };
}

// ============ Tests ============ //

const M31 = @import("../fields/mersenne31.zig").Mersenne31;

test "ProverContext init and deinit" {
    var ctx = try ProverContext(M31, 10).init(std.testing.allocator);
    defer ctx.deinit();

    // Verify scratch buffers are the right size
    try std.testing.expectEqual(@as(usize, 1024), ctx.scratch.len);
    try std.testing.expectEqual(@as(usize, 1024), ctx.scratch_aux.len);
}

test "ProverContext scratch buffer reuse" {
    var ctx = try ProverContext(M31, 10).init(std.testing.allocator);
    defer ctx.deinit();

    // First use
    const data1 = [_]M31{ M31.fromU64(1), M31.fromU64(2), M31.fromU64(3), M31.fromU64(4) };
    const view1 = ctx.copyToScratch(&data1);
    try std.testing.expect(view1[0].eql(M31.fromU64(1)));

    // Same buffer, new data
    const data2 = [_]M31{ M31.fromU64(10), M31.fromU64(20) };
    const view2 = ctx.copyToScratch(&data2);
    try std.testing.expect(view2[0].eql(M31.fromU64(10)));

    // Views point to same underlying memory
    try std.testing.expectEqual(@intFromPtr(view1.ptr), @intFromPtr(view2.ptr));
}

test "ProverContext arena allocation and reset" {
    var ctx = try ProverContext(M31, 10).init(std.testing.allocator);
    defer ctx.deinit();

    // Allocate some data in arena
    const alloc1 = try ctx.alloc(M31, 100);
    alloc1[0] = M31.fromU64(42);

    // Reset clears arena
    ctx.reset();

    // New allocation may reuse memory but starts fresh
    const alloc2 = try ctx.alloc(M31, 100);
    _ = alloc2; // Arena memory is reused, values uninitialized
}

test "VerifierContext basic usage" {
    var ctx = VerifierContext(M31).init("test-domain");

    // Can squeeze challenges
    const c1 = ctx.transcript.squeeze();
    const c2 = ctx.transcript.squeeze();

    // Different challenges
    try std.testing.expect(!c1.eql(c2));

    // Reset gives fresh state
    ctx.reset("test-domain");
    const c3 = ctx.transcript.squeeze();
    try std.testing.expect(c1.eql(c3)); // Same domain, same first challenge
}

test "ProverContext transcript consistency" {
    var prover_ctx = try ProverContext(M31, 10).init(std.testing.allocator);
    defer prover_ctx.deinit();
    prover_ctx.resetWithDomain("test-protocol");

    var verifier_ctx = VerifierContext(M31).init("test-protocol");

    // Simulate protocol: absorb same values
    prover_ctx.transcript.absorb(M31.fromU64(100));
    verifier_ctx.transcript.absorb(M31.fromU64(100));

    // Squeeze should match
    const prover_c = prover_ctx.transcript.squeeze();
    const verifier_c = verifier_ctx.transcript.squeeze();

    try std.testing.expect(prover_c.eql(verifier_c));
}
