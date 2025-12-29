//! SIMD configuration for architecture-optimal vector widths.
//!
//! Uses std.simd.suggestVectorLength to query the target CPU at compile time:
//! - ARM NEON: 128-bit (4 x u32, 2 x u64)
//! - x86 SSE:  128-bit (4 x u32, 2 x u64)
//! - x86 AVX2: 256-bit (8 x u32, 4 x u64)
//! - AVX-512:  512-bit (16 x u32, 8 x u64)

const std = @import("std");

/// Optimal vector length for u32 on this target, or null if SIMD not beneficial.
pub const u32_len: ?comptime_int = std.simd.suggestVectorLength(u32);

/// Optimal vector length for u64 on this target, or null if SIMD not beneficial.
pub const u64_len: ?comptime_int = std.simd.suggestVectorLength(u64);

/// Optimal alignment in bytes for SIMD operations on u32 elements.
pub const alignment_bytes: usize = if (u32_len) |len| len * 4 else 16;

/// Optimal alignment for SIMD operations, for use with alignedAlloc.
pub const alignment: std.mem.Alignment = @enumFromInt(@ctz(alignment_bytes));
