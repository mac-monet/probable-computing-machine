const protocol = @import("protocol.zig");

/// Standard configuration for production use.
/// 80 queries provides ~80 bits of security.
pub const StandardConfig = protocol.ProtocolConfig{
    .num_queries = 80,
};

/// Fast configuration for testing.
/// Fewer queries means faster proving but less security.
pub const FastConfig = protocol.ProtocolConfig{
    .num_queries = 16,
};

/// Minimal configuration for unit tests.
/// Very fast but minimal security.
pub const TestConfig = protocol.ProtocolConfig{
    .num_queries = 4,
};
