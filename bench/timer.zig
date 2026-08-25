//! Monotonic timer for benchmarks.
//!
//! Zig 0.16 removed `std.time.Timer` and routes timing through the new `Io`
//! interface. Benchmarks do not otherwise need an `Io`, so this calls the C
//! monotonic clock directly rather than plumbing one through every bench.

const std = @import("std");

pub const Timer = struct {
    start_ns: u64,

    pub fn start() Timer {
        return .{ .start_ns = nanos() };
    }

    pub fn reset(self: *Timer) void {
        self.start_ns = nanos();
    }

    /// Nanoseconds since `start` or the last `reset`.
    pub fn read(self: Timer) u64 {
        return nanos() - self.start_ns;
    }

    fn nanos() u64 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }
};
