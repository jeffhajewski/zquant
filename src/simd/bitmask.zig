//! Expanding packed bits into SIMD lanes.
//!
//! Shared by the QJL sign-dot (one bit per coordinate) and the bit-plane code
//! unpacker (one plane per code bit). Both need the same primitive: 16 bits of a
//! bitmap into 16 lane predicates.
//!
//! Broadcast the two source bytes across eight lanes each, AND with a per-lane bit
//! selector, compare for equality. Three operations, branch-free, and `@shuffle`
//! with a comptime mask is portable — unlike a runtime-indexed lookup, which needs
//! the barrier trick in `simd/scan.zig`.

const std = @import("std");

pub const lanes: usize = 16;

const selector: @Vector(lanes, u8) = .{
    1, 2, 4, 8, 16, 32, 64, 128,
    1, 2, 4, 8, 16, 32, 64, 128,
};

/// Bit `i & 7` of byte `i >> 3` becomes lane `i`. This ordering is the one
/// `quant/prod.zig`'s `packSigns` and `quant/packing.zig`'s bit-plane writer both
/// use; all three must agree or every sign and every code is wrong.
pub inline fn expand16(pair: [2]u8) @Vector(lanes, bool) {
    const two: @Vector(2, u8) = pair;
    const spread = @shuffle(u8, two, undefined, @Vector(lanes, i32){
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1,
    });
    return (spread & selector) == selector;
}

/// Expand to `0` or `weight` per lane, for assembling multi-bit values.
pub inline fn expandWeighted(pair: [2]u8, comptime weight: u8) @Vector(lanes, u8) {
    const on: @Vector(lanes, u8) = @splat(weight);
    const off: @Vector(lanes, u8) = @splat(0);
    return @select(u8, expand16(pair), on, off);
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;

test "expand16 matches scalar bit extraction" {
    var prng = std.Random.DefaultPrng.init(1);
    for (0..500) |_| {
        const pair = [2]u8{ prng.random().int(u8), prng.random().int(u8) };
        const got: [lanes]bool = expand16(pair);
        for (got, 0..) |g, i| {
            const want = ((pair[i >> 3] >> @intCast(i & 7)) & 1) == 1;
            try testing.expectEqual(want, g);
        }
    }
}

test "expandWeighted yields zero or the weight" {
    var prng = std.Random.DefaultPrng.init(2);
    for (0..200) |_| {
        const pair = [2]u8{ prng.random().int(u8), prng.random().int(u8) };
        inline for ([_]u8{ 1, 2, 4 }) |weight| {
            const got: [lanes]u8 = expandWeighted(pair, weight);
            for (got, 0..) |g, i| {
                const set = ((pair[i >> 3] >> @intCast(i & 7)) & 1) == 1;
                try testing.expectEqual(@as(u8, if (set) weight else 0), g);
            }
        }
    }
}

test "edge patterns" {
    const zeros: [lanes]u8 = expandWeighted(.{ 0x00, 0x00 }, 4);
    try testing.expectEqual(@as(u8, 0), zeros[0]);

    const all_set: [lanes]bool = expand16(.{ 0xFF, 0xFF });
    const none_set: [lanes]bool = expand16(.{ 0x00, 0x00 });
    for (all_set, none_set) |a, b| {
        try testing.expect(a);
        try testing.expect(!b);
    }

    // Lane 0 is the low bit of byte 0; lane 15 the high bit of byte 1.
    const low: [lanes]bool = expand16(.{ 0x01, 0x00 });
    const high: [lanes]bool = expand16(.{ 0x00, 0x80 });
    try testing.expect(low[0]);
    try testing.expect(!low[1]);
    try testing.expect(high[15]);
}
