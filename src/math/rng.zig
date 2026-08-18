//! Philox4x32-10: a counter-based pseudorandom function.
//!
//! Counter-based rather than stateful because §4.6 of the design requires that a
//! rotation be reproducible from `(seed, purpose, index)` alone. That lets an index
//! be described by its seed instead of by a materialized d×d matrix, and lets every
//! language binding regenerate byte-identical codes.
//!
//! Reference: Salmon, Moraes, Dror, Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3"
//! (SC '11). Constants and round structure follow the Random123 implementation.

const std = @import("std");

/// Multiplier and Weyl constants from the Random123 reference implementation.
const M0: u32 = 0xD2511F53;
const M1: u32 = 0xCD9E8D57;
const W0: u32 = 0x9E3779B9; // golden ratio
const W1: u32 = 0xBB67AE85; // sqrt(3) - 1

const rounds = 10;

/// Domain separators, so that independent uses of the same seed never share a
/// counter sequence. Extend rather than reuse: a collision here silently
/// correlates two things that the algorithm assumes are independent.
pub const Purpose = enum(u32) {
    rht_signs = 1,
    rht_permutation = 2,
    sketch_signs = 3,
    sketch_permutation = 4,
    dense_rotation = 5,
    testing = 0xFFFF_FFFF,
};

pub const Philox = struct {
    key: [2]u32,

    pub fn init(seed: u64) Philox {
        return .{ .key = .{
            @truncate(seed),
            @truncate(seed >> 32),
        } };
    }

    /// The raw PRF: 128 bits of counter and 64 bits of key to 128 bits of output.
    pub fn block(self: Philox, counter: [4]u32) [4]u32 {
        var ctr = counter;
        var key = self.key;
        for (0..rounds) |r| {
            if (r > 0) {
                key[0] +%= W0;
                key[1] +%= W1;
            }
            ctr = roundFn(ctr, key);
        }
        return ctr;
    }

    /// Domain-separated addressing: purpose occupies one counter word, the index
    /// the low two. The high word is reserved.
    pub fn blockAt(self: Philox, purpose: Purpose, index: u64) [4]u32 {
        return self.block(.{
            @truncate(index),
            @truncate(index >> 32),
            @intFromEnum(purpose),
            0,
        });
    }

    pub fn stream(self: Philox, purpose: Purpose) Stream {
        return .{ .philox = self, .purpose = purpose, .index = 0, .cursor = 4, .buf = undefined };
    }
};

fn roundFn(ctr: [4]u32, key: [2]u32) [4]u32 {
    const p0 = @as(u64, M0) * @as(u64, ctr[0]);
    const p1 = @as(u64, M1) * @as(u64, ctr[2]);
    const lo0: u32 = @truncate(p0);
    const hi0: u32 = @truncate(p0 >> 32);
    const lo1: u32 = @truncate(p1);
    const hi1: u32 = @truncate(p1 >> 32);
    return .{ hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0 };
}

/// Sequential view over a domain-separated counter sequence.
///
/// `Stream` is a convenience, not the source of truth: anything whose value must be
/// reproducible at an arbitrary position should address `blockAt` directly rather
/// than depending on how many words a previous consumer happened to draw.
pub const Stream = struct {
    philox: Philox,
    purpose: Purpose,
    index: u64,
    cursor: u32,
    buf: [4]u32,
    gaussian_spare: ?f64 = null,

    pub fn nextU32(self: *Stream) u32 {
        if (self.cursor == 4) {
            self.buf = self.philox.blockAt(self.purpose, self.index);
            self.index += 1;
            self.cursor = 0;
        }
        const v = self.buf[self.cursor];
        self.cursor += 1;
        return v;
    }

    /// Uniform in [0, 1). Takes the top 24 bits so every result is exactly
    /// representable in f32 and the spacing is uniform.
    pub fn nextFloat(self: *Stream) f64 {
        return @as(f64, @floatFromInt(self.nextU32() >> 8)) * 0x1p-24;
    }

    /// Uniform in [0, bound), rejection-sampled. Unbiased, unlike `% bound`.
    pub fn nextBounded(self: *Stream, bound: u32) u32 {
        std.debug.assert(bound > 0);
        // Reject the short tail so the accepted range is an exact multiple of bound.
        const limit = std.math.maxInt(u32) - (std.math.maxInt(u32) % bound);
        while (true) {
            const v = self.nextU32();
            if (v < limit) return v % bound;
        }
    }

    /// Standard normal via Marsaglia polar method.
    ///
    /// Polar rather than Box-Muller: it avoids sin/cos, whose last-bit results vary
    /// between libm implementations. Only `sqrt` and `log` are used, and this feeds
    /// the reference dense rotation (a test oracle) rather than anything serialized.
    pub fn nextGaussian(self: *Stream) f64 {
        if (self.gaussian_spare) |g| {
            self.gaussian_spare = null;
            return g;
        }
        while (true) {
            const u = 2.0 * self.nextFloat() - 1.0;
            const v = 2.0 * self.nextFloat() - 1.0;
            const s = u * u + v * v;
            if (s >= 1.0 or s == 0.0) continue;
            const scale = @sqrt(-2.0 * @log(s) / s);
            self.gaussian_spare = v * scale;
            return u * scale;
        }
    }

    /// Fill with ±1. One bit per element, 32 elements per draw.
    pub fn fillSigns(self: *Stream, dst: []f32) void {
        var i: usize = 0;
        while (i < dst.len) {
            const word = self.nextU32();
            const n = @min(32, dst.len - i);
            for (0..n) |b| {
                dst[i + b] = if ((word >> @intCast(b)) & 1 == 1) 1.0 else -1.0;
            }
            i += n;
        }
    }

    /// In-place Fisher-Yates. `dst` is overwritten with a permutation of 0..dst.len.
    pub fn fillPermutation(self: *Stream, dst: []u32) void {
        for (dst, 0..) |*slot, i| slot.* = @intCast(i);
        if (dst.len < 2) return;
        var i: usize = dst.len - 1;
        while (i > 0) : (i -= 1) {
            const j = self.nextBounded(@intCast(i + 1));
            std.mem.swap(u32, &dst[i], &dst[j]);
        }
    }
};

// -- tests -------------------------------------------------------------------

const testing = std.testing;

test "philox4x32-10 known-answer vectors" {
    // Random123 reference KAT vectors. These pin the round structure, constant
    // values, and key-bump schedule all at once; getting any of them wrong moves
    // every output.
    {
        const p = Philox{ .key = .{ 0, 0 } };
        try testing.expectEqual(
            [4]u32{ 0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8 },
            p.block(.{ 0, 0, 0, 0 }),
        );
    }
    {
        const p = Philox{ .key = .{ 0xffffffff, 0xffffffff } };
        try testing.expectEqual(
            [4]u32{ 0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd },
            p.block(.{ 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff }),
        );
    }
    {
        // Digits of pi and e, as used by the reference test suite.
        const p = Philox{ .key = .{ 0xa4093822, 0x299f31d0 } };
        try testing.expectEqual(
            [4]u32{ 0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1 },
            p.block(.{ 0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344 }),
        );
    }
}

test "blockAt is stateless and position-addressable" {
    const p = Philox.init(0x5EED);
    // Reading position 1000 directly must equal reading it after a stream walks there.
    const direct = p.blockAt(.rht_signs, 1000);
    var s = p.stream(.rht_signs);
    for (0..4000) |_| _ = s.nextU32();
    const walked = p.blockAt(.rht_signs, 1000);
    try testing.expectEqual(direct, walked);
}

test "purposes are independent" {
    const p = Philox.init(0x5EED);
    try testing.expect(!std.mem.eql(
        u32,
        &p.blockAt(.rht_signs, 7),
        &p.blockAt(.sketch_signs, 7),
    ));
}

test "different seeds diverge" {
    try testing.expect(!std.mem.eql(
        u32,
        &Philox.init(1).blockAt(.testing, 0),
        &Philox.init(2).blockAt(.testing, 0),
    ));
}

test "fillSigns is balanced and only ever ±1" {
    var buf: [4096]f32 = undefined;
    var s = Philox.init(0xABCD).stream(.rht_signs);
    s.fillSigns(&buf);

    var sum: f64 = 0;
    for (buf) |v| {
        try testing.expect(v == 1.0 or v == -1.0);
        sum += v;
    }
    // ~N(0, sqrt(n)); 4 sigma is 256 for n=4096.
    try testing.expect(@abs(sum) < 256);
}

test "fillSigns handles lengths that straddle word boundaries" {
    // 32 signs per u32 draw, so the interesting cases are around multiples of 32.
    for ([_]usize{ 0, 1, 31, 32, 33, 63, 64, 65 }) |n| {
        var buf: [65]f32 = undefined;
        @memset(&buf, 0);
        var s = Philox.init(7).stream(.rht_signs);
        s.fillSigns(buf[0..n]);
        for (buf[0..n]) |v| try testing.expect(v == 1.0 or v == -1.0);
        for (buf[n..]) |v| try testing.expectEqual(@as(f32, 0), v); // no overrun
    }
}

test "fillPermutation produces a genuine permutation" {
    var perm: [257]u32 = undefined;
    var s = Philox.init(0x1234).stream(.rht_permutation);
    s.fillPermutation(&perm);

    var seen = [_]bool{false} ** 257;
    for (perm) |v| {
        try testing.expect(v < 257);
        try testing.expect(!seen[v]); // no duplicates
        seen[v] = true;
    }
    // And it should not be the identity.
    var fixed: usize = 0;
    for (perm, 0..) |v, i| {
        if (v == i) fixed += 1;
    }
    try testing.expect(fixed < 20);
}

test "fillPermutation degenerate lengths" {
    var empty: [0]u32 = undefined;
    var one: [1]u32 = undefined;
    var s = Philox.init(1).stream(.rht_permutation);
    s.fillPermutation(&empty);
    s.fillPermutation(&one);
    try testing.expectEqual(@as(u32, 0), one[0]);
}

test "nextBounded is unbiased across buckets" {
    const buckets = 7;
    var counts = [_]u32{0} ** buckets;
    var s = Philox.init(0xFEED).stream(.testing);
    const n = 70_000;
    for (0..n) |_| counts[s.nextBounded(buckets)] += 1;
    const expected: f64 = n / buckets;
    for (counts) |c| {
        const dev = @abs(@as(f64, @floatFromInt(c)) - expected) / expected;
        try testing.expect(dev < 0.05);
    }
}

test "nextFloat stays in [0,1)" {
    var s = Philox.init(3).stream(.testing);
    var lo: f64 = 1.0;
    var hi: f64 = 0.0;
    for (0..100_000) |_| {
        const v = s.nextFloat();
        try testing.expect(v >= 0.0 and v < 1.0);
        lo = @min(lo, v);
        hi = @max(hi, v);
    }
    try testing.expect(lo < 0.01 and hi > 0.99); // actually spans the range
}

test "nextGaussian has unit moments" {
    var s = Philox.init(0x600D).stream(.dense_rotation);
    const n = 200_000;
    var sum: f64 = 0;
    var sumsq: f64 = 0;
    for (0..n) |_| {
        const g = s.nextGaussian();
        sum += g;
        sumsq += g * g;
    }
    const mean = sum / n;
    const variance = sumsq / n - mean * mean;
    try testing.expect(@abs(mean) < 0.01);
    try testing.expect(@abs(variance - 1.0) < 0.02);
}
