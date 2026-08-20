//! QJL sign-dot kernel: the second half of the `prod` estimator.
//!
//!     ⟨q, x̃⟩ = ‖x‖ · [ ⟨p, ỹ⟩ + γ·scale·⟨S'p, qjl⟩ ]
//!              └─ simd/scan.zig ─┘   └──── here ────┘
//!
//! `qjl` is one bit per coordinate and `S'p` is the sketched query, so the term is a
//! sum of `S'p` with per-coordinate signs taken from a bitmap. No multiply.
//!
//! **This term is not optional.** Dropping it leaves exactly the MSE-only estimate
//! that `prod` exists to correct — biased by 2/π at one MSE bit, as measured in
//! `quant/prod.zig`. A scan that computes only the first term returns plausible,
//! consistently wrong scores.
//!
//! ## Bit expansion
//!
//! Turning 16 bitmap bits into 16 byte lanes is the whole trick: broadcast the two
//! source bytes across eight lanes each, AND with a per-lane bit selector, and
//! compare for equality. Three operations, no branches, and `@shuffle` with a
//! comptime mask is portable (unlike a runtime-indexed lookup — see `simd/scan.zig`).
//!
//! Accumulation is i16. Terms are bounded by 127, so a lane can absorb 258 of them
//! before overflowing; widening every 128 iterations leaves a wide margin at any
//! dimension this library will see.

const std = @import("std");
const bitmask = @import("bitmask.zig");

/// Coordinates handled per iteration.
pub const lanes: usize = bitmask.lanes;

/// Iterations between i16 → i32 widenings. 128 × 127 = 16256, half of i16's range.
const widen_interval: usize = 128;

/// The sketched query `S'p`, quantized to int8.
pub const Query = struct {
    values: []i8,
    /// (S'p)_j ≈ values[j] * scale
    scale: f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, dim: u32) std.mem.Allocator.Error!Query {
        return .{
            .values = try allocator.alloc(i8, dim),
            .scale = 1.0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Query) void {
        self.allocator.free(self.values);
        self.* = undefined;
    }

    pub fn load(self: *Query, sketched: []const f32) void {
        std.debug.assert(sketched.len == self.values.len);

        var magnitude: f32 = 0;
        for (sketched) |v| magnitude = @max(magnitude, @abs(v));
        self.scale = if (magnitude > 0) magnitude / 127.0 else 1.0;
        const inverse = 1.0 / self.scale;

        for (self.values, sketched) |*dst, v| {
            // Clamped to ±127, never −128, so negation cannot overflow i8.
            dst.* = @intFromFloat(@round(std.math.clamp(v * inverse, -127.0, 127.0)));
        }
    }
};

/// `⟨S'p, qjl⟩`, with `qjl` read as ±1 from `bitmap`.
pub fn signDot(query: Query, bitmap: []const u8, dim: u32) f32 {
    std.debug.assert(dim % lanes == 0);
    std.debug.assert(bitmap.len * 8 >= dim);
    std.debug.assert(query.values.len >= dim);

    var total: i32 = 0;
    var acc: @Vector(lanes, i16) = @splat(0);

    const iterations = dim / lanes;
    for (0..iterations) |i| {
        const set = bitmask.expand16(bitmap[i * 2 ..][0..2].*);
        const w: @Vector(lanes, i8) = query.values[i * lanes ..][0..lanes].*;
        const signed = @select(i8, set, w, -w);
        acc +%= @as(@Vector(lanes, i16), signed);

        if ((i + 1) % widen_interval == 0) {
            total += @reduce(.Add, @as(@Vector(lanes, i32), acc));
            acc = @splat(0);
        }
    }
    total += @reduce(.Add, @as(@Vector(lanes, i32), acc));

    return @as(f32, @floatFromInt(total)) * query.scale;
}

/// Exact f32 reference.
pub fn signDotExact(sketched: []const f32, bitmap: []const u8) f32 {
    var acc: f64 = 0;
    for (sketched, 0..) |w, i| {
        const bit = (bitmap[i >> 3] >> @intCast(i & 7)) & 1;
        acc += if (bit == 1) @as(f64, w) else -@as(f64, w);
    }
    return @floatCast(acc);
}

pub fn canVectorize(dim: u32) bool {
    return dim % lanes == 0;
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;

/// Three-sigma bound on the int8 quantization error of a `dim`-term signed sum.
/// Each term carries a uniform error of at most half a step, so the sum's standard
/// deviation is √(dim/12)·step and 3σ is a little under √dim·step.
fn quantizationTolerance(dim: u32, scale: f32) f32 {
    return 3.0 * @sqrt(@as(f32, @floatFromInt(dim)) / 12.0) * scale;
}

fn fill(values: []f32, bitmap: []u8, random: std.Random) void {
    for (values) |*v| v.* = random.floatNorm(f32);
    for (bitmap) |*b| b.* = random.int(u8);
}

test "expansion ordering matches packSigns in prod" {
    // The bitmask module and `packSigns` must agree exactly or every sign is wrong.
    var prng = std.Random.DefaultPrng.init(1);
    for (0..200) |_| {
        const pair = [2]u8{ prng.random().int(u8), prng.random().int(u8) };
        const got = bitmask.expand16(pair);
        for (0..lanes) |i| {
            const want = ((pair[i >> 3] >> @intCast(i & 7)) & 1) == 1;
            try testing.expectEqual(want, got[i]);
        }
    }
}

test "vectorized sign-dot tracks the exact reference" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const random = prng.random();

    for ([_]u32{ 16, 64, 256, 1024, 4096 }) |dim| {
        var query = try Query.init(allocator, dim);
        defer query.deinit();

        const sketched = try allocator.alloc(f32, dim);
        defer allocator.free(sketched);
        const bitmap = try allocator.alloc(u8, dim / 8);
        defer allocator.free(bitmap);

        var squared_error: f64 = 0;
        var squared_signal: f64 = 0;
        for (0..40) |_| {
            fill(sketched, bitmap, random);
            query.load(sketched);

            const fast = signDot(query, bitmap, dim);
            const exact = signDotExact(sketched, bitmap);
            const err = @as(f64, fast) - exact;
            squared_error += err * err;
            squared_signal += @as(f64, exact) * exact;
        }
        // int8 quantization of the sketched query is the only error source.
        try testing.expect(@sqrt(squared_error / squared_signal) < 0.01);
    }
}

test "all-ones and all-zeros bitmaps give the signed sums" {
    const allocator = testing.allocator;
    const dim: u32 = 256;

    var query = try Query.init(allocator, dim);
    defer query.deinit();
    const sketched = try allocator.alloc(f32, dim);
    defer allocator.free(sketched);
    const bitmap = try allocator.alloc(u8, dim / 8);
    defer allocator.free(bitmap);

    var prng = std.Random.DefaultPrng.init(3);
    for (sketched) |*v| v.* = prng.random().floatNorm(f32);
    query.load(sketched);

    var sum: f64 = 0;
    for (sketched) |v| sum += v;

    // Quantization error accumulates over a sum of `dim` terms as √dim · step/√3.
    // Derived rather than guessed: a fixed absolute tolerance is wrong here because
    // the error grows with dimension.
    const tolerance = quantizationTolerance(dim, query.scale);

    @memset(bitmap, 0xFF);
    try testing.expectApproxEqAbs(@as(f32, @floatCast(sum)), signDot(query, bitmap, dim), tolerance);

    @memset(bitmap, 0x00);
    try testing.expectApproxEqAbs(@as(f32, @floatCast(-sum)), signDot(query, bitmap, dim), tolerance);
}

test "no overflow at large dimensions with saturated values" {
    // Worst case for the i16 accumulator: every value at ±127 with matching signs,
    // so each lane accumulates the maximum possible magnitude.
    const allocator = testing.allocator;
    const dim: u32 = 4096;

    var query = try Query.init(allocator, dim);
    defer query.deinit();
    const sketched = try allocator.alloc(f32, dim);
    defer allocator.free(sketched);
    const bitmap = try allocator.alloc(u8, dim / 8);
    defer allocator.free(bitmap);

    for (sketched) |*v| v.* = 1.0; // all map to +127
    @memset(bitmap, 0xFF);
    query.load(sketched);

    const got = signDot(query, bitmap, dim);
    try testing.expectApproxEqRel(@as(f32, @floatFromInt(dim)), got, 1e-3);

    // And the alternating case, where partial sums cancel.
    for (sketched, 0..) |*v, i| v.* = if (i % 2 == 0) 1.0 else -1.0;
    query.load(sketched);
    @memset(bitmap, 0xFF);
    try testing.expectApproxEqAbs(@as(f32, 0), signDot(query, bitmap, dim), 1e-3);
}

test "zero sketch scores zero" {
    const allocator = testing.allocator;
    const dim: u32 = 64;
    var query = try Query.init(allocator, dim);
    defer query.deinit();
    const sketched = [_]f32{0} ** dim;
    query.load(&sketched);
    var bitmap = [_]u8{0b10110101} ** (dim / 8);
    try testing.expectEqual(@as(f32, 0), signDot(query, &bitmap, dim));
}

test "flipping every bit negates the result" {
    const allocator = testing.allocator;
    const dim: u32 = 512;

    var query = try Query.init(allocator, dim);
    defer query.deinit();
    const sketched = try allocator.alloc(f32, dim);
    defer allocator.free(sketched);
    const bitmap = try allocator.alloc(u8, dim / 8);
    defer allocator.free(bitmap);
    const flipped = try allocator.alloc(u8, dim / 8);
    defer allocator.free(flipped);

    var prng = std.Random.DefaultPrng.init(9);
    fill(sketched, bitmap, prng.random());
    query.load(sketched);
    for (flipped, bitmap) |*f, b| f.* = ~b;

    try testing.expectApproxEqAbs(
        signDot(query, bitmap, dim),
        -signDot(query, flipped, dim),
        1e-3,
    );
}

test "single set bit isolates one coordinate" {
    const allocator = testing.allocator;
    const dim: u32 = 128;

    var query = try Query.init(allocator, dim);
    defer query.deinit();
    const sketched = try allocator.alloc(f32, dim);
    defer allocator.free(sketched);
    const bitmap = try allocator.alloc(u8, dim / 8);
    defer allocator.free(bitmap);

    var prng = std.Random.DefaultPrng.init(21);
    for (sketched) |*v| v.* = prng.random().floatNorm(f32);
    query.load(sketched);

    var total: f64 = 0;
    for (sketched) |v| total += v;

    // With all bits clear the answer is −Σw; setting bit j adds 2·w_j.
    for ([_]usize{ 0, 1, 7, 8, 63, 127 }) |j| {
        @memset(bitmap, 0);
        bitmap[j >> 3] |= @as(u8, 1) << @intCast(j & 7);
        const expected = -total + 2.0 * sketched[j];
        try testing.expectApproxEqAbs(
            @as(f32, @floatCast(expected)),
            signDot(query, bitmap, dim),
            quantizationTolerance(dim, query.scale),
        );
    }
}

test "canVectorize gates on dimension" {
    try testing.expect(canVectorize(16));
    try testing.expect(canVectorize(1024));
    try testing.expect(!canVectorize(8));
    try testing.expect(!canVectorize(100));
}
