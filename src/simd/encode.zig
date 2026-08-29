//! Vectorized threshold encoder.
//!
//! Encoding is `idx = |{k : v > t_k}|` over ascending thresholds. The scalar
//! reference in `Codebook.encode` short-circuits on the first threshold not exceeded;
//! counting all of them is equivalent, because the thresholds are sorted, and it is
//! the form that vectorizes.
//!
//! ## Why not the binary search the design proposed
//!
//! docs/DESIGN.md §4.1 offered a threshold binary search at ~3b ops/element as the
//! better choice for b ≥ 3, against 2·(2^b − 1) for the linear scan. It does not
//! vectorize: each round probes `thresholds[idx]` at a *per-lane* runtime index,
//! which is a gather, and gathers are both slow and unavailable as a portable
//! builtin. The linear form uses only compares and adds against broadcast scalars.
//!
//! At b=4 that is 15 compare+add pairs per lane — about 3.8 ops/element at 8 lanes,
//! which is close to what the binary search would have cost scalar anyway. Encoding
//! is also not the hot path: it runs once per vector at insert, while the scan runs
//! once per vector per query.
//!
//! NaN: `NaN > t` is false for every threshold, so both paths yield index 0. Matching
//! by construction rather than by accident.

const std = @import("std");

/// Lanes per step. The target's natural width, floored at 4 so the code shape is the
/// same everywhere and LLVM splits or widens as it sees fit.
pub const lanes: usize = @max(4, std.simd.suggestVectorLength(f32) orelse 8);

/// Encode with a comptime-known threshold count, so the loop fully unrolls.
///
/// Only instantiated for b <= 4. Beyond that the unrolled form is a compile-time
/// disaster — b=8 is 255 broadcast constants and 255 unrolled compares per
/// specialization — for a bit-width outside the shuffle-LUT range the scan kernel
/// can use anyway.
fn encodeUnrolled(
    comptime count: usize,
    thresholds: []const f32,
    src: []const f32,
    dst: []u8,
) void {
    std.debug.assert(thresholds.len == count);

    const V = @Vector(lanes, f32);
    const B = @Vector(lanes, u8);
    const one: B = @splat(1);
    const zero: B = @splat(0);

    var broadcast: [count]V = undefined;
    inline for (0..count) |k| broadcast[k] = @splat(thresholds[k]);

    var i: usize = 0;
    while (i + lanes <= src.len) : (i += lanes) {
        const v: V = src[i..][0..lanes].*;
        var idx: B = zero;
        inline for (0..count) |k| {
            idx += @select(u8, v > broadcast[k], one, zero);
        }
        dst[i..][0..lanes].* = idx;
    }
    encodeTail(thresholds, src, dst, i);
}

/// Vector body with a runtime threshold loop, for bit-widths past the unrolled range.
fn encodeRolled(thresholds: []const f32, src: []const f32, dst: []u8) void {
    const V = @Vector(lanes, f32);
    const B = @Vector(lanes, u8);
    const one: B = @splat(1);
    const zero: B = @splat(0);

    var i: usize = 0;
    while (i + lanes <= src.len) : (i += lanes) {
        const v: V = src[i..][0..lanes].*;
        var idx: B = zero;
        for (thresholds) |t| {
            const tv: V = @splat(t);
            idx += @select(u8, v > tv, one, zero);
        }
        dst[i..][0..lanes].* = idx;
    }
    encodeTail(thresholds, src, dst, i);
}

/// Scalar remainder. Same counting rule, so it cannot disagree with the vector body.
fn encodeTail(thresholds: []const f32, src: []const f32, dst: []u8, start: usize) void {
    var i = start;
    while (i < src.len) : (i += 1) {
        var idx: u8 = 0;
        for (thresholds) |t| {
            if (src[i] > t) idx += 1;
        }
        dst[i] = idx;
    }
}

/// Runtime bit-width dispatch.
pub fn encodeSlice(bits: u6, thresholds: []const f32, src: []const f32, dst: []u8) void {
    std.debug.assert(src.len == dst.len);
    std.debug.assert(thresholds.len == (@as(usize, 1) << bits) - 1);
    switch (bits) {
        0 => @memset(dst, 0),
        1 => encodeUnrolled(1, thresholds, src, dst),
        2 => encodeUnrolled(3, thresholds, src, dst),
        3 => encodeUnrolled(7, thresholds, src, dst),
        4 => encodeUnrolled(15, thresholds, src, dst),
        else => encodeRolled(thresholds, src, dst),
    }
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;
const Codebook = @import("../quant/codebook.zig").Codebook;
const Density = @import("../math/density.zig").Density;

test "vector encoder matches the scalar reference bit for bit" {
    // The contract: this must be indistinguishable from `Codebook.encode`, which is
    // itself verified against a direct argmin. Any divergence is a silent corruption
    // of every code produced.
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const random = prng.random();

    // b=1..5 covers both kernels: unrolled up to 4, rolled at 5. Stopping there is
    // deliberate — the Lloyd solve costs ~27k iterations at b=8 (see docs/notes.md
    // 8ff805c), and paying that in a test that is really about the vector body would
    // add half a minute to every run for no extra path coverage.
    for (1..6) |bits| {
        var cb = try Codebook.init(testing.allocator, Density.gauss(1.0), @intCast(bits));
        defer cb.deinit();

        // Lengths that straddle the vector width in every way that matters.
        for ([_]usize{ 0, 1, lanes - 1, lanes, lanes + 1, 3 * lanes + 5, 1000 }) |n| {
            const src = try testing.allocator.alloc(f32, n);
            defer testing.allocator.free(src);
            const fast = try testing.allocator.alloc(u8, n);
            defer testing.allocator.free(fast);
            const reference = try testing.allocator.alloc(u8, n);
            defer testing.allocator.free(reference);

            for (src) |*v| v.* = (random.float(f32) - 0.5) * 12.0;

            encodeSlice(@intCast(bits), cb.thresholds, src, fast);
            for (src, reference) |v, *r| r.* = cb.encode(v);

            try testing.expectEqualSlices(u8, reference, fast);
        }
    }
}

test "values landing exactly on thresholds agree with the reference" {
    // The boundary case where a strict-vs-non-strict comparison mismatch would show
    // up, and nowhere else. Random inputs would essentially never hit it.
    var cb = try Codebook.init(testing.allocator, Density.gauss(1.0), 4);
    defer cb.deinit();

    var src = std.ArrayList(f32).empty;
    defer src.deinit(testing.allocator);
    for (cb.thresholds) |t| {
        try src.append(testing.allocator, t);
        try src.append(testing.allocator, std.math.nextAfter(f32, t, -1e30));
        try src.append(testing.allocator, std.math.nextAfter(f32, t, 1e30));
    }
    for (cb.centroids) |c| try src.append(testing.allocator, c);

    const fast = try testing.allocator.alloc(u8, src.items.len);
    defer testing.allocator.free(fast);
    encodeSlice(4, cb.thresholds, src.items, fast);

    for (src.items, fast) |v, got| try testing.expectEqual(cb.encode(v), got);
}

test "extreme and non-finite inputs agree with the reference" {
    var cb = try Codebook.init(testing.allocator, Density.gauss(1.0), 3);
    defer cb.deinit();

    const src = [_]f32{
        -std.math.inf(f32),     std.math.inf(f32),
        std.math.nan(f32),      -std.math.nan(f32),
        -1e30,                  1e30,
        0.0,                    -0.0,
        std.math.floatMin(f32), -std.math.floatMin(f32),
    };
    var fast: [src.len]u8 = undefined;
    encodeSlice(3, cb.thresholds, &src, &fast);

    for (src, fast) |v, got| try testing.expectEqual(cb.encode(v), got);
    // Spelled out, since NaN handling here is load-bearing rather than incidental.
    try testing.expectEqual(@as(u8, 0), fast[2]);
    try testing.expectEqual(@as(u8, 0), fast[3]);
}

test "zero bits encodes to zero" {
    var dst = [_]u8{ 9, 9, 9 };
    encodeSlice(0, &.{}, &.{ 1.0, -1.0, 0.0 }, &dst);
    for (dst) |v| try testing.expectEqual(@as(u8, 0), v);
}

test "every code is in range" {
    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();
    for (1..6) |bits| {
        var cb = try Codebook.init(testing.allocator, Density.sphereCoord(256), @intCast(bits));
        defer cb.deinit();

        var src: [512]f32 = undefined;
        var dst: [512]u8 = undefined;
        for (&src) |*v| v.* = (random.float(f32) - 0.5) * 20.0;
        encodeSlice(@intCast(bits), cb.thresholds, &src, &dst);

        const levels: u8 = @intCast(@as(u16, 1) << @as(u4, @intCast(bits)));
        for (dst) |c| try testing.expect(c < levels);
    }
}
