//! int8 dot-product accumulation.
//!
//! `acc[j] += Σ a[4j+m]·b[4j+m]` for four groups of four — one instruction where the
//! target has `SDOT`, a widening chain where it does not.
//!
//! ## Why this exists
//!
//! Profiling put the scan kernel at **IPC 4.0**, essentially peak. It was not
//! executing slowly; it was executing too many instructions. Accumulating through
//! i16 costs `smlal`, `smlal2`, and two `saddw` per 16 products — four instructions
//! where `SDOT` does the same work in one, and without i16's overflow ceiling, since
//! it accumulates directly into i32.
//!
//! LLVM will not produce `SDOT` from the natural Zig formulation (tested: the 4-way
//! grouping lowers to scalar `umov` extraction), so this is inline assembly for the
//! same reason `simd/shuffle.zig` is.

const std = @import("std");
const builtin = @import("builtin");

pub const Acc = @Vector(4, i32);
pub const Bytes = @Vector(16, i8);

const has_sdot = switch (builtin.cpu.arch) {
    .aarch64, .aarch64_be => std.Target.aarch64.featureSetHas(builtin.cpu.features, .dotprod),
    else => false,
};

/// Whether the single-instruction path is available. False means correct but slower.
pub const accelerated = has_sdot;

/// Four independent dot products of four int8 pairs each, accumulated into `acc`.
///
/// Lane assignment is not meaningful to callers: the kernel sums all four lanes at
/// the end, so only the total matters.
pub inline fn accumulate(acc: Acc, a: Bytes, b: Bytes) Acc {
    if (comptime has_sdot) {
        var out = acc;
        asm ("sdot %[acc].4s, %[x].16b, %[y].16b"
            : [acc] "+w" (out),
            : [x] "w" (a),
              [y] "w" (b),
        );
        return out;
    }
    return scalar(acc, a, b);
}

/// Portable fallback. Widening multiply-accumulate, four products per lane.
fn scalar(acc: Acc, a: Bytes, b: Bytes) Acc {
    const wide = @as(@Vector(16, i32), a) * @as(@Vector(16, i32), b);
    const values: [16]i32 = wide;
    var out: [4]i32 = acc;
    inline for (0..4) |lane| {
        inline for (0..4) |m| out[lane] += values[lane * 4 + m];
    }
    return out;
}

/// Sum of all four lanes.
pub inline fn total(acc: Acc) i32 {
    return @reduce(.Add, acc);
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;

test "accumulate matches a scalar dot product" {
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const random = prng.random();

    for (0..500) |_| {
        var av: [16]i8 = undefined;
        var bv: [16]i8 = undefined;
        var seed: [4]i32 = undefined;
        for (&av) |*v| v.* = random.intRangeAtMost(i8, -128, 127);
        for (&bv) |*v| v.* = random.intRangeAtMost(i8, -128, 127);
        for (&seed) |*v| v.* = random.intRangeAtMost(i32, -1000, 1000);

        const got: [4]i32 = accumulate(seed, av, bv);
        const want: [4]i32 = scalar(seed, av, bv);
        try testing.expectEqualSlices(i32, &want, &got);

        // And against a plain loop, so both paths are checked rather than each other.
        var reference: [4]i32 = seed;
        for (0..16) |i| reference[i / 4] += @as(i32, av[i]) * @as(i32, bv[i]);
        try testing.expectEqualSlices(i32, &reference, &got);
    }
}

test "saturating inputs do not overflow" {
    // The reason this replaced i16 accumulation: products reach 127·127 and i32 has
    // room for millions of them.
    const a: [16]i8 = @splat(127);
    const b: [16]i8 = @splat(127);
    var acc: Acc = @splat(0);
    for (0..1000) |_| acc = accumulate(acc, a, b);
    // 1000 iterations x 4 products x 16129 per lane.
    try testing.expectEqual(@as(i32, 1000 * 4 * 16129), @as([4]i32, acc)[0]);
    try testing.expectEqual(@as(i32, 4 * 1000 * 4 * 16129), total(acc));
}

test "negative values and zero" {
    const a: [16]i8 = @splat(-128);
    const b: [16]i8 = @splat(1);
    const zero: Acc = @splat(0);
    try testing.expectEqual(@as(i32, 4 * -128), @as([4]i32, accumulate(zero, a, b))[0]);

    const zeros: [16]i8 = @splat(0);
    try testing.expectEqual(@as(i32, 0), total(accumulate(zero, a, zeros)));
}
