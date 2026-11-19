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
//!
//! ## x86
//!
//! x86 has no signed×signed byte dot product short of VNNI, and even that
//! (`vpdpbusd`) takes one operand unsigned. What SSSE3 has is `pmaddubsw`: unsigned
//! × signed bytes, adjacent pairs summed into saturating i16. The standard route to
//! signed×signed moves `a`'s sign onto `b`,
//!
//!     a·b = |a| · (sign(a)·b)          pmaddubsw(pabsb(a), psignb(b, a))
//!
//! then `pmaddwd` against ones sums adjacent i16 pairs into i32. Four lanes of four
//! bytes each, the same lane assignment as `sdot` — so the paths agree lane for lane,
//! not just in total. Four instructions where `sdot` is one. LLVM forms neither
//! `pmaddubsw` nor `psignb` from the portable formulation (checked: it widens to i32
//! and emits ~30 instructions), hence assembly again.
//!
//! **Range.** This is exact only while `b` avoids −128. `psignb` negates in i8, so
//! −(−128) wraps back to −128 and the product takes the wrong sign; and the only pair
//! that could reach i16 saturation, 2·128·128 = 32768, needs exactly that lane. With
//! `b` in [−127, 127] a pair is bounded by 2·128·127 = 32512 < 32767 whatever `a` is
//! (`pabsb(−128)` is 0x80, which `pmaddubsw` reads as unsigned 128 — still exact).
//! Every caller passes the int8 query as `b`, and `scan.Query.load` clamps it to ±127;
//! `scan.Table.init` clamps the other side too. Tests in `simd/scan.zig` pin both, and
//! safety-checked builds assert it here.
//!
//! VNNI is not used: no machine this was developed on can execute it, and a kernel
//! that has never run is not one to ship.

const std = @import("std");
const builtin = @import("builtin");

pub const Acc = @Vector(4, i32);
pub const Bytes = @Vector(16, i8);

const has_sdot = switch (builtin.cpu.arch) {
    .aarch64, .aarch64_be => std.Target.aarch64.featureSetHas(builtin.cpu.features, .dotprod),
    else => false,
};

const has_ssse3 = switch (builtin.cpu.arch) {
    .x86, .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .ssse3),
    else => false,
};

/// VEX encodings. A legacy-SSE instruction in AVX code pays a state transition or a
/// false dependency (depending on the core) while the upper ymm halves are dirty, and
/// LLVM uses ymm freely in the code around these.
const has_avx = switch (builtin.cpu.arch) {
    .x86, .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .avx),
    else => false,
};

/// Whether a native dot-product path is available. False means correct but slower.
pub const accelerated = has_sdot or has_ssse3;

/// Four independent dot products of four int8 pairs each, accumulated into `acc`.
///
/// Lane assignment is not meaningful to callers: the kernel sums all four lanes at
/// the end, so only the total matters.
///
/// `b` must not contain −128 (see "Range" above); `a` may.
pub inline fn accumulate(acc: Acc, a: Bytes, b: Bytes) Acc {
    if (std.debug.runtime_safety) {
        const raw: [16]i8 = b;
        for (raw) |v| std.debug.assert(v != -128);
    }

    if (comptime has_sdot) {
        var out = acc;
        asm ("sdot %[acc].4s, %[x].16b, %[y].16b"
            : [acc] "+w" (out),
            : [x] "w" (a),
              [y] "w" (b),
        );
        return out;
    }

    if (comptime has_ssse3) {
        // `@abs` is left to LLVM, which emits `pabsb` for it and can then share one
        // across every query a table value is multiplied against.
        const magnitude: Bytes = @bitCast(@abs(a));
        return acc + maddPairs(maddubs(magnitude, signOf(a, b)));
    }

    return scalar(acc, a, b);
}

/// `psignb`: `x` negated where `s` is negative, zeroed where `s` is zero. The zeroing
/// is harmless here, since it happens exactly where `|a|` is zero.
inline fn signOf(s: Bytes, x: Bytes) Bytes {
    if (comptime has_avx) {
        return asm ("vpsignb %[s], %[x], %[out]"
            : [out] "=x" (-> Bytes),
            : [x] "x" (x),
              [s] "x" (s),
        );
    }
    // Two-operand form: `x` is read and written.
    var out = x;
    asm ("psignb %[s], %[out]"
        : [out] "+x" (out),
        : [s] "x" (s),
    );
    return out;
}

/// `pmaddubsw`: `u` read as unsigned, `s` as signed, adjacent products summed into
/// saturating i16.
inline fn maddubs(u: Bytes, s: Bytes) @Vector(8, i16) {
    if (comptime has_avx) {
        return asm ("vpmaddubsw %[s], %[u], %[out]"
            : [out] "=x" (-> @Vector(8, i16)),
            : [u] "x" (u),
              [s] "x" (s),
        );
    }
    var out: @Vector(8, i16) = @bitCast(u);
    asm ("pmaddubsw %[s], %[out]"
        : [out] "+x" (out),
        : [s] "x" (s),
    );
    return out;
}

/// `pmaddwd` against ones: adjacent i16 pairs summed into i32.
inline fn maddPairs(x: @Vector(8, i16)) Acc {
    const ones: @Vector(8, i16) = @splat(1);
    if (comptime has_avx) {
        return asm ("vpmaddwd %[ones], %[x], %[out]"
            : [out] "=x" (-> Acc),
            : [x] "x" (x),
              [ones] "x" (ones),
        );
    }
    var out: Acc = @bitCast(x);
    asm ("pmaddwd %[ones], %[out]"
        : [out] "+x" (out),
        : [ones] "x" (ones),
    );
    return out;
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

// -- 256-bit forms -----------------------------------------------------------
//
// The same arithmetic on 32 bytes at a time, for x86 targets with AVX2. The 128-bit
// chain above is several instructions per 16 bytes where `sdot` is one, so x86 runs
// out of issue slots long before it runs out of memory bandwidth; doing 32 bytes per
// instruction halves the instruction count. Where VNNI is available the
// `pmaddubsw`/`pmaddwd`/add chain becomes one `vpdpbusd`.
//
// Results are identical to the 128-bit path: the same products, summed in a
// different order, in integers.

pub const Acc8 = @Vector(8, i32);
pub const Bytes32 = @Vector(32, i8);

pub const has_avx2 = switch (builtin.cpu.arch) {
    .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .avx2),
    else => false,
};

/// `vpdpbusd` on ymm registers: the AVX-512 encoding needs VL as well as VNNI; the
/// VEX encoding (Alder Lake, Zen 5) is a feature of its own.
///
/// Only under LLVM. Zig's own x86_64 backend, which compiles Debug builds, has no
/// `vpdpbusd` in its assembler, so a Debug build for a VNNI CPU failed to compile. CI
/// found it the first time a runner drew one (a Granite Rapids). The fallback chain
/// gives the same integer sums, and release builds always go through LLVM.
const vnni_assembles = builtin.zig_backend == .stage2_llvm;
const has_vnni_evex = has_avx2 and vnni_assembles and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vnni) and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vl);
const has_vnni_vex = has_avx2 and vnni_assembles and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avxvnni);
pub const has_vnni = has_vnni_evex or has_vnni_vex;

/// `accumulate` on 32 bytes. `b` must not contain −128, as there.
pub inline fn accumulate32(acc: Acc8, a: Bytes32, b: Bytes32) Acc8 {
    comptime std.debug.assert(has_avx2);
    if (std.debug.runtime_safety) {
        const raw: [32]i8 = b;
        for (raw) |v| std.debug.assert(v != -128);
    }
    const magnitude: Bytes32 = @bitCast(@abs(a));
    const signed = asm ("vpsignb %[s], %[x], %[out]"
        : [out] "=x" (-> Bytes32),
        : [x] "x" (b),
          [s] "x" (a),
    );
    if (comptime has_vnni) {
        // acc += Σ4 u8(magnitude) · s8(signed), straight into i32: no i16 stage, so
        // no saturation to reason about either.
        var out = acc;
        if (comptime has_vnni_evex) {
            asm ("vpdpbusd %[s], %[u], %[acc]"
                : [acc] "+x" (out),
                : [u] "x" (magnitude),
                  [s] "x" (signed),
            );
        } else {
            asm ("{vex} vpdpbusd %[s], %[u], %[acc]"
                : [acc] "+x" (out),
                : [u] "x" (magnitude),
                  [s] "x" (signed),
            );
        }
        return out;
    }
    const pairs = asm ("vpmaddubsw %[s], %[u], %[out]"
        : [out] "=x" (-> @Vector(16, i16)),
        : [u] "x" (magnitude),
          [s] "x" (signed),
    );
    const ones: @Vector(16, i16) = @splat(1);
    const quads = asm ("vpmaddwd %[ones], %[x], %[out]"
        : [out] "=x" (-> Acc8),
        : [x] "x" (pairs),
          [ones] "x" (ones),
    );
    return acc + quads;
}

/// Fold eight lanes into the four of `Acc`, so a kernel can finish a 256-bit loop
/// with a 128-bit tail and sum both the same way.
pub inline fn fold(acc: Acc8) Acc {
    const lo: Acc = @shuffle(i32, acc, undefined, [4]i32{ 0, 1, 2, 3 });
    const hi: Acc = @shuffle(i32, acc, undefined, [4]i32{ 4, 5, 6, 7 });
    return lo + hi;
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
        // `b` is the query side, which never holds −128; `a` covers the full range.
        for (&bv) |*v| v.* = random.intRangeAtMost(i8, -127, 127);
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

test "every pair the contract allows is exact" {
    // Exhaustive over a in [-128, 127] and b in [-127, 127]. Broadcasting each pair to
    // all 16 lanes makes every i16 intermediate on the x86 path as large as that pair
    // can make it, so this covers the saturation edge (2·128·127 = 32512) as well as
    // the sign handling: a = -128 reaches pmaddubsw as unsigned 128, and a negative a
    // flips b's sign before the multiply.
    const zero: Acc = @splat(0);
    var a: i32 = -128;
    while (a <= 127) : (a += 1) {
        var b: i32 = -127;
        while (b <= 127) : (b += 1) {
            const av: [16]i8 = @splat(@intCast(a));
            const bv: [16]i8 = @splat(@intCast(b));
            const want: [4]i32 = @splat(4 * a * b);
            try testing.expectEqualSlices(i32, &want, &@as([4]i32, accumulate(zero, av, bv)));
        }
    }

    // Mixed signs within one pair, at the extremes, so the two products of a
    // pmaddubsw pair cancel rather than add.
    const mixed_a: [16]i8 = .{ -128, 127, -128, -128, 127, 127, -1, 1, -128, 0, 127, -127, 5, -5, 64, -64 };
    const mixed_b: [16]i8 = .{ 127, -127, -127, -127, 127, 127, -127, 127, 127, -127, -127, 127, -127, -127, 127, -127 };
    var reference: [4]i32 = @splat(0);
    for (0..16) |i| reference[i / 4] += @as(i32, mixed_a[i]) * @as(i32, mixed_b[i]);
    try testing.expectEqualSlices(i32, &reference, &@as([4]i32, accumulate(zero, mixed_a, mixed_b)));
}
