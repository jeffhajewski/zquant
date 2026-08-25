//! 16-entry byte table lookup.
//!
//! The single primitive the scan kernel is built on: given 16 centroid values and 16
//! code indices, produce the 16 corresponding values in one instruction.
//!
//! ## Why this is inline assembly again
//!
//! docs/DESIGN.md §4.2 originally specified per-architecture assembly behind a
//! dispatch layer. During P1 that looked unnecessary — Zig 0.15.2 let a vector be
//! indexed by a runtime value, and LLVM pattern-matched
//!
//!     inline for (0..16) |i| out[i] = table[idx[i]];
//!
//! into exactly `tbl` / `pshufb` / `vpshufb`. So the dispatch layer was deleted as
//! machinery the compiler made redundant.
//!
//! **Zig 0.16 removed runtime vector indexing**, and that construct is now a compile
//! error. The obvious port — stage the vector through an array, which still permits
//! runtime indexing — compiles but does *not* recover the instruction: it spills to
//! the stack and issues scalar loads, which measured ~14× slower when this lookup
//! previously failed to vectorize. So the original design was right, and the
//! simplification did not survive one minor compiler release.
//!
//! ## Semantics
//!
//! All three paths agree only while every index is in 0..15:
//!
//!   - `tbl` zeroes a lane whose index is ≥ 16.
//!   - `pshufb` zeroes a lane whose index has bit 7 set, and otherwise uses the low
//!     four bits — so index 16 reads entry 0 rather than zeroing.
//!   - the scalar fallback would read out of bounds.
//!
//! Codes are at most 4 bits by construction (`scan.max_table_bits`), so this holds.
//! It is asserted rather than assumed, because the three disagree in exactly the way
//! that would produce plausible wrong answers instead of a crash.

const std = @import("std");
const builtin = @import("builtin");

pub const Vec = @Vector(16, i8);
pub const Idx = @Vector(16, u8);

const has_neon = switch (builtin.cpu.arch) {
    .aarch64, .aarch64_be => true,
    else => false,
};

const has_ssse3 = switch (builtin.cpu.arch) {
    .x86, .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .ssse3),
    else => false,
};

/// Whether this target has a single-instruction path. False means correct but slow.
pub const accelerated = has_neon or has_ssse3;

/// `out[i] = table[idx[i]]` for all 16 lanes. Every index must be < 16.
pub inline fn table16(table: Vec, idx: Idx) Vec {
    if (std.debug.runtime_safety) {
        const raw: [16]u8 = idx;
        for (raw) |v| std.debug.assert(v < 16);
    }

    if (comptime has_neon) {
        return asm ("tbl %[out].16b, {%[t].16b}, %[i].16b"
            : [out] "=w" (-> Vec),
            : [t] "w" (table),
              [i] "w" (idx),
        );
    }

    if (comptime has_ssse3) {
        // Two-operand form: the table register is read and written.
        var t = table;
        asm ("pshufb %[i], %[t]"
            : [t] "+x" (t),
            : [i] "x" (idx),
        );
        return t;
    }

    return scalar(table, idx);
}

/// Portable fallback. Correct everywhere, one load per lane.
fn scalar(table: Vec, idx: Idx) Vec {
    const values: [16]i8 = table;
    const indices: [16]u8 = idx;
    var out: [16]i8 = undefined;
    for (&out, indices) |*o, i| o.* = values[i];
    return out;
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;

test "table16 matches a scalar lookup" {
    // The accelerated paths and the fallback must be indistinguishable, or codes
    // decode to the wrong centroids everywhere.
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const random = prng.random();

    for (0..500) |_| {
        var values: [16]i8 = undefined;
        var indices: [16]u8 = undefined;
        for (&values) |*v| v.* = random.intRangeAtMost(i8, -128, 127);
        for (&indices) |*v| v.* = random.uintAtMost(u8, 15);

        const got: [16]i8 = table16(values, indices);
        const want: [16]i8 = scalar(values, indices);
        try testing.expectEqualSlices(i8, &want, &got);
        for (got, indices) |g, i| try testing.expectEqual(values[i], g);
    }
}

test "constant and identity patterns" {
    const values: [16]i8 = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    // Identity permutation returns the table unchanged.
    const identity: [16]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    try testing.expectEqualSlices(i8, &values, &@as([16]i8, table16(values, identity)));

    // A constant index broadcasts one entry.
    for ([_]u8{ 0, 7, 15 }) |k| {
        const all: [16]u8 = @splat(k);
        const got: [16]i8 = table16(values, all);
        for (got) |g| try testing.expectEqual(@as(i8, @intCast(k)), g);
    }

    // Reversal, to catch any lane-order confusion between architectures.
    var reversed: [16]u8 = undefined;
    for (&reversed, 0..) |*r, i| r.* = @intCast(15 - i);
    const got: [16]i8 = table16(values, reversed);
    for (got, 0..) |g, i| try testing.expectEqual(@as(i8, @intCast(15 - i)), g);
}

test "negative table values survive the round trip" {
    // The centroid table is signed and mostly negative in its lower half; a path
    // that treated it as unsigned would pass the identity test and fail here.
    var values: [16]i8 = undefined;
    for (&values, 0..) |*v, i| v.* = @intCast(@as(i32, @intCast(i)) - 8);
    var indices: [16]u8 = undefined;
    for (&indices, 0..) |*x, i| x.* = @intCast((i * 7) % 16);

    const got: [16]i8 = table16(values, indices);
    for (got, indices) |g, i| try testing.expectEqual(values[i], g);
}
