//! The one-bit scan: codes stored a column at a time, scored by table lookup.
//!
//! At one bit per dimension the per-vector kernels (`scan.zig`) spend most of their
//! work unpacking: each 16-byte load holds 128 codes, which takes eight lookups and
//! eight `sdot`s to score, about 1.75 vector instructions per vector-query pair per 16
//! dimensions. Scoring sixteen vectors at once does better. Stored so that one 16-byte
//! load holds the same byte of sixteen vectors (`index/codes.zig`), each of its two
//! nibbles is four dimensions of sixteen vectors, and a per-query table of the 16
//! possible partial sums of those four dimensions scores all sixteen in one lookup:
//!
//!     Σ_j p_j · b_j  =  Σ_g T_g[nibble_g]          T_g[m] = Σ_t p_{4g+t} · bit_t(m)
//!
//! That is FAISS's fast-scan, and it is the scalar codebook's argument in reverse
//! (`quant/packing.zig`): with two levels a table over four dimensions is as cheap to
//! hold as one over a single dimension at four bits. Measured on an Apple M5 at d=768,
//! these kernels score 1.95 times as fast as the 4x4 tile with a batch, and 3.7 times as
//! fast as the 8-vector tile for one query (bench/fastscan_bench.zig).
//!
//! ## The table
//!
//! Per group of four dimensions, the 16 sums are shifted to start at zero and
//! quantized to u8, with one scale for the whole query, so the sums accumulate in u16
//! lanes. Shifted, an entry is a sum of `|p|` over the dimensions whose bit agrees with
//! the sign of p, so no entry is negative and the largest is the group's Σ|p|. The
//! query's scale is the widest group's Σ|p| over 255.
//!
//!     Σ_j p_j · b_j  ≈  scale · S + Σ_g min_m T_g[m]
//!
//! and with the codebook's two levels c0 < c1, Σ_j p_j c[b_j] = c0 · Σ_j p_j +
//! (c1 − c0) · Σ_j p_j b_j. Rounding the table costs about what rounding the query to
//! int8 did: the same error against the exact sum within 2% (rms 0.0077 against 0.0076
//! on unit-scale sums), well under the one-bit codes' own error.
//!
//! Every path that scores these codes, batched, single, partitioned and by id, takes
//! the same integer sum and the same `Affine`, so they report the same bits.

const std = @import("std");
const builtin = @import("builtin");
const shuffle = @import("shuffle.zig");

/// Vectors per block: one 16-byte load holds the same byte of each.
pub const block = 16;

/// Table bytes per query for a one-bit code of `stride` bytes: two 16-entry tables
/// per byte.
pub fn lutLen(stride: usize) usize {
    return stride * 2 * 16;
}

/// What a table needs besides its entries to turn a sum back into ⟨p, b⟩.
pub const LutTerms = struct {
    /// One table step, in units of p.
    scale: f32,
    /// Σ over groups of each group's smallest sum.
    low: f32,
    /// Σ_j p_j, for the codebook's lower level.
    total: f32,
};

/// Build the query's tables from its rotated coordinates. `lut` holds `lutLen(stride)`
/// bytes; dimensions past `rotated.len` (the stride's padding) get tables of zeros, so
/// their zero bytes add nothing.
pub fn buildLut(rotated: []const f32, lut: []u8) LutTerms {
    const groups = lut.len / 16;
    std.debug.assert(lut.len % 16 == 0 and groups * 4 >= rotated.len);

    var widest: f32 = 0;
    var low: f64 = 0;
    var total: f64 = 0;
    for (0..groups) |g| {
        const p = groupOf(rotated, g);
        var range: f32 = 0;
        var neg: f32 = 0;
        for (p) |x| {
            range += @abs(x);
            neg += @min(x, 0);
            total += x;
        }
        widest = @max(widest, range);
        low += neg;
    }
    const scale: f32 = if (widest > 0) widest / 255.0 else 1.0;
    const inverse = 1.0 / scale;

    const F = @Vector(16, f32);
    for (0..groups) |g| {
        const p = groupOf(rotated, g);
        // Entry m sums |p_t| over the t whose bit in m agrees with p_t's sign, in t
        // order, so every platform and every lane width adds the same values the same
        // way.
        var e: F = @splat(0);
        inline for (0..4) |t| {
            const bit: @Vector(16, bool) = comptime blk: {
                var b: [16]bool = undefined;
                for (0..16) |m| b[m] = (m >> t) & 1 == 1;
                break :blk b;
            };
            const positive: @Vector(16, bool) = @splat(p[t] > 0);
            const agree = bit == positive;
            e += @select(f32, agree, @as(F, @splat(@abs(p[t]))), @as(F, @splat(0)));
        }
        const q = @floor(e * @as(F, @splat(inverse)) + @as(F, @splat(0.5)));
        const clamped = @min(q, @as(F, @splat(255)));
        const bytes: @Vector(16, u8) = @intFromFloat(clamped);
        lut[g * 16 ..][0..16].* = bytes;
    }
    return .{ .scale = scale, .low = @floatCast(low), .total = @floatCast(total) };
}

fn groupOf(rotated: []const f32, g: usize) [4]f32 {
    var p: [4]f32 = @splat(0);
    for (0..4) |t| {
        const j = g * 4 + t;
        if (j < rotated.len) p[t] = rotated[j];
    }
    return p;
}

/// A sum to ⟨p, ũ⟩: `a · S + b`, per query and codebook.
pub const Affine = struct {
    a: f32,
    b: f32,

    /// For the two-level codebook `low`, `low + step`.
    pub fn init(terms: LutTerms, low: f32, step: f32) Affine {
        return .{ .a = step * terms.scale, .b = low * terms.total + step * terms.low };
    }

    pub inline fn apply(self: Affine, sum: u32) f32 {
        return self.a * @as(f32, @floatFromInt(sum)) + self.b;
    }

    /// `apply` lane by lane, the same operations in the same order.
    pub inline fn applyVec(self: Affine, sums: @Vector(block, u32)) @Vector(block, f32) {
        const s: @Vector(block, f32) = @floatFromInt(sums);
        return @as(@Vector(block, f32), @splat(self.a)) * s + @as(@Vector(block, f32), @splat(self.b));
    }
};

/// The sum for one vector, lane `lane` of the block at `blk`: what the block kernel
/// computes for it, one byte at a time.
pub fn sumAt(lut: []const u8, blk: [*]const u8, lane: usize, stride: usize) u32 {
    std.debug.assert(lane < block and lut.len == lutLen(stride));
    var s: u32 = 0;
    for (0..stride) |k| {
        const byte = blk[k * block + lane];
        s += lut[(2 * k) * 16 + (byte & 0x0F)];
        s += lut[(2 * k + 1) * 16 + (byte >> 4)];
    }
    return s;
}

const U8x16 = @Vector(16, u8);
const U16x16 = @Vector(16, u16);
const U32x16 = @Vector(16, u32);

/// Bytes per pass before the u16 accumulators are folded into u32: each byte adds at
/// most 255 to each of two, and their sum must stay within 65535.
const pass = 128;

/// The sums for the 16 vectors of the block at `blk` against `Q` queries' tables.
pub inline fn blockSums(comptime Q: usize, blk: [*]const u8, stride: usize, luts: [Q][*]const u8) [Q]U32x16 {
    var total: [Q]U32x16 = @splat(@splat(0));
    const mask: U8x16 = @splat(0x0F);
    var start: usize = 0;
    while (start < stride) : (start += pass) {
        const end = @min(stride, start + pass);
        // Two accumulators per query, so consecutive adds do not wait on each other.
        var lo: [Q]U16x16 = @splat(@splat(0));
        var hi: [Q]U16x16 = @splat(@splat(0));
        for (start..end) |k| {
            const c: U8x16 = blk[k * block ..][0..16].*;
            const low_idx = c & mask;
            const high_idx = c >> @splat(4);
            inline for (0..Q) |q| {
                const tl: @Vector(16, i8) = @bitCast(@as(U8x16, luts[q][(2 * k) * 16 ..][0..16].*));
                const th: @Vector(16, i8) = @bitCast(@as(U8x16, luts[q][(2 * k + 1) * 16 ..][0..16].*));
                const rl: U8x16 = @bitCast(shuffle.table16(tl, low_idx));
                const rh: U8x16 = @bitCast(shuffle.table16(th, high_idx));
                lo[q] += @as(U16x16, rl);
                hi[q] += @as(U16x16, rh);
            }
        }
        inline for (0..Q) |q| total[q] += @as(U32x16, lo[q] + hi[q]);
    }
    return total;
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;

test "the block kernel's sums are the per-vector ones, for any stride" {
    var prng = std.Random.DefaultPrng.init(0xC01);
    const random = prng.random();
    // One pass, exactly one, and three with a short last one.
    for ([_]usize{ 16, 96, 128, 272 }) |stride| {
        const codes = try testing.allocator.alloc(u8, block * stride);
        defer testing.allocator.free(codes);
        random.bytes(codes);
        var luts: [3][]u8 = undefined;
        var ptrs: [3][*]const u8 = undefined;
        for (&luts, &ptrs) |*l, *p| {
            l.* = try testing.allocator.alloc(u8, lutLen(stride));
            // Worst case for overflow: every entry 255.
            @memset(l.*, 255);
            random.bytes(l.*[0 .. l.len / 2]);
            p.* = l.ptr;
        }
        defer for (luts) |l| testing.allocator.free(l);

        const got3 = blockSums(3, codes.ptr, stride, ptrs);
        const got1 = blockSums(1, codes.ptr, stride, .{ptrs[2]});
        const one: [block]u32 = got1[0];
        for (0..3) |q| {
            const lanes: [block]u32 = got3[q];
            for (0..block) |lane| {
                const want = sumAt(luts[q], codes.ptr, lane, stride);
                try testing.expectEqual(want, lanes[lane]);
                if (q == 2) try testing.expectEqual(want, one[lane]);
            }
        }
    }
}

test "the table gives ⟨p, b⟩ to within its rounding" {
    var prng = std.Random.DefaultPrng.init(0xC02);
    const random = prng.random();
    const dim = 200; // not a multiple of the stride's 8 dimensions per byte times 16
    const stride = 32;
    var p: [dim]f32 = undefined;
    for (&p) |*x| x.* = random.floatNorm(f32);
    const lut = try testing.allocator.alloc(u8, lutLen(stride));
    defer testing.allocator.free(lut);
    const terms = buildLut(&p, lut);

    var blk: [block * stride]u8 = @splat(0);
    for (0..block) |lane| {
        for (0..dim / 8) |k| blk[k * block + lane] = random.int(u8);
    }
    for (0..block) |lane| {
        var exact: f64 = 0;
        for (0..dim) |j| {
            const byte = blk[(j / 8) * block + lane];
            if ((byte >> @intCast(j % 8)) & 1 == 1) exact += p[j];
        }
        const s = sumAt(lut, &blk, lane, stride);
        const got = terms.scale * @as(f32, @floatFromInt(s)) + terms.low;
        // Half a step per group, at most.
        try testing.expectApproxEqAbs(@as(f32, @floatCast(exact)), got, 0.5 * terms.scale * (dim / 4) + 1e-4);
    }
    // The two-level form: c0·Σp + (c1−c0)·⟨p, b⟩.
    const aff = Affine.init(terms, -0.5, 1.0);
    const s0 = sumAt(lut, &blk, 0, stride);
    try testing.expectEqual(1.0 * terms.scale * @as(f32, @floatFromInt(s0)) + (-0.5 * terms.total + 1.0 * terms.low), aff.apply(s0));
}

test "a table's entries span 0 to 255 and the zero query gives zeros" {
    var lut: [lutLen(16)]u8 = undefined;
    const zero: [128]f32 = @splat(0);
    const t0 = buildLut(&zero, &lut);
    for (lut) |b| try testing.expectEqual(@as(u8, 0), b);
    try testing.expectEqual(@as(f32, 0), t0.total);

    var p: [128]f32 = undefined;
    for (&p, 0..) |*x, j| x.* = @as(f32, @floatFromInt(j % 7)) - 3;
    _ = buildLut(&p, &lut);
    var top: u8 = 0;
    for (lut) |b| top = @max(top, b);
    try testing.expectEqual(@as(u8, 255), top);
}
