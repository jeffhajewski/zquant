//! Dot products with the lists' centroids, and the list nearest each row.
//!
//! k-means and adding vectors both need, for each row y, the list whose centroid is
//! nearest: the largest ⟨y, c⟩ − ‖c‖²/2. Exactly, that's a dot product with every
//! centroid in f32, and at a million vectors and 1000 lists those took nine tenths of
//! building the index.
//!
//! `Nearest` gets the same answer for less. It first takes every dot product in int8,
//! rows and centroids each scaled to ±127, which `sdot` does sixteen at a time. How far
//! each was rounded bounds how far an approximate dot product can be from the exact one:
//!
//!     |⟨y, c⟩ − ⟨ỹ, c̃⟩|  ≤  ‖y − ỹ‖·‖c‖ + ‖ỹ‖·‖c − c̃‖
//!
//! which holds because ⟨y, c⟩ − ⟨ỹ, c̃⟩ = ⟨y − ỹ, c⟩ + ⟨ỹ, c − c̃⟩, plus a margin for
//! f32 rounding on both sides. A list whose key is bounded below the best lower bound
//! can neither be nearest nor tie for it. The rest, a dozen or so of a thousand on
//! embeddings, are scored exactly with `dotExact`, and the largest key wins, ties to the
//! lower list: the exhaustive answer, bit for bit, on every platform. Where `sdot` isn't
//! there to take the int8 products (see `bounded`), the exhaustive path runs instead.

const std = @import("std");
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");
const dot = @import("../simd/dot.zig");

pub const V = @Vector(16, f32);

/// ⟨a, b⟩ with a fixed summation order, so the same pair gives the same bits whichever
/// caller computes it and however its rows were blocked.
pub fn dotExact(a: []const f32, b: []const f32) f32 {
    var acc: V = @splat(0);
    var j: usize = 0;
    while (j + 16 <= a.len) : (j += 16) {
        const x: V = a[j..][0..16].*;
        const y: V = b[j..][0..16].*;
        acc += x * y;
    }
    return finish(acc, a[j..], b[j..]);
}

inline fn finish(acc: V, a_tail: []const f32, b_tail: []const f32) f32 {
    const lanes: [16]f32 = acc;
    var s: f32 = 0;
    for (lanes) |x| s += x;
    for (a_tail, b_tail) |x, y| s += x * y;
    return s;
}

/// `out[r * k + l] = ⟨rows[r], centroids[l]⟩` for every row of `rows` (row-major, `p`
/// wide) and each of `k` centroids. Rows go four at a time so each centroid load serves
/// four dot products; every result is bit-identical to `dotExact`.
pub fn centroidDots(rows: []const f32, p: usize, centroids: []const f32, k: usize, out: []f32) void {
    const n = rows.len / p;
    var r: usize = 0;
    while (r + 4 <= n) : (r += 4) {
        const a0 = rows[r * p ..][0..p];
        const a1 = rows[(r + 1) * p ..][0..p];
        const a2 = rows[(r + 2) * p ..][0..p];
        const a3 = rows[(r + 3) * p ..][0..p];
        for (0..k) |l| {
            const c = centroids[l * p ..][0..p];
            var s0: V = @splat(0);
            var s1: V = @splat(0);
            var s2: V = @splat(0);
            var s3: V = @splat(0);
            var j: usize = 0;
            while (j + 16 <= p) : (j += 16) {
                const cv: V = c[j..][0..16].*;
                s0 += @as(V, a0[j..][0..16].*) * cv;
                s1 += @as(V, a1[j..][0..16].*) * cv;
                s2 += @as(V, a2[j..][0..16].*) * cv;
                s3 += @as(V, a3[j..][0..16].*) * cv;
            }
            out[r * k + l] = finish(s0, a0[j..], c[j..]);
            out[(r + 1) * k + l] = finish(s1, a1[j..], c[j..]);
            out[(r + 2) * k + l] = finish(s2, a2[j..], c[j..]);
            out[(r + 3) * k + l] = finish(s3, a3[j..], c[j..]);
        }
    }
    while (r < n) : (r += 1) {
        const a = rows[r * p ..][0..p];
        for (0..k) |l| out[r * k + l] = dotExact(a, centroids[l * p ..][0..p]);
    }
}

/// The list nearest a row, given ⟨y, c_l⟩ for every list: the largest ⟨y, c⟩ − ‖c‖²/2,
/// which is the smallest ‖y − c‖². Ties go to the lower list.
pub fn nearest(dots: []const f32, centroid_sq: []const f32) usize {
    var best: usize = 0;
    var best_key = -std.math.inf(f32);
    for (dots, centroid_sq, 0..) |dt, sq, l| {
        const key = dt - 0.5 * sq;
        if (key > best_key) {
            best_key = key;
            best = l;
        }
    }
    return best;
}

/// Whether `Nearest` takes the int8 pre-pass on this target: where `sdot` takes the
/// products, 4.5 times as fast as the exhaustive path on an M5 (bench/coarse_bench.zig).
/// x86 has only the `pmaddubsw` chain, four instructions to `sdot`'s one. Under Rosetta
/// it ran at half the exhaustive path's speed, which says little about x86 itself, but
/// it hasn't been measured on x86 itself either, so x86 keeps the exhaustive path. Tests
/// run the pre-pass everywhere.
pub const bounded = builtin.cpu.arch.isAARCH64() and dot.accelerated;

/// Rows per micro-kernel: four where `sdot` leaves registers for sixteen accumulators,
/// two on x86, whose sixteen registers also hold the sign-and-magnitude temporaries.
const tile_rows = if (builtin.cpu.arch.isAARCH64()) 4 else 2;
/// A power of two, as `std.mem.alignForward` needs.
const tile_lists = 4;

/// Bytes of int8 centroids a pass keeps in L1 while rows stream past them.
const list_block_bytes = 32 * 1024;

/// Up to this many rows per `Nearest.assign` call.
pub const max_rows = 64;

/// Lists the bound is taken for at once.
const width = 8;
const F = @Vector(width, f32);

/// The nearest list for rows, by int8 pre-pass and exact recheck. `load` takes the
/// centroids; each worker then needs its own `Scratch`.
pub const Nearest = struct {
    k: usize,
    p: usize,
    /// `p` rounded up to 16: the int8 rows' width.
    p16: usize,
    /// `k` rounded up to `tile_lists`.
    k_pad: usize,
    codes: []i8 = &.{},
    /// Per list, what its bound needs, each rounded up but `scale`: c̃ = scale · codes,
    /// `err` = ‖c − c̃‖, `norm` = ‖c‖, and `mag` = ‖c‖ + ‖c − c̃‖, which bounds ‖c‖
    /// and ‖c̃‖ both. One array each, so the bound runs `width` lists at a time.
    scale: []f32 = &.{},
    err: []f32 = &.{},
    norm: []f32 = &.{},
    mag: []f32 = &.{},
    /// Take the exhaustive path, as a target without a native int8 dot product does.
    /// May be set (tests clear it) before making any `Scratch`.
    exhaustive: bool,

    pub fn init(a: Allocator, k: usize, p: usize) Allocator.Error!Nearest {
        const p16 = std.mem.alignForward(usize, p, 16);
        const k_pad = std.mem.alignForward(usize, k, tile_lists);
        const codes = try a.alloc(i8, k_pad * p16);
        errdefer a.free(codes);
        @memset(codes, 0);
        const terms = try a.alloc(f32, 4 * k);
        return .{
            .k = k,
            .p = p,
            .p16 = p16,
            .k_pad = k_pad,
            .codes = codes,
            .scale = terms[0..k],
            .err = terms[k..][0..k],
            .norm = terms[2 * k ..][0..k],
            .mag = terms[3 * k ..][0..k],
            // Past this width an int8 dot product could leave i32 (p16 · 127² ≥ 2³¹).
            .exhaustive = !bounded or p16 > 131072,
        };
    }

    pub fn deinit(self: *Nearest, a: Allocator) void {
        a.free(self.codes);
        a.free(self.scale.ptr[0 .. 4 * self.k]);
        self.* = undefined;
    }

    /// Quantize `centroids` (k rows, `p` wide). Call again whenever they change.
    pub fn load(self: *Nearest, centroids: []const f32) void {
        std.debug.assert(centroids.len == self.k * self.p);
        for (0..self.k) |l| {
            const q = quantize(centroids[l * self.p ..][0..self.p], self.codes[l * self.p16 ..][0..self.p16]);
            self.scale[l] = q.scale;
            self.err[l] = q.err;
            self.norm[l] = q.norm;
            self.mag[l] = q.norm + q.err;
        }
    }

    pub const Scratch = struct {
        codes: []i8,
        terms: []Row,
        dots: []i32,
        upper: []f32,
        exact: []f32,
        /// Lists scored exactly so far, for benchmarks.
        rechecked: usize = 0,

        pub fn init(a: Allocator, nearest_: *const Nearest) !Scratch {
            // Only what the path `nearest_` takes uses.
            const n_pad = std.mem.alignForward(usize, max_rows, tile_rows);
            const int8 = !nearest_.exhaustive;
            const codes = try a.alloc(i8, if (int8) n_pad * nearest_.p16 else 0);
            errdefer a.free(codes);
            const terms = try a.alloc(Row, if (int8) n_pad else 0);
            errdefer a.free(terms);
            const dots = try a.alloc(i32, if (int8) n_pad * nearest_.k_pad else 0);
            errdefer a.free(dots);
            const upper = try a.alloc(f32, if (int8) nearest_.k_pad else 0);
            errdefer a.free(upper);
            const exact = try a.alloc(f32, if (int8) 0 else max_rows * nearest_.k);
            return .{ .codes = codes, .terms = terms, .dots = dots, .upper = upper, .exact = exact };
        }

        pub fn deinit(self: *Scratch, a: Allocator) void {
            a.free(self.codes);
            a.free(self.terms);
            a.free(self.dots);
            a.free(self.upper);
            a.free(self.exact);
            self.* = undefined;
        }
    };

    /// The int8 half of `assign`: quantize the rows into `s` and take their int8 dot
    /// products with every list, `s.dots[r · k_pad + l]`. Public for benchmarks.
    pub fn int8Pass(self: *const Nearest, data: []const f32, s: *Scratch) void {
        const n = data.len / self.p;
        const n_pad = std.mem.alignForward(usize, n, tile_rows);
        for (0..n) |r| {
            s.terms[r] = quantize(data[r * self.p ..][0..self.p], s.codes[r * self.p16 ..][0..self.p16]);
        }
        @memset(s.codes[n * self.p16 .. n_pad * self.p16], 0);
        int8Dots(s.codes, n_pad, self.codes, self.k_pad, self.p16, s.dots);
    }

    /// `out[r]` = the list nearest `rows[r]` (row-major, `p` wide, at most `max_rows`
    /// rows): the largest ⟨y, c_l⟩ − ‖c_l‖²/2 with ⟨y, c_l⟩ by `dotExact` and
    /// `centroid_sq[l]` as given, ties to the lower list. Exactly `nearest` over
    /// `centroidDots`, whichever path runs.
    pub fn assign(self: *const Nearest, data: []const f32, centroids: []const f32, centroid_sq: []const f32, s: *Scratch, out: []u32) void {
        const p = self.p;
        const k = self.k;
        const n = data.len / p;
        std.debug.assert(n <= max_rows and out.len == n and centroid_sq.len == k);
        if (self.exhaustive) {
            centroidDots(data, p, centroids, k, s.exact[0 .. n * k]);
            for (0..n) |r| out[r] = @intCast(nearest(s.exact[r * k ..][0..k], centroid_sq));
            return;
        }

        self.int8Pass(data, s);

        // Twice the rounding a dot product of p terms and the subtraction can carry,
        // relative to the largest magnitude involved; see the module comment.
        const rho: f32 = @as(f32, @floatFromInt(self.p16 / 16 + 64)) * 0x1p-23;
        const widen: f32 = 1 + 0x1p-10;
        const whole = k / width * width;
        for (0..n) |r| {
            const y = data[r * p ..][0..p];
            const row = s.terms[r];
            const dots = s.dots[r * self.k_pad ..][0..k];
            // Each list's key lies within [approx − eps, approx + eps]: keep the upper
            // ends, and the best of the lower ends.
            var lowest: F = @splat(-std.math.inf(f32));
            var l: usize = 0;
            while (l < whole) : (l += width) {
                const d: F = @floatFromInt(@as(@Vector(width, i32), dots[l..][0..width].*));
                const half_sq = @as(F, @splat(0.5)) * @as(F, centroid_sq[l..][0..width].*);
                const terms = bound(d, half_sq, row, self.scale[l..][0..width].*, self.err[l..][0..width].*, self.norm[l..][0..width].*, self.mag[l..][0..width].*, rho, widen);
                s.upper[l..][0..width].* = terms[0] + terms[1];
                lowest = @max(lowest, terms[0] - terms[1]);
            }
            var lower = @reduce(.Max, lowest);
            while (l < k) : (l += 1) {
                const d: F = @splat(@floatFromInt(dots[l]));
                const half_sq: F = @splat(0.5 * centroid_sq[l]);
                const terms = bound(d, half_sq, row, @splat(self.scale[l]), @splat(self.err[l]), @splat(self.norm[l]), @splat(self.mag[l]), rho, widen);
                s.upper[l] = terms[0][0] + terms[1][0];
                lower = @max(lower, terms[0][0] - terms[1][0]);
            }

            // Score exactly every list whose upper end reaches the best lower end, in
            // list order, skipping `width` at a time where none does.
            var best: usize = 0;
            var best_key = -std.math.inf(f32);
            l = 0;
            while (l < k) {
                if (l + width <= k and @reduce(.Max, @as(F, s.upper[l..][0..width].*)) < lower) {
                    l += width;
                    continue;
                }
                const end = @min(k, l + width);
                while (l < end) : (l += 1) {
                    if (s.upper[l] < lower) continue;
                    s.rechecked += 1;
                    const key = dotExact(y, centroids[l * p ..][0..p]) - 0.5 * centroid_sq[l];
                    if (key > best_key) {
                        best_key = key;
                        best = l;
                    }
                }
            }
            out[r] = @intCast(best);
        }
    }
};

/// The approximate key and its error bound for `width` lists against one row; see the
/// module comment. Both are evaluated in f32, which `widen` covers.
inline fn bound(d: F, half_sq: F, row: Row, scale: F, err: F, norm: F, mag: F, rho: f32, widen: f32) [2]F {
    const approx = d * (@as(F, @splat(row.scale)) * scale) - half_sq;
    const cauchy = @as(F, @splat(row.err)) * norm + @as(F, @splat(row.norm_q)) * err;
    const rounding = @as(F, @splat(rho)) * (@as(F, @splat(row.norm + row.err)) * mag + @abs(half_sq));
    return .{ approx, (cauchy + rounding) * @as(F, @splat(widen)) };
}

const Row = struct {
    scale: f32,
    /// ‖y − ỹ‖, rounded up.
    err: f32,
    /// ‖y‖, rounded up.
    norm: f32,
    /// ‖ỹ‖, rounded up.
    norm_q: f32,
};

/// `codes` = x scaled to ±127 and rounded, zero past `x.len`; returns the scale and the
/// norms the bound uses, each rounded up.
fn quantize(x: []const f32, codes: []i8) Row {
    var widest: f32 = 0;
    for (x) |v| widest = @max(widest, @abs(v));
    const scale: f32 = widest / 127;
    const inv: f32 = if (scale > 0) 1 / scale else 0;
    var err: f64 = 0;
    var norm: f64 = 0;
    var norm_q: f64 = 0;
    for (x, codes[0..x.len]) |v, *c| {
        const q = std.math.clamp(@round(v * inv), -127, 127);
        c.* = @intFromFloat(q);
        const rec = @as(f64, scale) * q;
        err += (v - rec) * (v - rec);
        norm += @as(f64, v) * v;
        norm_q += rec * rec;
    }
    @memset(codes[x.len..], 0);
    return .{ .scale = scale, .err = up(err), .norm = up(norm), .norm_q = up(norm_q) };
}

/// √x as f32, rounded up with room to spare for the f64 sum's own rounding.
fn up(sum_sq: f64) f32 {
    return @floatCast(@sqrt(sum_sq) * (1 + 0x1p-20));
}

/// `out[r · k_pad + l]` = Σ_j rows[r][j] · lists[l][j] for `n_pad` rows (a multiple of
/// `tile_rows`) and `k_pad` lists (a multiple of `tile_lists`), each `p16` bytes. Lists go
/// in blocks that stay in L1 while every row passes them.
fn int8Dots(rows: []const i8, n_pad: usize, lists: []const i8, k_pad: usize, p16: usize, out: []i32) void {
    std.debug.assert(n_pad % tile_rows == 0 and k_pad % tile_lists == 0 and p16 % 16 == 0);
    const block = @max(tile_lists, list_block_bytes / p16 / tile_lists * tile_lists);
    var l0: usize = 0;
    while (l0 < k_pad) : (l0 += block) {
        const l1 = @min(k_pad, l0 + block);
        var r: usize = 0;
        while (r < n_pad) : (r += tile_rows) {
            var l = l0;
            while (l < l1) : (l += tile_lists) {
                tile(rows[r * p16 ..].ptr, lists[l * p16 ..].ptr, p16, out[r * k_pad + l ..].ptr, k_pad);
            }
        }
    }
}

inline fn tile(rows: [*]const i8, lists: [*]const i8, p16: usize, out: [*]i32, out_stride: usize) void {
    var acc: [tile_rows][tile_lists]dot.Acc = @splat(@splat(@splat(0)));
    var j: usize = 0;
    while (j < p16) : (j += 16) {
        var a: [tile_rows]dot.Bytes = undefined;
        inline for (0..tile_rows) |u| a[u] = rows[u * p16 + j ..][0..16].*;
        inline for (0..tile_lists) |c| {
            const b: dot.Bytes = lists[c * p16 + j ..][0..16].*;
            inline for (0..tile_rows) |u| acc[u][c] = dot.accumulate(acc[u][c], a[u], b);
        }
    }
    inline for (0..tile_rows) |u| {
        inline for (0..tile_lists) |c| out[u * out_stride + c] = dot.total(acc[u][c]);
    }
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;

fn exhaustiveNearest(data: []const f32, p: usize, centroids: []const f32, sq: []const f32, out: []u32) !void {
    const k = sq.len;
    const n = data.len / p;
    const dots = try testing.allocator.alloc(f32, n * k);
    defer testing.allocator.free(dots);
    centroidDots(data, p, centroids, k, dots);
    for (0..n) |r| out[r] = @intCast(nearest(dots[r * k ..][0..k], sq));
}

test "the pre-pass finds the exhaustive nearest list, ties included" {
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const random = prng.random();
    const a = testing.allocator;
    // Widths below, at and past 16, lists not a multiple of four, and a few centroids
    // repeated or nudged so exact ties and near-ties come up.
    for ([_][2]usize{ .{ 4, 3 }, .{ 40, 7 }, .{ 64, 33 }, .{ 768, 50 }, .{ 200, 129 } }) |shape| {
        const p = shape[0];
        const k = shape[1];
        const n = 300;
        const centroids = try a.alloc(f32, k * p);
        defer a.free(centroids);
        for (centroids) |*c| c.* = random.floatNorm(f32) * 0.3;
        @memcpy(centroids[(k - 1) * p ..][0..p], centroids[0..p]);
        if (k > 2) for (centroids[(k - 2) * p ..][0..p], centroids[p..][0..p], 0..) |*o, c, j| {
            o.* = c * (1 + if (j % 2 == 0) @as(f32, 1.0 / 1024.0) else -1.0 / 1024.0);
        };
        const sq = try a.alloc(f32, k);
        defer a.free(sq);
        for (0..k) |l| sq[l] = dotExact(centroids[l * p ..][0..p], centroids[l * p ..][0..p]);

        // Rows near the centroids, a row exactly on one, a row midway between the
        // repeated pair's neighbours, and a zero row.
        const data = try a.alloc(f32, n * p);
        defer a.free(data);
        for (0..n) |r| {
            const c = random.uintLessThan(usize, k);
            for (data[r * p ..][0..p], centroids[c * p ..][0..p]) |*x, cv| x.* = cv + 0.05 * random.floatNorm(f32);
        }
        @memcpy(data[0..p], centroids[0..p]);
        for (data[p..][0..p], centroids[0..p], centroids[p..][0..p]) |*x, c0, c1| x.* = 0.5 * (c0 + c1);
        @memset(data[2 * p ..][0..p], 0);

        var near = try Nearest.init(a, k, p);
        defer near.deinit(a);
        near.exhaustive = false;
        near.load(centroids);
        var scratch = try Nearest.Scratch.init(a, &near);
        defer scratch.deinit(a);

        const want = try a.alloc(u32, n);
        defer a.free(want);
        try exhaustiveNearest(data, p, centroids, sq, want);
        const got = try a.alloc(u32, n);
        defer a.free(got);
        var r: usize = 0;
        while (r < n) : (r += max_rows) {
            const take = @min(max_rows, n - r);
            near.assign(data[r * p ..][0 .. take * p], centroids, sq, &scratch, got[r..][0..take]);
        }
        try testing.expectEqualSlices(u32, want, got);
        // The zero row ties every list at ⟨0, c⟩ = 0 and goes to the smallest ‖c‖.
        try testing.expectEqual(want[2], got[2]);
    }
}

test "int8 tiles sum exactly" {
    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();
    const p16 = 48;
    const n_pad = 2 * tile_rows;
    const k_pad = 3 * tile_lists;
    var rows: [n_pad * p16]i8 = undefined;
    var lists: [k_pad * p16]i8 = undefined;
    for (&rows) |*v| v.* = random.intRangeAtMost(i8, -127, 127);
    for (&lists) |*v| v.* = random.intRangeAtMost(i8, -127, 127);
    var out: [n_pad * k_pad]i32 = undefined;
    int8Dots(&rows, n_pad, &lists, k_pad, p16, &out);
    for (0..n_pad) |r| for (0..k_pad) |l| {
        var want: i32 = 0;
        for (0..p16) |j| want += @as(i32, rows[r * p16 + j]) * lists[l * p16 + j];
        try testing.expectEqual(want, out[r * k_pad + l]);
    };
}
