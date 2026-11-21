//! The one-bit scan, by table lookup over vectors side by side, against the row-major
//! int8 kernels it replaced, and the layouts considered on the way.
//!
//! The row-major kernels score one vector at a time: per 16 dimensions, unpack the codes
//! (shared across a 4-query tile) and one `sdot` per query, about 1.75 vector ops per
//! vector-query pair. FAISS's fast-scan instead stores 32 vectors' codes side by side,
//! 4 dimensions to a nibble, and looks up a per-query table of partial sums: one `tbl`
//! scores 16 vectors for 4 dimensions. The library stores 16 vectors by byte column
//! (`src/index/codes.zig`) and runs `simd/columns.zig`; the last two rows below are it.
//! docs/notes.md, "A one-bit kernel", has the numbers.

const std = @import("std");
const builtin = @import("builtin");
const zq = @import("zquant");
const Timer = @import("timer.zig").Timer;

const dim: u32 = 768;
const n: usize = 100_000;
const nq: usize = 32;
const trials = 7;
const block = 32;
const groups = dim / 4;

const U8x16 = @Vector(16, u8);
const U16x8 = @Vector(8, u16);

pub fn main() !void {
    const a = std.heap.smp_allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();

    // Random 1-bit codes, per vector: the current layout packs them sequentially.
    const layout = zq.packing.Layout.init(dim, 1);
    const stride = layout.stride();
    const bits = try a.alloc(u8, n * dim); // one byte per code bit, for building layouts
    defer a.free(bits);
    for (bits) |*b| b.* = random.uintLessThan(u8, 2);
    const codes = try a.alloc(u8, n * stride);
    defer a.free(codes);
    for (0..n) |v| layout.pack(bits[v * dim ..][0..dim], codes[v * stride ..][0..stride]);

    // The side-by-side layout: per block of 32 vectors, per group of 4 dimensions, 16
    // bytes; byte j holds vector j's nibble low and vector j+16's high.
    const blocks = n / block;
    const fs = try a.alloc(u8, blocks * groups * 16);
    defer a.free(fs);
    for (0..blocks) |bk| for (0..groups) |g| for (0..16) |j| {
        var lo: u8 = 0;
        var hi: u8 = 0;
        for (0..4) |t| {
            lo |= bits[(bk * block + j) * dim + g * 4 + t] << @intCast(t);
            hi |= bits[(bk * block + j + 16) * dim + g * 4 + t] << @intCast(t);
        }
        fs[(bk * groups + g) * 16 + j] = lo | (hi << 4);
    };

    // Queries: rotated unit vectors, roughly.
    const rotated = try a.alloc(f32, nq * dim);
    defer a.free(rotated);
    for (rotated) |*x| x.* = random.floatNorm(f32) / @sqrt(@as(f32, dim));

    // Current path: the 4x4 batch tile through the Job interface.
    const codebook = [_]f32{ -0.03, 0.03 };
    const table = zq.simd_scan.Table.init(&codebook);
    const queries = try a.alloc(zq.simd_scan.Query, nq);
    defer a.free(queries);
    for (queries, 0..) |*q, i| {
        q.* = try zq.simd_scan.Query.init(a, layout);
        q.load(rotated[i * dim ..][0..dim]);
    }
    defer for (queries) |*q| q.deinit();
    const out = try a.alloc(f32, n * nq);
    defer a.free(out);
    const single_out = try a.alloc(f32, n);
    defer a.free(single_out);

    var best_cur: u64 = std.math.maxInt(u64);
    for (0..trials) |_| {
        var t = Timer.start();
        var job: zq.simd_tiles.Job = .{
            .table = &table,
            .queries = queries.ptr,
            .n = nq,
            .codes = codes.ptr,
            .stride = stride,
            .padded = dim,
            .bits = 1,
            .expanded = false,
            .table_scale = table.scale,
            .out = out.ptr,
            .out_stride = nq,
        };
        var v: usize = 0;
        while (v + 4 <= n) : (v += 4) {
            job.out = out.ptr + v * nq;
            zq.simd_tiles.batchTile(&job, v);
        }
        best_cur = @min(best_cur, t.read());
    }
    const cur_rate = @as(f64, @floatFromInt(n * nq)) / (@as(f64, @floatFromInt(best_cur)) / 1e9) / 1e6;

    // Fast-scan: per query, per group, 16 u8 partial sums; dot = scale * sum - offset.
    const luts = try a.alloc(u8, nq * groups * 16);
    defer a.free(luts);
    const lut_scale = try a.alloc(f32, nq);
    defer a.free(lut_scale);
    const lut_offset = try a.alloc(f32, nq);
    defer a.free(lut_offset);
    const fs_out = try a.alloc(u32, n * nq);
    defer a.free(fs_out);

    for (0..nq) |i| buildLut(rotated[i * dim ..][0..dim], luts[i * groups * 16 ..][0 .. groups * 16], &lut_scale[i], &lut_offset[i]);
    const fs_rate = rate(blocks * block * nq, timeBlocked(4, fs, luts, fs_out));
    const fs1_rate = rate(blocks * block * nq, timeBlocked(1, fs, luts, fs_out));
    const pk_out = try a.alloc(u32, n * nq);
    defer a.free(pk_out);
    const pk1_rate = rate(n * nq, timePacked(1, codes, stride, luts, pk_out));
    const pk_rate = rate(n * nq, timePacked(4, codes, stride, luts, pk_out));
    const pk2_rate = rate(n * nq, timePacked(2, codes, stride, luts, pk_out));
    for (0..n * nq) |i| if (pk_out[i] != fs_out[i]) {
        std.debug.print("packed and blocked disagree at {d}: {d} vs {d}\n", .{ i, pk_out[i], fs_out[i] });
        return error.Mismatch;
    };

    // Stored as byte columns: per block of 16 vectors, byte k of each, 16 bytes per k.
    const t16 = try a.alloc(u8, n * stride);
    defer a.free(t16);
    for (0..n / 16) |bk| for (0..stride) |k| for (0..16) |j| {
        t16[(bk * stride + k) * 16 + j] = codes[(bk * 16 + j) * stride + k];
    };
    const c16_4 = rate(n * nq, timeColumns(4, t16, stride, luts, pk_out));
    try same(pk_out, fs_out);
    const c16_2 = rate(n * nq, timeColumns(2, t16, stride, luts, pk_out));
    try same(pk_out, fs_out);
    const c16_1 = rate(n * nq, timeColumns(1, t16, stride, luts, pk_out));
    try same(pk_out, fs_out);

    // What the library runs on that layout: `tiles.columnTile` over every query and
    // `columnSingle` for one, with `scan.Query`'s own tables.
    const lib_out = try a.alloc(f32, 16 * nq);
    defer a.free(lib_out);
    var best_lib: u64 = std.math.maxInt(u64);
    var best_lib1: u64 = std.math.maxInt(u64);
    for (0..trials) |_| {
        var job: zq.simd_tiles.Job = .{
            .table = &table,
            .queries = queries.ptr,
            .n = nq,
            .codes = t16.ptr,
            .stride = stride,
            .padded = dim,
            .bits = 1,
            .expanded = false,
            .table_scale = table.scale,
            .out = lib_out.ptr,
            .out_stride = nq,
        };
        var t = Timer.start();
        for (0..n / 16) |b| zq.simd_tiles.columnTile(&job, b);
        best_lib = @min(best_lib, t.read());
        t = Timer.start();
        for (0..nq) |i| {
            job.queries = queries.ptr + i;
            job.n = 1;
            job.out = single_out.ptr;
            zq.simd_tiles.columnSingle(&job, 0, n / 16);
        }
        best_lib1 = @min(best_lib1, t.read());
    }
    const lib_rate = rate(n * nq, best_lib);
    const lib1_rate = rate(n * nq, best_lib1);

    // The row-major kernel one query at a time, through `singleTiles`.
    var best_single: u64 = std.math.maxInt(u64);
    for (0..trials) |_| {
        var t = Timer.start();
        for (0..nq) |i| {
            var job: zq.simd_tiles.Job = .{
                .table = &table,
                .queries = queries.ptr + i,
                .n = 1,
                .codes = codes.ptr,
                .stride = stride,
                .padded = dim,
                .bits = 1,
                .expanded = false,
                .table_scale = table.scale,
                .out = single_out.ptr,
                .out_stride = 1,
            };
            const per = zq.simd_tiles.single_tile * 64;
            var v: usize = 0;
            while (v + per <= n) : (v += per) {
                job.out = single_out.ptr + v;
                zq.simd_tiles.singleTiles(&job, v, 64);
            }
        }
        best_single = @min(best_single, t.read());
    }
    const single_rate = rate(n / (zq.simd_tiles.single_tile * 64) * zq.simd_tiles.single_tile * 64 * nq, best_single);

    // Accuracy: both against the exact ±1 dot product.
    var err_cur: f64 = 0;
    var err_fs: f64 = 0;
    var mag: f64 = 0;
    const check = 2000;
    for (0..check) |v| for (0..nq) |i| {
        var exact: f64 = 0;
        for (0..dim) |j| {
            const s: f64 = if (bits[v * dim + j] == 1) 1 else -1;
            exact += s * rotated[i * dim + j];
        }
        const cur = out[v * nq + i] / 0.03;
        const got = lut_scale[i] * @as(f32, @floatFromInt(fs_out[v * nq + i])) - lut_offset[i];
        err_cur += (cur - exact) * (cur - exact);
        err_fs += (got - exact) * (got - exact);
        mag += exact * exact;
    };
    std.debug.print("fastscan_bench: {s} {s}, {d} x {d}, {d} queries, M scores/s\n", .{ @tagName(builtin.cpu.arch), builtin.cpu.model.name, n, dim, nq });
    std.debug.print("  row-major int8 4x4 tile            {d:7.1}   one query {d:7.1}   rms error {d:.5} of rms {d:.4}\n", .{ cur_rate, single_rate, @sqrt(err_cur / (check * nq)), @sqrt(mag / (check * nq)) });
    std.debug.print("  FAISS's layout, Q=4                {d:7.1}   Q=1       {d:7.1}   rms error {d:.5}\n", .{ fs_rate, fs1_rate, @sqrt(err_fs / (check * nq)) });
    std.debug.print("  row-major, transposed at search    {d:7.1}   Q=2 {d:7.1}   Q=1 {d:7.1}\n", .{ pk_rate, pk2_rate, pk1_rate });
    std.debug.print("  16 by byte column, Q=4             {d:7.1}   Q=2 {d:7.1}   Q=1 {d:7.1}\n", .{ c16_4, c16_2, c16_1 });
    std.debug.print("  the library: columnTile            {d:7.1}   columnSingle {d:7.1}   ({d:.2}x and {d:.2}x)\n", .{ lib_rate, lib1_rate, lib_rate / cur_rate, lib1_rate / single_rate });
}

/// Per group g and nibble m, Σ_t ±q[4g+t] (sign by bit t of m), shifted by Σ_t |q| to be
/// non-negative and quantized to u8 with one scale for the query.
fn buildLut(q: []const f32, lut: []u8, scale: *f32, offset: *f32) void {
    var widest: f32 = 0;
    var total: f32 = 0;
    for (0..groups) |g| {
        var s: f32 = 0;
        for (q[g * 4 ..][0..4]) |x| s += @abs(x);
        widest = @max(widest, 2 * s);
        total += s;
    }
    const sc = if (widest > 0) widest / 255.0 else 1;
    const inv = 1 / sc;
    for (0..groups) |g| {
        var shift: f32 = 0;
        for (q[g * 4 ..][0..4]) |x| shift += @abs(x);
        for (0..16) |m| {
            var v: f32 = shift;
            for (0..4) |t| v += if ((m >> @intCast(t)) & 1 == 1) q[g * 4 + t] else -q[g * 4 + t];
            lut[g * 16 + m] = @intFromFloat(@round(std.math.clamp(v * inv, 0, 255)));
        }
    }
    scale.* = sc;
    offset.* = total;
}

inline fn widenLo(x: U8x16) U16x8 {
    return @intCast(@shuffle(u8, x, undefined, [8]i32{ 0, 1, 2, 3, 4, 5, 6, 7 }));
}
inline fn widenHi(x: U8x16) U16x8 {
    return @intCast(@shuffle(u8, x, undefined, [8]i32{ 8, 9, 10, 11, 12, 13, 14, 15 }));
}

/// One block of 32 vectors against Q queries.
fn scanBlock(comptime Q: usize, codes: []const u8, luts: []const u8, out: []u32, out_stride: usize) void {
    var acc: [Q][4]U16x8 = @splat(@splat(@splat(0)));
    const mask: U8x16 = @splat(0x0F);
    for (0..groups) |g| {
        const c: U8x16 = codes[g * 16 ..][0..16].*;
        const lo = c & mask;
        const hi = c >> @splat(4);
        inline for (0..Q) |q| {
            const t: @Vector(16, i8) = @bitCast(@as(U8x16, luts[(q * groups + g) * 16 ..][0..16].*));
            const rl: U8x16 = @bitCast(zq.simd_shuffle.table16(t, lo));
            const rh: U8x16 = @bitCast(zq.simd_shuffle.table16(t, hi));
            acc[q][0] += widenLo(rl);
            acc[q][1] += widenHi(rl);
            acc[q][2] += widenLo(rh);
            acc[q][3] += widenHi(rh);
        }
    }
    inline for (0..Q) |q| {
        inline for (0..4) |part| {
            const lanes: [8]u16 = acc[q][part];
            for (0..8) |l| {
                // part 0: vectors 0..7, 1: 8..15, 2: 16..23, 3: 24..31
                out[(part * 8 + l) * out_stride + q] = lanes[l];
            }
        }
    }
}

fn rate(scores: usize, ns: u64) f64 {
    return @as(f64, @floatFromInt(scores)) / (@as(f64, @floatFromInt(ns)) / 1e9) / 1e6;
}

fn timeBlocked(comptime Q: usize, fs: []const u8, luts: []const u8, fs_out: []u32) u64 {
    const blocks = n / block;
    var best: u64 = std.math.maxInt(u64);
    for (0..trials) |_| {
        var t = Timer.start();
        for (0..blocks) |bk| {
            var q: usize = 0;
            while (q + Q <= nq) : (q += Q) {
                scanBlock(Q, fs[bk * groups * 16 ..][0 .. groups * 16], luts[q * groups * 16 ..], fs_out[bk * block * nq + q ..], nq);
            }
        }
        best = @min(best, t.read());
    }
    return best;
}

fn timePacked(comptime Q: usize, codes: []const u8, stride: usize, luts: []const u8, out: []u32) u64 {
    var best: u64 = std.math.maxInt(u64);
    for (0..trials) |_| {
        var t = Timer.start();
        var v: usize = 0;
        while (v + 16 <= n) : (v += 16) {
            var q: usize = 0;
            while (q + Q <= nq) : (q += Q) {
                scanPacked(Q, codes.ptr + v * stride, stride, luts.ptr + q * groups * 16, out[v * nq + q ..], nq);
            }
        }
        best = @min(best, t.read());
    }
    return best;
}

fn zipMask(comptime high: bool) [16]i32 {
    var m: [16]i32 = undefined;
    const base: i32 = if (high) 8 else 0;
    for (0..8) |j| {
        m[2 * j] = base + @as(i32, @intCast(j));
        m[2 * j + 1] = ~(base + @as(i32, @intCast(j)));
    }
    return m;
}

/// A 16x16 byte transpose: four perfect shuffles, each rotating the (row, lane) address
/// left by a bit.
inline fn transpose16(rows: *[16]U8x16) void {
    @setEvalBranchQuota(10_000);
    inline for (0..4) |_| {
        var next: [16]U8x16 = undefined;
        inline for (0..8) |i| {
            next[2 * i] = @shuffle(u8, rows[i], rows[i + 8], zipMask(false));
            next[2 * i + 1] = @shuffle(u8, rows[i], rows[i + 8], zipMask(true));
        }
        rows.* = next;
    }
}

/// 16 vectors in the stored, one-after-another layout against Q queries: transpose each
/// 16 bytes of the 16 vectors so a register holds one byte of each, then look up.
fn scanPacked(comptime Q: usize, codes: [*]const u8, stride: usize, luts: [*]const u8, out: []u32, out_stride: usize) void {
    @setEvalBranchQuota(20_000);
    var acc: [Q][2]U16x8 = @splat(@splat(@splat(0)));
    const mask: U8x16 = @splat(0x0F);
    const chunks = stride / 16;
    for (0..chunks) |c| {
        var rows: [16]U8x16 = undefined;
        inline for (0..16) |j| rows[j] = codes[j * stride + c * 16 ..][0..16].*;
        transpose16(&rows);
        inline for (0..16) |k| {
            const lo = rows[k] & mask;
            const hi = rows[k] >> @splat(4);
            const g = (c * 16 + k) * 2;
            inline for (0..Q) |q| {
                const tl: @Vector(16, i8) = @bitCast(@as(U8x16, luts[(q * groups + g) * 16 ..][0..16].*));
                const th: @Vector(16, i8) = @bitCast(@as(U8x16, luts[(q * groups + g + 1) * 16 ..][0..16].*));
                const r1: U8x16 = @bitCast(zq.simd_shuffle.table16(tl, lo));
                const r2: U8x16 = @bitCast(zq.simd_shuffle.table16(th, hi));
                acc[q][0] += widenLo(r1);
                acc[q][1] += widenHi(r1);
                acc[q][0] += widenLo(r2);
                acc[q][1] += widenHi(r2);
            }
        }
    }
    inline for (0..Q) |q| {
        inline for (0..2) |part| {
            const lanes: [8]u16 = acc[q][part];
            for (0..8) |l| out[(part * 8 + l) * out_stride + q] = lanes[l];
        }
    }
}

fn same(got: []const u32, want: []const u32) !void {
    for (got, want, 0..) |g, w, i| if (g != w) {
        std.debug.print("disagree at {d}: {d} vs {d}\n", .{ i, g, w });
        return error.Mismatch;
    };
}

fn timeColumns(comptime Q: usize, t16: []const u8, stride: usize, luts: []const u8, out: []u32) u64 {
    var best: u64 = std.math.maxInt(u64);
    for (0..trials) |_| {
        var t = Timer.start();
        var v: usize = 0;
        while (v + 16 <= n) : (v += 16) {
            var q: usize = 0;
            while (q + Q <= nq) : (q += Q) {
                scanColumns(Q, t16.ptr + v * stride, stride, luts.ptr + q * groups * 16, out[v * nq + q ..], nq);
            }
        }
        best = @min(best, t.read());
    }
    return best;
}

/// 16 vectors stored as byte columns against Q queries: each 16-byte load is one byte of
/// every vector, two nibbles, two lookups.
fn scanColumns(comptime Q: usize, cols: [*]const u8, stride: usize, luts: [*]const u8, out: []u32, out_stride: usize) void {
    var acc: [Q][4]U16x8 = @splat(@splat(@splat(0)));
    const mask: U8x16 = @splat(0x0F);
    for (0..stride) |k| {
        const c: U8x16 = cols[k * 16 ..][0..16].*;
        const lo = c & mask;
        const hi = c >> @splat(4);
        inline for (0..Q) |q| {
            const tl: @Vector(16, i8) = @bitCast(@as(U8x16, luts[(q * groups + 2 * k) * 16 ..][0..16].*));
            const th: @Vector(16, i8) = @bitCast(@as(U8x16, luts[(q * groups + 2 * k + 1) * 16 ..][0..16].*));
            const r1: U8x16 = @bitCast(zq.simd_shuffle.table16(tl, lo));
            const r2: U8x16 = @bitCast(zq.simd_shuffle.table16(th, hi));
            acc[q][0] += widenLo(r1);
            acc[q][1] += widenHi(r1);
            acc[q][2] += widenLo(r2);
            acc[q][3] += widenHi(r2);
        }
    }
    inline for (0..Q) |q| {
        const lo_half: [8]u16 = acc[q][0] + acc[q][2];
        const hi_half: [8]u16 = acc[q][1] + acc[q][3];
        for (0..8) |l| {
            out[l * out_stride + q] = lo_half[l];
            out[(8 + l) * out_stride + q] = hi_half[l];
        }
    }
}
