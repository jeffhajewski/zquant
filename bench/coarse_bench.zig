//! The nearest list for each row: exhaustively in f32, against `index/coarse.zig`'s
//! int8 pre-pass with exact recheck, which gives the same lists.
//!
//! Real rows, as training sees them: the first 256k of data/cohere-768-1m-norm, rotated
//! by an index's quantizer, against the 1000 centroids that index trains on them. One
//! thread, 64 rows a call, as k-means and adding run per thread.

const std = @import("std");
const zq = @import("zquant");
const Timer = @import("timer.zig").Timer;

const dim: u32 = 768;
const lists: u32 = 1000;
const n_train: usize = 256_000;
const n: usize = 32_768;
const trials = 3;

fn seconds(t: Timer) f64 {
    return @as(f64, @floatFromInt(t.read())) / std.time.ns_per_s;
}

pub fn main() !void {
    const a = std.heap.smp_allocator;
    // Only the first rows of a 3 GB file, so read them through C rather than whole.
    const path = "data/cohere-768-1m-norm/base.fvecs";
    const file = std.c.fopen(path, "rb") orelse {
        std.debug.print("no {s}: run bench/py/prepare.py cohere-768-1m-norm\n", .{path});
        return;
    };
    defer _ = std.c.fclose(file);
    const raw = try a.alloc(u32, n_train * (dim + 1));
    defer a.free(raw);
    if (std.c.fread(std.mem.sliceAsBytes(raw).ptr, 4, raw.len, file) != raw.len) return error.ShortRead;
    const base = try a.alloc(f32, n_train * dim);
    defer a.free(base);
    for (0..n_train) |r| {
        std.debug.assert(raw[r * (dim + 1)] == dim);
        @memcpy(base[r * dim ..][0..dim], @as([]const f32, @ptrCast(raw[r * (dim + 1) + 1 ..][0..dim])));
    }

    var index = try zq.ivf.IvfIndex.init(a, .{ .dim = dim, .bits = 2, .lists = lists });
    defer index.deinit();
    var t = Timer.start();
    try index.train(base, 8);
    std.debug.print("coarse_bench: {d} x {d}, {d} lists; training on 8 threads took {d:.1} s\n", .{ n_train, dim, lists, seconds(t) });

    const p = index.quantizer.padded();
    var ws = try zq.prod.Workspace.init(a, index.quantizer);
    defer ws.deinit();
    const rows = try a.alloc(f32, n * p);
    defer a.free(rows);
    for (0..n) |r| _ = index.quantizer.mse.encodeRotated(base[r * dim ..][0..dim], rows[r * p ..][0..p], ws.staging);

    const block = zq.coarse.max_rows;
    const want = try a.alloc(u32, n);
    defer a.free(want);
    const got = try a.alloc(u32, n);
    defer a.free(got);
    const dots = try a.alloc(f32, block * lists);
    defer a.free(dots);

    var near = try zq.coarse.Nearest.init(a, lists, p);
    defer near.deinit(a);
    near.load(index.centroids);
    var scratch = try zq.coarse.Nearest.Scratch.init(a, &near);
    defer scratch.deinit(a);

    var exhaustive: f64 = std.math.inf(f64);
    var bounded: f64 = std.math.inf(f64);
    var int8: f64 = std.math.inf(f64);
    for (0..trials) |_| {
        t = Timer.start();
        var r: usize = 0;
        while (r < n) : (r += block) {
            zq.coarse.centroidDots(rows[r * p ..][0 .. block * p], p, index.centroids, lists, dots);
            for (0..block) |u| want[r + u] = @intCast(zq.coarse.nearest(dots[u * lists ..][0..lists], index.centroid_sq));
        }
        exhaustive = @min(exhaustive, seconds(t));

        scratch.rechecked = 0;
        t = Timer.start();
        r = 0;
        while (r < n) : (r += block) near.assign(rows[r * p ..][0 .. block * p], index.centroids, index.centroid_sq, &scratch, got[r..][0..block]);
        bounded = @min(bounded, seconds(t));

        t = Timer.start();
        r = 0;
        while (r < n) : (r += block) near.int8Pass(rows[r * p ..][0 .. block * p], &scratch);
        int8 = @min(int8, seconds(t));
    }
    if (!std.mem.eql(u32, want, got)) @panic("the pre-pass chose differently");

    const rate = struct {
        fn f(secs: f64) f64 {
            return @as(f64, @floatFromInt(n)) / secs / 1000;
        }
    }.f;
    std.debug.print("  exhaustive f32                {d:8.1} k rows/s\n", .{rate(exhaustive)});
    std.debug.print("  int8 pre-pass + exact recheck {d:8.1} k rows/s  ({d:.2}x), same lists\n", .{ rate(bounded), exhaustive / bounded });
    std.debug.print("    of which the int8 pass      {d:8.1} k rows/s  ({d:.0}% of the time)\n", .{ rate(int8), 100 * int8 / bounded });
    std.debug.print("    lists rechecked per row     {d:8.1} of {d}\n", .{ @as(f64, @floatFromInt(scratch.rechecked)) / n, lists });
}
