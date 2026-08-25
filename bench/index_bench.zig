const std = @import("std");
const zq = @import("zquant");
const Timer = @import("timer.zig").Timer;

fn randomUnit(buf: []f32, r: std.Random) void {
    var n: f64 = 0;
    for (buf) |*v| { const g = r.floatNorm(f32); v.* = g; n += @as(f64,g)*g; }
    const inv: f32 = @floatCast(1.0/@sqrt(n));
    for (buf) |*v| v.* *= inv;
}
fn dot(a: []const f32, b: []const f32) f64 {
    var s: f64 = 0; for (a,b) |x,y| s += @as(f64,x)*y; return s;
}

pub fn main() !void {
    // smp_allocator: 0.16 dropped GeneralPurposeAllocator, and benchmarks want
    // throughput rather than leak tracking.
    const a = std.heap.smp_allocator;

    const dim: u32 = 1024;
    const n: usize = 100_000;
    const nq = 100;
    const k = 10;

    const corpus = try a.alloc(f32, n * dim);
    defer a.free(corpus);
    var prng = std.Random.DefaultPrng.init(1);
    for (0..n) |i| randomUnit(corpus[i*dim..][0..dim], prng.random());

    const queries = try a.alloc(f32, nq * dim);
    defer a.free(queries);
    for (0..nq) |i| randomUnit(queries[i*dim..][0..dim], prng.random());

    // Ground truth.
    const truth = try a.alloc(u32, nq);
    defer a.free(truth);
    for (0..nq) |qi| {
        var best: u32 = 0; var bs: f64 = -1e300;
        for (0..n) |i| {
            const s = dot(queries[qi*dim..][0..dim], corpus[i*dim..][0..dim]);
            if (s > bs) { bs = s; best = @intCast(i); }
        }
        truth[qi] = best;
    }

    std.debug.print("\nFlatIndex: d={d} n={d} k={d}, {d} queries\n", .{ dim, n, k, nq });
    std.debug.print("{s:>5} {s:>8} {s:>7} {s:>10} {s:>11} {s:>9} {s:>7}\n",
        .{ "bits", "B/vector", "simd", "build/vec", "QPS", "us/query", "1@10" });

    for ([_]u6{ 2, 3, 4, 5 }) |bits| {
        var index = try zq.flat.FlatIndex.init(a, .{ .dim = dim, .bits = bits, .seed = 0x5EED });
        defer index.deinit();

        var t = Timer.start();
        try index.addBatch(corpus);
        const build_ns = t.read();

        var searcher = try zq.flat.FlatIndex.Searcher.init(a, index, k);
        defer searcher.deinit();

        // warm
        _ = index.search(queries[0..dim], &searcher);

        var hits: usize = 0;
        t.reset();
        for (0..nq) |qi| {
            const res = index.search(queries[qi*dim..][0..dim], &searcher);
            for (res) |e| if (e.id == truth[qi]) { hits += 1; break; };
        }
        const search_ns = t.read();

        const qps = @as(f64, @floatFromInt(nq)) / (@as(f64,@floatFromInt(search_ns))/1e9);
        std.debug.print("{d:>5} {d:>8} {s:>7} {d:>8.1}us {d:>11.1} {d:>9.0} {d:>7.3}\n", .{
            bits, index.bytesPerVector(), if (index.vectorized()) "yes" else "no",
            @as(f64,@floatFromInt(build_ns))/@as(f64,@floatFromInt(n))/1000.0,
            qps, @as(f64,@floatFromInt(search_ns))/@as(f64,@floatFromInt(nq))/1000.0,
            @as(f64,@floatFromInt(hits))/@as(f64,@floatFromInt(nq)),
        });
    }
}
