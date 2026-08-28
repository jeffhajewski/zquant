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
    const n: usize = 200_000;
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
    std.debug.print("{s:>5} {s:>8} {s:>8} {s:>9} {s:>9} {s:>8} {s:>7}\n",
        .{ "bits", "B/vector", "corpus", "1-thread", "batched", "10-thr", "speedup" });

    for ([_]u6{ 2, 3, 4, 5 }) |bits| {
        var index = try zq.flat.FlatIndex.init(a, .{ .dim = dim, .bits = bits, .seed = 0x5EED });
        defer index.deinit();

        var t = Timer.start();
        try index.addBatch(corpus);
        _ = t.read(); // build time no longer reported

        var searcher = try zq.flat.FlatIndex.Searcher.init(a, index, k);
        defer searcher.deinit();
        var batch_searcher = try zq.flat.FlatIndex.BatchSearcher.init(a, index, 32, k);
        defer batch_searcher.deinit();

        // warm
        _ = index.search(queries[0..dim], &searcher);

        var hits: usize = 0;
        t.reset();
        for (0..nq) |qi| {
            const res = index.search(queries[qi*dim..][0..dim], &searcher);
            for (res) |e| if (e.id == truth[qi]) { hits += 1; break; };
        }
        const search_ns = t.read();

        // Batched: the corpus is read once per batch instead of once per query.
        _ = index.searchBatch(queries[0 .. 32 * dim], &batch_searcher);
        t.reset();
        {
            var off: usize = 0;
            while (off < nq) : (off += 32) {
                const take = @min(32, nq - off);
                std.mem.doNotOptimizeAway(
                    index.searchBatch(queries[off * dim ..][0 .. take * dim], &batch_searcher),
                );
            }
        }
        const batch_ns = t.read();

        // Query-parallel: threads own disjoint queries.
        const threads = 10;
        var par = try zq.flat.FlatIndex.ParallelSearcher.init(a, index, threads, 32, k);
        defer par.deinit();
        _ = try index.searchBatchParallel(queries[0 .. 32 * dim], &par);
        t.reset();
        {
            var off: usize = 0;
            const chunk = threads * 32;
            while (off < nq) : (off += chunk) {
                const take = @min(chunk, nq - off);
                std.mem.doNotOptimizeAway(
                    try index.searchBatchParallel(queries[off * dim ..][0 .. take * dim], &par),
                );
            }
        }
        const par_ns = t.read();

        const qps = @as(f64, @floatFromInt(nq)) / (@as(f64,@floatFromInt(search_ns))/1e9);
        // Retained: the sweep now reports speedup against its own 1-thread run.
        std.mem.doNotOptimizeAway(qps);

        // Thread-count sweep, to separate "our scaling is poor" from "this machine has
        // 4 performance and 6 efficiency cores"., to separate "our scaling is poor" from "this machine has
        // 4 performance and 6 efficiency cores".
        if (bits == 5) {
            // Sweep over a replicated query set, not the 100 ground-truth queries.
            // With only 100, ten threads get batches of ten while eight get twelve,
            // so the per-vector cost amortizes over fewer queries as the thread count
            // rises — the sweep would measure that, not scaling. Every thread here
            // gets a full 32-query batch at every thread count.
            const sweep_n = 640;
            const sweep_queries = try a.alloc(f32, sweep_n * dim);
            defer a.free(sweep_queries);
            for (0..sweep_n) |i| {
                @memcpy(sweep_queries[i * dim ..][0..dim], queries[(i % nq) * dim ..][0..dim]);
            }

            std.debug.print("  thread sweep at bits=5 ({d} queries, 32 per thread per call):\n",
                .{sweep_n});
            var base_qps: f64 = 0;
            for ([_]usize{ 1, 2, 4, 6, 8, 10 }) |tc| {
                var sweep = try zq.flat.FlatIndex.ParallelSearcher.init(a, index, tc, 32, k);
                defer sweep.deinit();
                _ = try index.searchBatchParallel(sweep_queries[0 .. 32 * dim], &sweep);
                var st = Timer.start();
                var off: usize = 0;
                const chunk = tc * 32;
                while (off < sweep_n) : (off += chunk) {
                    const take = @min(chunk, sweep_n - off);
                    std.mem.doNotOptimizeAway(
                        try index.searchBatchParallel(sweep_queries[off * dim ..][0 .. take * dim], &sweep),
                    );
                }
                const ns = st.read();
                const tq = @as(f64, @floatFromInt(sweep_n)) / (@as(f64, @floatFromInt(ns)) / 1e9);
                if (tc == 1) base_qps = tq;
                std.debug.print("    {d:>2} threads: {d:>8.0} QPS  {d:>5.2}x  ({d:.0}% eff)\n",
                    .{ tc, tq, tq / base_qps, tq / base_qps / @as(f64, @floatFromInt(tc)) * 100 });
            }
        }

        const corpus_mb = @as(f64, @floatFromInt(n * index.bytesPerVector())) / 1e6;
        const par_qps = @as(f64, @floatFromInt(nq)) / (@as(f64, @floatFromInt(par_ns)) / 1e9);
        std.debug.print("{d:>5} {d:>8} {d:>6.0}MB {d:>9.1} {d:>9.1} {d:>8.0} {d:>6.1}x {d:>7.3}\n", .{
            bits,
            index.bytesPerVector(),
            corpus_mb,
            qps,
            @as(f64, @floatFromInt(nq)) / (@as(f64, @floatFromInt(batch_ns)) / 1e9),
            par_qps,
            par_qps / qps,
            @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(nq)),
        });
    }
}
