//! Attribute the batch loop's cost by ablation.
//!
//! The index reaches ~134 G dim/s where the isolated kernel reaches ~181 at d=256.
//! Memory bandwidth and epilogue addressing were both tested and neither explains the
//! difference (docs/notes.md), so this measures each stage by subtraction instead of
//! reasoning about it: kernel alone, then the estimate, then the threshold compare,
//! then the real `searchBatch`. Each variant runs the same tiled kernel over the same
//! corpus, so the differences are the stages themselves.

const std = @import("std");
const zq = @import("zquant");
const Timer = @import("timer.zig").Timer;

const tile = 4;
const group = 4;
const max_batch = 32;

fn randomUnit(buf: []f32, r: std.Random) void {
    var n: f64 = 0;
    for (buf) |*v| {
        const g = r.floatNorm(f32);
        v.* = g;
        n += @as(f64, g) * g;
    }
    const inv: f32 = @floatCast(1.0 / @sqrt(n));
    for (buf) |*v| v.* *= inv;
}

pub fn main() !void {
    const a = std.heap.smp_allocator;
    const dim: u32 = 256;
    const n: usize = 100_000;
    const nq = 32;
    const k = 10;
    const trials = 5;

    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();

    const corpus = try a.alloc(f32, n * dim);
    defer a.free(corpus);
    for (0..n) |i| randomUnit(corpus[i * dim ..][0..dim], random);
    const queries = try a.alloc(f32, nq * dim);
    defer a.free(queries);
    for (0..nq) |i| randomUnit(queries[i * dim ..][0..dim], random);

    inline for ([_]zq.flat.Residency{ .compact, .expanded }) |residency| {
        var index = try zq.flat.FlatIndex.init(a, .{
            .dim = dim,
            .bits = 5,
            .metric = .inner_product,
            .seed = 0x5EED,
            .residency = residency,
        });
        defer index.deinit();
        try index.addBatch(corpus);

        var searcher = try zq.flat.FlatIndex.BatchSearcher.init(a, index, nq, k);
        defer searcher.deinit();

        // Prime the query state exactly as searchBatch does.
        for (0..nq) |i| {
            index.quantizer.prepareQuery(
                queries[i * dim ..][0..dim],
                &searcher.query_states[i],
                &searcher.workspace,
            );
            searcher.scan_queries[i].load(searcher.query_states[i].rotated);
        }

        const stride = index.codeStride();
        const padded = index.quantizer.padded();
        var mse_block: [tile * max_batch]f32 = undefined;
        // A plausible filled-heap threshold, so the compare mostly rejects as it does
        // in a real search. Comparing against -inf would take the branch every time
        // and measure the wrong path.
        const thresholds: [max_batch]f32 = @splat(0.25);
        var sink: f32 = 0;

        // stage 0 = kernel only, 1 = + estimate, 2 = + threshold compare.
        var stage_ns: [3]u64 = @splat(std.math.maxInt(u64));
        for (0..trials) |_| {
            inline for (0..3) |stage| {
                var t = Timer.start();
                var v0: usize = 0;
                while (v0 + tile <= n) : (v0 += tile) {
                    var i: usize = 0;
                    while (i + group <= nq) : (i += group) {
                        switch (residency) {
                            .expanded => zq.simd_scan.scoreExpandedTiled(
                                tile,
                                group,
                                @ptrCast(index.codes.items.ptr + v0 * stride),
                                stride,
                                searcher.scan_queries[i..],
                                index.table.scale,
                                padded,
                                mse_block[i..],
                                max_batch,
                            ),
                            .compact => zq.simd_scan.scoreInt8Tiled(
                                tile,
                                group,
                                index.layout,
                                searcher.table,
                                searcher.scan_queries[i..],
                                index.codes.items.ptr + v0 * stride,
                                stride,
                                padded,
                                mse_block[i..],
                                max_batch,
                            ),
                        }
                    }
                    if (stage == 0) {
                        sink += mse_block[0];
                        continue;
                    }
                    for (0..tile) |u| {
                        const s = index.scalars.items[v0 + u];
                        const ng = s.norm * s.gamma;
                        for (0..nq) |q| {
                            const est = ng * mse_block[u * max_batch + q];
                            if (stage == 1) {
                                sink += est;
                            } else if (est > thresholds[q]) {
                                sink += est;
                            }
                        }
                    }
                }
                const took = t.read();
                stage_ns[stage] = @min(stage_ns[stage], took);
            }
        }

        const work_pre: f64 = @floatFromInt(n * dim * nq);

        // Full searchBatch across retrieval depths. Depth is the one knob that moved
        // the index without moving the kernel - 18% on nytimes and 55% on SIFT - and
        // the sift arithmetic is far too small to explain it, so this measures rather
        // than assumes.
        var full_ns: u64 = std.math.maxInt(u64);
        for ([_]usize{ 10, 50, 100, 200 }) |kk| {
            var s2 = try zq.flat.FlatIndex.BatchSearcher.init(a, index, nq, kk);
            defer s2.deinit();
            var best: u64 = std.math.maxInt(u64);
            for (0..trials) |_| {
                var t = Timer.start();
                std.mem.doNotOptimizeAway(index.searchBatch(queries, &s2));
                best = @min(best, t.read());
            }
            if (kk == 10) full_ns = best;
            const g = work_pre / @as(f64, @floatFromInt(best));
            std.debug.print("{s:>10}k={d:<4} {d:>12.1} {d:>9.0}% {d:>11.1}%\n", .{
                "searchBatch ",                                              kk, g,
                g / (work_pre / @as(f64, @floatFromInt(stage_ns[0]))) * 100,
                (@as(f64, @floatFromInt(best)) - @as(f64, @floatFromInt(stage_ns[0]))) /
                    @as(f64, @floatFromInt(stage_ns[0])) * 100,
            });
        }

        std.mem.doNotOptimizeAway(sink);

        const work = work_pre;
        const names = [_][]const u8{ "kernel only", "+ estimate", "+ compare" };
        std.debug.print("\n{s} residency, d={d} n={d} batch={d}\n", .{ @tagName(residency), dim, n, nq });
        std.debug.print("{s:>16} {s:>12} {s:>10} {s:>12}\n", .{ "stage", "G dim/s", "vs kernel", "added cost" });
        for (names, stage_ns) |name, ns| {
            const g = work / @as(f64, @floatFromInt(ns));
            std.debug.print("{s:>16} {d:>12.1} {d:>9.0}% {d:>11.1}%\n", .{
                name,
                g,
                g / (work / @as(f64, @floatFromInt(stage_ns[0]))) * 100,
                (@as(f64, @floatFromInt(ns)) - @as(f64, @floatFromInt(stage_ns[0]))) /
                    @as(f64, @floatFromInt(stage_ns[0])) * 100,
            });
        }
        std.mem.doNotOptimizeAway(full_ns);
    }
}
