//! Self-contained benchmark: `zig build quickbench`.
//!
//! No downloads, no Python, no FAISS. Generates its own corpora, computes exact ground
//! truth, and reports recall against storage and throughput. Runs in well under a minute
//! so that cloning the repository and forming an opinion is one command.
//!
//! Two corpora, because one number would be misleading. Both are otherwise identical
//! Gaussian directions; they differ only in **centroid norm** — how far the corpus of
//! unit directions sits from the origin — which is what determines whether the
//! per-coordinate calibration is worth anything (docs/notes.md). `centred` stands in for
//! corpora like nytimes-256 (centroid 0.12) where calibration is neutral; `offset` stands
//! in for corpora like SIFT (centroid 0.65) where it is worth ten points.
//!
//! The competitive comparison against FAISS and turbovec needs real corpora and their
//! dependencies; see docs/comparison.md. This answers a narrower question: does the thing
//! work, and what does it cost.

const std = @import("std");
const zq = @import("zquant");
const Timer = @import("timer.zig").Timer;

const dim: u32 = 256;
const n: usize = 50_000;
const nq: usize = 200;
const k: usize = 10;

fn buildCorpus(buf: []f32, offset: f32, random: std.Random) void {
    var i: usize = 0;
    while (i < buf.len) : (i += dim) {
        const row = buf[i..][0..dim];
        var sq: f64 = 0;
        for (row) |*v| {
            v.* = random.floatNorm(f32) + offset;
            sq += @as(f64, v.*) * v.*;
        }
        const inv: f32 = @floatCast(1.0 / @sqrt(sq));
        for (row) |*v| v.* *= inv;
    }
}

/// Norm of the mean of the unit directions: the statistic that predicts whether
/// calibration pays.
fn centroidNorm(buf: []const f32) f32 {
    var mean = [_]f64{0} ** dim;
    var rows: usize = 0;
    var i: usize = 0;
    while (i < buf.len) : (i += dim) {
        for (buf[i..][0..dim], 0..) |v, j| mean[j] += v;
        rows += 1;
    }
    var sq: f64 = 0;
    for (mean) |m| {
        const a = m / @as(f64, @floatFromInt(rows));
        sq += a * a;
    }
    return @floatCast(@sqrt(sq));
}

fn exactTopK(corpus: []const f32, query: []const f32, out: []u32, scratch: []f32) void {
    for (0..corpus.len / dim) |i| {
        var acc: f32 = 0;
        for (query, corpus[i * dim ..][0..dim]) |q, x| acc += q * x;
        scratch[i] = acc;
    }
    // Partial selection by insertion: only the top `out.len` are needed, and k is small.
    std.debug.assert(out.len <= 64);
    var best: [64]u32 = undefined;
    var best_n: usize = 0;
    for (0..corpus.len / dim) |i| {
        const s = scratch[i];
        if (best_n < out.len) {
            best[best_n] = @intCast(i);
            best_n += 1;
            var j = best_n - 1;
            while (j > 0 and scratch[best[j - 1]] < scratch[best[j]]) : (j -= 1) {
                std.mem.swap(u32, &best[j - 1], &best[j]);
            }
        } else if (s > scratch[best[best_n - 1]]) {
            best[best_n - 1] = @intCast(i);
            var j = best_n - 1;
            while (j > 0 and scratch[best[j - 1]] < scratch[best[j]]) : (j -= 1) {
                std.mem.swap(u32, &best[j - 1], &best[j]);
            }
        }
    }
    @memcpy(out, best[0..out.len]);
}

pub fn main() !void {
    const a = std.heap.smp_allocator;
    var prng = std.Random.DefaultPrng.init(0x2C0FFEE);
    const random = prng.random();

    std.debug.print(
        \\zquant quickbench — {s}, {d} threads available
        \\  {d} vectors x {d} dims, {d} queries, k={d}, exact ground truth
        \\
        \\
    , .{ @tagName(@import("builtin").cpu.arch), std.Thread.getCpuCount() catch 0, n, dim, nq, k });

    const corpus = try a.alloc(f32, n * dim);
    defer a.free(corpus);
    const queries = try a.alloc(f32, nq * dim);
    defer a.free(queries);
    const truth = try a.alloc(u32, nq * k);
    defer a.free(truth);
    const scratch = try a.alloc(f32, n);
    defer a.free(scratch);

    for ([_]struct { name: []const u8, offset: f32 }{
        .{ .name = "centred", .offset = 0.0 },
        .{ .name = "offset", .offset = 0.85 },
    }) |corpus_kind| {
        buildCorpus(corpus, corpus_kind.offset, random);
        buildCorpus(queries, corpus_kind.offset, random);
        for (0..nq) |i| {
            exactTopK(corpus, queries[i * dim ..][0..dim], truth[i * k ..][0..k], scratch);
        }

        std.debug.print("  corpus '{s}': centroid norm {d:.3}  ({s})\n", .{
            corpus_kind.name,
            centroidNorm(corpus),
            if (corpus_kind.offset == 0) "calibration should be neutral" else "calibration should pay",
        });
        std.debug.print("  {s:>5} {s:>10} {s:>8} {s:>8} {s:>11} {s:>11}\n",
            .{ "bits", "calibrate", "B/vec", "R@10", "QPS 1-thr", "QPS all" });

        for ([_]u6{ 2, 3, 4, 5 }) |bits| {
            for ([_]bool{ false, true }) |calibrated| {
                var index = try zq.flat.FlatIndex.init(a, .{
                    .dim = dim,
                    .bits = bits,
                    .metric = .inner_product,
                    .seed = 0x5EED,
                });
                defer index.deinit();
                if (calibrated) try index.calibrate(corpus[0 .. 4096 * dim]);
                try index.addBatch(corpus);

                var batch = try zq.flat.FlatIndex.BatchSearcher.init(a, index, 32, k);
                defer batch.deinit();
                const threads: usize = @min(@as(usize, 10), std.Thread.getCpuCount() catch 1);
                var par = try zq.flat.FlatIndex.ParallelSearcher.init(a, index, threads, 32, k);
                defer par.deinit();

                var hits: usize = 0;
                var t = Timer.start();
                var off: usize = 0;
                while (off < nq) : (off += 32) {
                    const take = @min(32, nq - off);
                    const res = index.searchBatch(queries[off * dim ..][0 .. take * dim], &batch);
                    for (0..take) |qi| {
                        for (res[qi * k ..][0..k]) |e| {
                            for (truth[(off + qi) * k ..][0..k]) |g| {
                                if (e.id == g) {
                                    hits += 1;
                                    break;
                                }
                            }
                        }
                    }
                }
                const one_ns = t.read();

                _ = try index.searchBatchParallel(queries[0 .. 32 * dim], &par);
                t.reset();
                off = 0;
                const chunk = threads * 32;
                while (off < nq) : (off += chunk) {
                    const take = @min(chunk, nq - off);
                    std.mem.doNotOptimizeAway(
                        try index.searchBatchParallel(queries[off * dim ..][0 .. take * dim], &par),
                    );
                }
                const par_ns = t.read();

                const fnq: f64 = @floatFromInt(nq);
                std.debug.print("  {d:>5} {s:>10} {d:>7}B {d:>8.3} {d:>11.0} {d:>11.0}\n", .{
                    bits,
                    if (calibrated) "yes" else "no",
                    index.bytesPerVector(),
                    @as(f64, @floatFromInt(hits)) / (fnq * @as(f64, @floatFromInt(k))),
                    fnq / (@as(f64, @floatFromInt(one_ns)) / 1e9),
                    fnq / (@as(f64, @floatFromInt(par_ns)) / 1e9),
                });
            }
        }
        std.debug.print("\n", .{});
    }
}
