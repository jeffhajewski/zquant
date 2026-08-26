//! zquant's arm of the cross-system comparison.
//!
//! Reads data/sift-norm — the same normalized corpus, queries, and inner-product
//! ground truth that bench/py/baselines.py feeds to PQ, RaBitQ, and turbovec — and
//! writes data/zquant.csv in the same schema. Sharing the input files rather than
//! three loaders agreeing is what makes the numbers comparable.

const std = @import("std");
const zq = @import("zquant");
const Timer = @import("timer.zig").Timer;

const K = 10;
/// Retrieve deeper than K so the true-NN rank distribution is not censored at K.
const RETRIEVE = 100;


fn readFvecs(a: std.mem.Allocator, io: std.Io, path: []const u8) !struct { data: []f32, dim: u32, count: usize } {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 31));
    defer a.free(bytes);
    const dim = std.mem.readInt(u32, bytes[0..4], .little);
    const record = 4 + 4 * @as(usize, dim);
    const count = bytes.len / record;
    const data = try a.alloc(f32, count * dim);
    for (0..count) |i| for (0..dim) |j| {
        const raw = std.mem.readInt(u32, bytes[i * record + 4 + 4 * j ..][0..4], .little);
        data[i * dim + j] = @bitCast(raw);
    };
    return .{ .data = data, .dim = dim, .count = count };
}

fn readIvecs(a: std.mem.Allocator, io: std.Io, path: []const u8) !struct { data: []u32, width: usize } {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 31));
    defer a.free(bytes);
    const width = std.mem.readInt(u32, bytes[0..4], .little);
    const record = 4 + 4 * @as(usize, width);
    const count = bytes.len / record;
    const data = try a.alloc(u32, count * width);
    for (0..count) |i| for (0..width) |j| {
        data[i * width + j] = std.mem.readInt(u32, bytes[i * record + 4 + 4 * j ..][0..4], .little);
    };
    return .{ .data = data, .width = width };
}

pub fn main() !void {
    const a = std.heap.smp_allocator;
    var threaded = std.Io.Threaded.init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // Which corpus to run is written by bench/py/prepare.py, so the Zig and Python
    // arms cannot drift onto different data.
    const name_raw = std.Io.Dir.cwd().readFileAlloc(io, "data/dataset.txt", a, .limited(256)) catch |e| {
        std.debug.print("no data/dataset.txt ({s}) — run bench/py/prepare.py first\n", .{@errorName(e)});
        return;
    };
    defer a.free(name_raw);
    const name = std.mem.trim(u8, name_raw, " \n\r\t");

    const dir = try std.fmt.allocPrint(a, "data/{s}", .{name});
    defer a.free(dir);
    const base_path = try std.fmt.allocPrint(a, "{s}/base.fvecs", .{dir});
    defer a.free(base_path);
    const query_path = try std.fmt.allocPrint(a, "{s}/query.fvecs", .{dir});
    defer a.free(query_path);
    const truth_path = try std.fmt.allocPrint(a, "{s}/groundtruth.ivecs", .{dir});
    defer a.free(truth_path);

    const base = readFvecs(a, io, base_path) catch |e| {
        std.debug.print("missing {s} ({s}) — run bench/py/prepare.py\n", .{ dir, @errorName(e) });
        return;
    };
    defer a.free(base.data);
    const queries = try readFvecs(a, io, query_path);
    defer a.free(queries.data);
    const truth = try readIvecs(a, io, truth_path);
    defer a.free(truth.data);

    const d = base.dim;
    const nq = queries.count;

    var out = std.ArrayList(u8).empty;
    defer out.deinit(a);
    try out.appendSlice(a, "system,config,bytes_per_vector,recall_at_10,rank_median,rank_p90,rank_worst,qps,batched_qps\n");

    std.debug.print("zquant on {s}: {d}x{d}, {d} queries, k={d}\n", .{ dir, base.count, d, nq, K });

    // Control: an exact unquantized scan through this same harness must score 1.000.
    // Anything less means the harness disagrees with the ground truth, and every
    // quantized number below would be measuring that instead of quantization.
    {
        var recall: f64 = 0;
        const scores = try a.alloc(f64, base.count);
        defer a.free(scores);
        const order = try a.alloc(u32, base.count);
        defer a.free(order);
        for (0..nq) |qi| {
            const query = queries.data[qi * d ..][0..d];
            for (0..base.count) |i| {
                var acc: f64 = 0;
                for (query, base.data[i * d ..][0..d]) |qv, xv| acc += @as(f64, qv) * xv;
                scores[i] = acc;
            }
            for (order, 0..) |*o, i| o.* = @intCast(i);
            std.mem.sort(u32, order, scores, struct {
                fn desc(sc: []const f64, x: u32, y: u32) bool {
                    return sc[x] > sc[y];
                }
            }.desc);
            const gt = truth.data[qi * truth.width ..][0..truth.width];
            var overlap: usize = 0;
            for (order[0..K]) |c| {
                for (gt[0..K]) |g| {
                    if (c == g) {
                        overlap += 1;
                        break;
                    }
                }
            }
            recall += @as(f64, @floatFromInt(overlap)) / @as(f64, K);
        }
        std.debug.print("  CONTROL exact f32 scan: R@10 = {d:.4} (must be 1.0000)\n",
            .{recall / @as(f64, @floatFromInt(nq))});
    }

    for ([_]u6{ 2, 3, 4, 5 }) |bits| {
        for ([_]bool{ false, true }) |calibrated| {
            const residency: zq.flat.Residency = .compact;
            const exact = false;
            const sketch = true;
            const correction: zq.flat.Correction = .scalar;
            var index = try zq.flat.FlatIndex.init(a, .{
                .dim = d,
                .bits = bits,
                .metric = .inner_product,
                .seed = 0x5EED,
                .exact_scan = exact,
                .use_sketch = sketch,
                .correction = correction,
                .residency = residency,
            });
            defer index.deinit();
            if (calibrated) {
                // Uniform draw from the corpus, as the fit requires.
                const rows = 1024;
                const stride = base.count / rows;
                const sample = try a.alloc(f32, rows * d);
                defer a.free(sample);
                for (0..rows) |i| {
                    @memcpy(sample[i * d ..][0..d], base.data[i * stride * d ..][0..d]);
                }
                try index.calibrate(sample);
            }
            try index.addBatch(base.data);

            var searcher = try zq.flat.FlatIndex.Searcher.init(a, index, RETRIEVE);
            defer searcher.deinit();

            const ranks = try a.alloc(usize, nq);
            defer a.free(ranks);
            var recall: f64 = 0;

            // Batched timing, matching how the Python baselines are measured: they
            // hand every query to one call, so timing ours one at a time was
            // comparing different things.
            var batch_searcher = try zq.flat.FlatIndex.BatchSearcher.init(a, index, 32, K);
            defer batch_searcher.deinit();
            _ = index.searchBatch(queries.data[0 .. 32 * d], &batch_searcher);
            var batch_timer = Timer.start();
            {
                var off: usize = 0;
                while (off < nq) : (off += 32) {
                    const take = @min(32, nq - off);
                    std.mem.doNotOptimizeAway(
                        index.searchBatch(queries.data[off * d ..][0 .. take * d], &batch_searcher),
                    );
                }
            }
            const batch_ns = batch_timer.read();

            // Query-parallel across 10 threads.
            const threads = 10;
            var par = try zq.flat.FlatIndex.ParallelSearcher.init(a, index, threads, 32, K);
            defer par.deinit();
            _ = try index.searchBatchParallel(queries.data[0 .. 32 * d], &par);
            var par_timer = Timer.start();
            {
                var off: usize = 0;
                const chunk = threads * 32;
                while (off < nq) : (off += chunk) {
                    const take = @min(chunk, nq - off);
                    std.mem.doNotOptimizeAway(
                        try index.searchBatchParallel(queries.data[off * d ..][0 .. take * d], &par),
                    );
                }
            }
            const par_ns = par_timer.read();

            _ = index.search(queries.data[0..d], &searcher);
            var timer = Timer.start();
            for (0..nq) |qi| {
                const res = index.search(queries.data[qi * d ..][0..d], &searcher);
                const gt = truth.data[qi * truth.width ..][0..truth.width];

                ranks[qi] = res.len;
                for (res, 0..) |e, r| {
                    if (e.id == gt[0]) {
                        ranks[qi] = r;
                        break;
                    }
                }
                var overlap: usize = 0;
                for (res[0..@min(K, res.len)]) |e| {
                    for (gt[0..K]) |g| {
                        if (e.id == g) {
                            overlap += 1;
                            break;
                        }
                    }
                }
                recall += @as(f64, @floatFromInt(overlap)) / @as(f64, K);
            }
            const elapsed = timer.read();

            std.mem.sort(usize, ranks, {}, std.sort.asc(usize));
            const fnq: f64 = @floatFromInt(nq);
            const qps = fnq / (@as(f64, @floatFromInt(elapsed)) / 1e9);
            const batch_qps = fnq / (@as(f64, @floatFromInt(batch_ns)) / 1e9);
            const par_qps = fnq / (@as(f64, @floatFromInt(par_ns)) / 1e9);

            try out.print(a, "zquant,bits={d}{s},{d},{d:.4},{d},{d},{d},{d:.1},{d:.1}\n", .{
                bits,
                if (calibrated) " +calibrate" else "",
                index.bytesPerVector(),
                recall / fnq,
                ranks[nq / 2],
                ranks[nq * 9 / 10],
                ranks[nq - 1],
                qps,
                batch_qps,
            });
            std.debug.print("  bits={d}{s:<12} {d:>3}B  R@10={d:.3}  med={d} p90={d} worst={d}  {d:.0} QPS  {d:.0} par\n", .{
                bits,
                if (calibrated) " +calibrate" else "",
                index.bytesPerVector(),
                recall / fnq,
                ranks[nq / 2],
                ranks[nq * 9 / 10],
                ranks[nq - 1],
                qps,
                par_qps,
            });
        }
    }

    var file = try std.Io.Dir.cwd().createFile(io, "data/zquant.csv", .{});
    defer file.close(io);
    var buf: [256]u8 = undefined;
    var writer = file.writer(io, &buf);
    try writer.interface.writeAll(out.items);
    try writer.interface.flush();
    std.debug.print("wrote data/zquant.csv\n", .{});
}
