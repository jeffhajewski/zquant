//! Recall against a real dataset with published ground truth.
//!
//! ANN_SIFT10K: 10,000 base vectors, 100 queries, 128 dimensions, top-100 exact L2
//! neighbours. Run `tools/fetch_datasets.sh` first.
//!
//! This is the first measurement here on real data. Everything before it used
//! synthetic corpora — uniform-on-sphere or synthetic clusters — whose difficulty was
//! chosen by me and therefore proves little about behaviour on embeddings.
//!
//! **Reports the rank distribution of the true nearest neighbour, not just 1@k.**
//! An earlier version reported 1@10 and 1@100, both of which saturate at 1.00 here,
//! and a saturated metric says nothing about margin: at b=4 the worst query sits at
//! rank 9, so "1@10 = 1.00" was true by exactly one position. Percentiles show the
//! headroom that a ceiling hides. `bench/sift_verify.zig` validates this harness.

const std = @import("std");
const zq = @import("zquant");
const Timer = @import("timer.zig").Timer;

const Vectors = struct {
    data: []f32,
    dim: u32,
    count: usize,
    fn row(self: Vectors, i: usize) []const f32 {
        return self.data[i * self.dim ..][0..self.dim];
    }
};

/// .fvecs: each record is an i32 dimension followed by that many f32.
fn readFvecs(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Vectors {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 31));
    defer allocator.free(bytes);

    const dim = std.mem.readInt(u32, bytes[0..4], .little);
    const record = 4 + 4 * @as(usize, dim);
    const count = bytes.len / record;

    const data = try allocator.alloc(f32, count * dim);
    for (0..count) |i| {
        const base = i * record + 4;
        for (0..dim) |j| {
            const raw = std.mem.readInt(u32, bytes[base + 4 * j ..][0..4], .little);
            data[i * dim + j] = @bitCast(raw);
        }
    }
    return .{ .data = data, .dim = dim, .count = count };
}

/// .ivecs: same framing, i32 payload.
fn readIvecs(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !struct { data: []u32, width: usize, count: usize } {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 31));
    defer allocator.free(bytes);

    const width = std.mem.readInt(u32, bytes[0..4], .little);
    const record = 4 + 4 * @as(usize, width);
    const count = bytes.len / record;

    const data = try allocator.alloc(u32, count * width);
    for (0..count) |i| {
        const base = i * record + 4;
        for (0..width) |j| {
            data[i * width + j] = std.mem.readInt(u32, bytes[base + 4 * j ..][0..4], .little);
        }
    }
    return .{ .data = data, .width = width, .count = count };
}

pub fn main() !void {
    // smp_allocator: 0.16 dropped GeneralPurposeAllocator, and benchmarks want
    // throughput rather than leak tracking.
    const a = std.heap.smp_allocator;

    // 0.16 routes file I/O through the Io interface; benches need one only
    // to read the dataset.
    var threaded = std.Io.Threaded.init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const base = readFvecs(a, io, "data/siftsmall/siftsmall_base.fvecs") catch |err| {
        std.debug.print("could not read SIFT10K ({s}). Run tools/fetch_datasets.sh\n", .{@errorName(err)});
        return;
    };
    defer a.free(base.data);
    const queries = try readFvecs(a, io, "data/siftsmall/siftsmall_query.fvecs");
    defer a.free(queries.data);
    const truth = try readIvecs(a, io, "data/siftsmall/siftsmall_groundtruth.ivecs");
    defer a.free(truth.data);

    std.debug.print("\nSIFT10K: {d} base x {d}d, {d} queries, top-{d} exact L2 ground truth\n", .{ base.count, base.dim, queries.count, truth.width });
    // Rank of the true nearest neighbour in our results: median, p90, and worst.
    // The worst case is what a rerank candidate count has to be sized against.
    std.debug.print("{s:>5} {s:>7} {s:>7} {s:>6} {s:>6} {s:>7} {s:>7} {s:>7} {s:>8}\n", .{ "bits", "totalB", "ratio", "med", "p90", "worst", "1@10", "R@10", "QPS" });

    for ([_]u6{ 2, 3, 4, 5, 6 }) |bits| {
        var index = try zq.flat.FlatIndex.init(a, .{
            .dim = base.dim,
            .bits = bits,
            .metric = .l2,
            .seed = 0x5EED,
        });
        defer index.deinit();
        try index.addBatch(base.data);

        // Two searchers. Ranks need the whole corpus ordered, which makes the heap
        // enormous and is not how anyone queries; timing uses a realistic k. Sharing
        // one searcher between them reported QPS 7x low.
        var rank_searcher = try zq.flat.FlatIndex.Searcher.init(a, index, base.count);
        defer rank_searcher.deinit();
        var searcher = try zq.flat.FlatIndex.Searcher.init(a, index, 10);
        defer searcher.deinit();

        const ranks = try a.alloc(usize, queries.count);
        defer a.free(ranks);
        var recall_at_10: f64 = 0;

        for (0..queries.count) |qi| {
            const results = index.search(queries.row(qi), &rank_searcher);
            const gt = truth.data[qi * truth.width ..][0..truth.width];

            // Where did the true nearest neighbour actually land?
            ranks[qi] = results.len;
            for (results, 0..) |e, r| {
                if (e.id == gt[0]) {
                    ranks[qi] = r;
                    break;
                }
            }

            // R@10: overlap between our top-10 and the true top-10.
            var overlap: usize = 0;
            for (results[0..@min(10, results.len)]) |e| {
                for (gt[0..10]) |g| {
                    if (e.id == g) {
                        overlap += 1;
                        break;
                    }
                }
            }
            recall_at_10 += @as(f64, @floatFromInt(overlap)) / 10.0;
        }

        // Timed pass at a realistic k.
        _ = index.search(queries.row(0), &searcher);
        var timer = Timer.start();
        for (0..queries.count) |qi| std.mem.doNotOptimizeAway(index.search(queries.row(qi), &searcher));
        const elapsed = timer.read();

        const nq: f64 = @floatFromInt(queries.count);
        const raw_bytes = base.dim * @sizeOf(f32);

        var within_10: usize = 0;
        for (ranks) |r| {
            if (r < 10) within_10 += 1;
        }
        std.mem.sort(usize, ranks, {}, std.sort.asc(usize));

        std.debug.print("{d:>5} {d:>7} {d:>6.1}x {d:>6} {d:>6} {d:>7} {d:>7.2} {d:>7.3} {d:>8.0}\n", .{
            bits,
            index.bytesPerVector(),
            @as(f64, @floatFromInt(raw_bytes)) / @as(f64, @floatFromInt(index.bytesPerVector())),
            ranks[queries.count / 2],
            ranks[queries.count * 9 / 10],
            ranks[queries.count - 1],
            @as(f64, @floatFromInt(within_10)) / nq,
            recall_at_10 / nq,
            nq / (@as(f64, @floatFromInt(elapsed)) / 1e9),
        });
    }
}
