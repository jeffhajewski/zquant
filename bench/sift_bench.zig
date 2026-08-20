//! Recall against a real dataset with published ground truth.
//!
//! ANN_SIFT10K: 10,000 base vectors, 100 queries, 128 dimensions, top-100 exact L2
//! neighbours. Run `tools/fetch_datasets.sh` first.
//!
//! This is the first measurement here on real data. Everything before it used
//! synthetic corpora — uniform-on-sphere or synthetic clusters — whose difficulty was
//! chosen by me and therefore proves little about behaviour on embeddings.

const std = @import("std");
const zq = @import("zquant");

const Vectors = struct {
    data: []f32,
    dim: u32,
    count: usize,
    fn row(self: Vectors, i: usize) []const f32 {
        return self.data[i * self.dim ..][0..self.dim];
    }
};

/// .fvecs: each record is an i32 dimension followed by that many f32.
fn readFvecs(allocator: std.mem.Allocator, path: []const u8) !Vectors {
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 1 << 31);
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
fn readIvecs(allocator: std.mem.Allocator, path: []const u8) !struct { data: []u32, width: usize, count: usize } {
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 1 << 31);
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
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const a = gpa.allocator();

    const base = readFvecs(a, "data/siftsmall/siftsmall_base.fvecs") catch |err| {
        std.debug.print("could not read SIFT10K ({s}). Run tools/fetch_datasets.sh\n", .{@errorName(err)});
        return;
    };
    defer a.free(base.data);
    const queries = try readFvecs(a, "data/siftsmall/siftsmall_query.fvecs");
    defer a.free(queries.data);
    const truth = try readIvecs(a, "data/siftsmall/siftsmall_groundtruth.ivecs");
    defer a.free(truth.data);

    std.debug.print("\nSIFT10K: {d} base x {d}d, {d} queries, top-{d} exact L2 ground truth\n",
        .{ base.count, base.dim, queries.count, truth.width });
    std.debug.print("{s:>5} {s:>7} {s:>8} {s:>7} {s:>7} {s:>7} {s:>7} {s:>8} {s:>8}\n",
        .{ "bits", "codeB", "totalB", "ratio", "1@1", "1@10", "1@100", "R@10", "QPS" });

    const ks = [_]usize{ 1, 10, 100 };

    for ([_]u6{ 2, 3, 4, 5, 6 }) |bits| {
        var index = try zq.flat.FlatIndex.init(a, .{
            .dim = base.dim,
            .bits = bits,
            .metric = .l2,
            .seed = 0x5EED,
        });
        defer index.deinit();
        try index.addBatch(base.data);

        var searcher = try zq.flat.FlatIndex.Searcher.init(a, index, 100);
        defer searcher.deinit();

        var found = [_]usize{0} ** ks.len;
        var recall_at_10: f64 = 0;

        // Warm, then time.
        _ = index.search(queries.row(0), &searcher);
        var timer = try std.time.Timer.start();

        for (0..queries.count) |qi| {
            const results = index.search(queries.row(qi), &searcher);
            const gt = truth.data[qi * truth.width ..][0..truth.width];

            // 1@k: is the true nearest neighbour within the first k returned?
            for (ks, 0..) |k, ki| {
                for (results[0..@min(k, results.len)]) |e| {
                    if (e.id == gt[0]) {
                        found[ki] += 1;
                        break;
                    }
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
        const elapsed = timer.read();

        const nq: f64 = @floatFromInt(queries.count);
        const raw_bytes = base.dim * @sizeOf(f32);
        std.debug.print("{d:>5} {d:>7} {d:>8} {d:>6.1}x {d:>7.2} {d:>7.2} {d:>7.2} {d:>8.3} {d:>8.0}\n", .{
            bits,
            index.codeBytesPerVector(),
            index.bytesPerVector(),
            @as(f64, @floatFromInt(raw_bytes)) / @as(f64, @floatFromInt(index.bytesPerVector())),
            @as(f64, @floatFromInt(found[0])) / nq,
            @as(f64, @floatFromInt(found[1])) / nq,
            @as(f64, @floatFromInt(found[2])) / nq,
            recall_at_10 / nq,
            nq / (@as(f64, @floatFromInt(elapsed)) / 1e9),
        });
    }
}
