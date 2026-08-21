//! Diagnostics for the SIFT10K recall numbers. Verifies the harness itself before
//! any result from it is trusted.
const std = @import("std");
const zq = @import("zquant");

fn readFvecs(a: std.mem.Allocator, path: []const u8) !struct { data: []f32, dim: u32, count: usize } {
    const bytes = try std.fs.cwd().readFileAlloc(a, path, 1 << 31);
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
fn readIvecs(a: std.mem.Allocator, path: []const u8) !struct { data: []u32, width: usize, count: usize } {
    const bytes = try std.fs.cwd().readFileAlloc(a, path, 1 << 31);
    defer a.free(bytes);
    const width = std.mem.readInt(u32, bytes[0..4], .little);
    const record = 4 + 4 * @as(usize, width);
    const count = bytes.len / record;
    const data = try a.alloc(u32, count * width);
    for (0..count) |i| for (0..width) |j| {
        data[i * width + j] = std.mem.readInt(u32, bytes[i * record + 4 + 4 * j ..][0..4], .little);
    };
    return .{ .data = data, .width = width, .count = count };
}

fn l2sq(a: []const f32, b: []const f32) f64 {
    var s: f64 = 0;
    for (a, b) |x, y| { const d = @as(f64, x) - y; s += d * d; }
    return s;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const a = gpa.allocator();

    const base = try readFvecs(a, "data/siftsmall/siftsmall_base.fvecs");
    defer a.free(base.data);
    const q = try readFvecs(a, "data/siftsmall/siftsmall_query.fvecs");
    defer a.free(q.data);
    const gt = try readIvecs(a, "data/siftsmall/siftsmall_groundtruth.ivecs");
    defer a.free(gt.data);
    const d = base.dim;

    // ---- CHECK 1: does the published ground truth match brute force? ----
    var gt_ok: usize = 0;
    var gt_bad: usize = 0;
    for (0..q.count) |qi| {
        var best: u32 = 0; var bd: f64 = 1e300;
        for (0..base.count) |i| {
            const dist = l2sq(q.data[qi*d..][0..d], base.data[i*d..][0..d]);
            if (dist < bd) { bd = dist; best = @intCast(i); }
        }
        if (best == gt.data[qi*gt.width]) gt_ok += 1 else {
            gt_bad += 1;
            if (gt_bad <= 3) std.debug.print("  q{d}: brute={d} gt={d}\n", .{qi, best, gt.data[qi*gt.width]});
        }
    }
    std.debug.print("CHECK 1  ground truth vs brute force: {d}/{d} agree\n", .{gt_ok, q.count});

    // ---- CHECK 2: how separated is the true NN from the 100th? ----
    var ratio_sum: f64 = 0;
    var min_ratio: f64 = 1e300;
    const dists = try a.alloc(f64, base.count);
    defer a.free(dists);
    for (0..q.count) |qi| {
        for (0..base.count) |i| dists[i] = l2sq(q.data[qi*d..][0..d], base.data[i*d..][0..d]);
        std.mem.sort(f64, dists, {}, std.sort.asc(f64));
        const r = @sqrt(dists[0]) / @sqrt(dists[99]);
        ratio_sum += r;
        min_ratio = @min(min_ratio, r);
    }
    std.debug.print("CHECK 2  dist(NN)/dist(100th): mean={d:.3} worst={d:.3}\n",
        .{ ratio_sum/@as(f64,@floatFromInt(q.count)), min_ratio });

    // ---- CHECK 3: actual rank of the true NN in our results ----
    for ([_]u6{ 2, 4 }) |bits| {
        var index = try zq.flat.FlatIndex.init(a, .{ .dim = d, .bits = bits, .metric = .l2, .seed = 0x5EED });
        defer index.deinit();
        try index.addBatch(base.data);

        // Ask for the whole corpus so we can see the true rank, not a truncated one.
        var searcher = try zq.flat.FlatIndex.Searcher.init(a, index, base.count);
        defer searcher.deinit();

        var ranks = try a.alloc(usize, q.count);
        defer a.free(ranks);
        var worst: usize = 0;
        var sum: usize = 0;
        var not_found: usize = 0;
        for (0..q.count) |qi| {
            const res = index.search(q.data[qi*d..][0..d], &searcher);
            var rank: usize = res.len; // sentinel: absent
            for (res, 0..) |e, r| if (e.id == gt.data[qi*gt.width]) { rank = r; break; };
            if (rank == res.len) not_found += 1;
            ranks[qi] = rank;
            sum += rank;
            worst = @max(worst, rank);
        }
        std.mem.sort(usize, ranks, {}, std.sort.asc(usize));
        std.debug.print("CHECK 3  b={d}: true-NN rank  mean={d:.1} median={d} p90={d} worst={d} absent={d} (n={d})\n",
            .{ bits, @as(f64,@floatFromInt(sum))/@as(f64,@floatFromInt(q.count)),
               ranks[q.count/2], ranks[q.count*9/10], worst, not_found, base.count });
    }

    // ---- CHECK 4: a deliberately broken index must score badly ----
    // If random codes still yield 1@100 = 1.00, the metric is meaningless.
    {
        var index = try zq.flat.FlatIndex.init(a, .{ .dim = d, .bits = 4, .metric = .l2, .seed = 1 });
        defer index.deinit();
        var prng = std.Random.DefaultPrng.init(99);
        const shuffled = try a.alloc(f32, base.count * d);
        defer a.free(shuffled);
        @memcpy(shuffled, base.data);
        // Shuffle the corpus rows so ids no longer correspond to ground-truth ids.
        for (0..base.count) |i| {
            const j = prng.random().uintLessThan(usize, base.count);
            for (0..d) |c| std.mem.swap(f32, &shuffled[i*d+c], &shuffled[j*d+c]);
        }
        try index.addBatch(shuffled);
        var searcher = try zq.flat.FlatIndex.Searcher.init(a, index, 100);
        defer searcher.deinit();
        var hits: usize = 0;
        for (0..q.count) |qi| {
            for (index.search(q.data[qi*d..][0..d], &searcher)) |e| {
                if (e.id == gt.data[qi*gt.width]) { hits += 1; break; }
            }
        }
        std.debug.print("CHECK 4  shuffled-corpus control: 1@100 = {d:.2} (must be near 0.01)\n",
            .{ @as(f64,@floatFromInt(hits))/@as(f64,@floatFromInt(q.count)) });
    }
}
