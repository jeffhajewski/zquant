//! Exact rescoring of search candidates against the original vectors.
//!
//! The compressed scan ranks well but not perfectly: near-ties between neighbours are
//! decided by quantization error, so a true top-10 vector sometimes lands at rank 12
//! or 30. The usual fix is to over-fetch — ask the index for 4k–10k candidates — and rescore those few
//! exactly against full-precision vectors kept elsewhere (on disk, in a database, in a
//! memory map). The index stays small and the final order is exact.
//!
//! Exactness here means f32 arithmetic on the originals, under the same
//! larger-is-better convention as search: cosine similarity, raw inner product, or
//! negated squared distance.

const std = @import("std");
const flat = @import("flat.zig");
const Entry = flat.Entry;
const Metric = flat.Metric;

/// The exact score of `x` against `query` under `metric`.
pub fn exactScore(metric: Metric, query: []const f32, x: []const f32) f32 {
    std.debug.assert(query.len == x.len);
    const score = switch (metric) {
        .inner_product => dot(query, x),
        .cosine => blk: {
            const qq = dot(query, query);
            const xx = dot(x, x);
            // A zero vector has no direction; the index scores it the same way.
            if (qq == 0 or xx == 0) break :blk -std.math.inf(f32);
            break :blk dot(query, x) / (@sqrt(qq) * @sqrt(xx));
        },
        .l2 => -squaredDistance(query, x),
    };
    // A NaN would break the sort's ordering; rank it last instead.
    return if (std.math.isNan(score)) -std.math.inf(f32) else score;
}

/// Rescore `candidates` exactly and return the best `k` of them, best first, as a
/// prefix of `candidates` (which is reordered in place). Ties go to the smaller id, so
/// the order is deterministic.
///
/// `originals` holds the full-precision vectors row-major, `dim` floats each, indexed
/// by id: row `id` must be the vector that was added as `id`.
pub fn rerank(
    metric: Metric,
    query: []const f32,
    originals: []const f32,
    dim: usize,
    candidates: []Entry,
    k: usize,
) []Entry {
    std.debug.assert(query.len == dim);
    for (candidates) |*c| {
        const row = originals[@as(usize, c.id) * dim ..][0..dim];
        c.score = exactScore(metric, query, row);
    }
    std.sort.pdq(Entry, candidates, {}, better);
    return candidates[0..@min(k, candidates.len)];
}

fn better(_: void, a: Entry, b: Entry) bool {
    if (a.score != b.score) return a.score > b.score;
    return a.id < b.id;
}

const lanes = 16;
const V = @Vector(lanes, f32);

fn dot(a: []const f32, b: []const f32) f32 {
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + lanes <= a.len) : (i += lanes) {
        const va: V = a[i..][0..lanes].*;
        const vb: V = b[i..][0..lanes].*;
        acc += va * vb;
    }
    var sum = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

fn squaredDistance(a: []const f32, b: []const f32) f32 {
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + lanes <= a.len) : (i += lanes) {
        const va: V = a[i..][0..lanes].*;
        const vb: V = b[i..][0..lanes].*;
        const d = va - vb;
        acc += d * d;
    }
    var sum = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) {
        const d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;

test "exact scores match a float64 reference for every metric" {
    var prng = std.Random.DefaultPrng.init(0xE7AC);
    const random = prng.random();
    for ([_]usize{ 1, 15, 16, 17, 100, 768 }) |dim| {
        const q = try testing.allocator.alloc(f32, dim);
        defer testing.allocator.free(q);
        const x = try testing.allocator.alloc(f32, dim);
        defer testing.allocator.free(x);
        for (q, x) |*a, *b| {
            a.* = random.floatNorm(f32);
            b.* = random.floatNorm(f32);
        }
        var qx: f64 = 0;
        var qq: f64 = 0;
        var xx: f64 = 0;
        var dd: f64 = 0;
        for (q, x) |a, b| {
            qx += @as(f64, a) * b;
            qq += @as(f64, a) * a;
            xx += @as(f64, b) * b;
            dd += (@as(f64, a) - b) * (@as(f64, a) - b);
        }
        const tol = 1e-4 * @as(f64, @floatFromInt(dim));
        try testing.expectApproxEqAbs(qx, exactScore(.inner_product, q, x), tol);
        try testing.expectApproxEqAbs(qx / @sqrt(qq * xx), exactScore(.cosine, q, x), 1e-5);
        try testing.expectApproxEqAbs(-dd, exactScore(.l2, q, x), tol);
    }
}

test "rerank orders candidates exactly, best first, ties by id" {
    const dim = 4;
    const originals = [_]f32{
        1, 0, 0, 0, // id 0
        0, 1, 0, 0, // id 1
        0.9, 0.1, 0, 0, // id 2
        1, 0, 0, 0, // id 3, ties with 0
        0, 0, 0, 0, // id 4, zero vector
    };
    const query = [_]f32{ 1, 0, 0, 0 };
    var candidates = [_]Entry{
        .{ .id = 4, .score = 9 },
        .{ .id = 1, .score = 8 },
        .{ .id = 3, .score = 7 },
        .{ .id = 2, .score = 6 },
        .{ .id = 0, .score = 5 },
    };
    const best = rerank(.cosine, &query, &originals, dim, &candidates, 3);
    try testing.expectEqual(@as(usize, 3), best.len);
    try testing.expectEqual(@as(u32, 0), best[0].id);
    try testing.expectEqual(@as(u32, 3), best[1].id);
    try testing.expectEqual(@as(u32, 2), best[2].id);
    try testing.expectEqual(@as(f32, 1), best[0].score);

    // k larger than the candidate list returns them all, the zero vector last.
    const all = rerank(.cosine, &query, &originals, dim, &candidates, 10);
    try testing.expectEqual(@as(usize, 5), all.len);
    try testing.expectEqual(@as(u32, 4), all[4].id);
    try testing.expectEqual(-std.math.inf(f32), all[4].score);
}

test "reranking over-fetched candidates recovers the exact top-k" {
    // The use case end to end: a coarse index, 10x over-fetch, exact rescoring. The
    // result should be the brute-force top-10 for nearly every query.
    var prng = std.Random.DefaultPrng.init(0x4E4A);
    const random = prng.random();
    const dim = 64;
    const n = 2000;
    const k = 10;
    const data = try testing.allocator.alloc(f32, n * dim);
    defer testing.allocator.free(data);
    for (data) |*v| v.* = random.floatNorm(f32);

    var index = try flat.FlatIndex.init(testing.allocator, .{ .dim = dim, .bits = 3, .metric = .l2, .seed = 5 });
    defer index.deinit();
    try index.addBatch(data);
    var searcher = try flat.FlatIndex.Searcher.init(testing.allocator, index, 10 * k);
    defer searcher.deinit();

    var exact_hits: usize = 0;
    var coarse_hits: usize = 0;
    const queries = 20;
    for (0..queries) |qi| {
        const q = data[qi * 7 * dim ..][0..dim];
        // Brute-force truth.
        var truth: [n]Entry = undefined;
        for (&truth, 0..) |*t, i| t.* = .{ .id = @intCast(i), .score = exactScore(.l2, q, data[i * dim ..][0..dim]) };
        std.sort.pdq(Entry, &truth, {}, better);

        const found = index.search(q, &searcher);
        for (found[0..k]) |c| {
            for (truth[0..k]) |t| if (t.id == c.id) {
                coarse_hits += 1;
            };
        }
        const best = rerank(.l2, q, data, dim, found, k);
        for (best) |c| {
            for (truth[0..k]) |t| if (t.id == c.id) {
                exact_hits += 1;
            };
        }
    }
    const coarse = @as(f64, @floatFromInt(coarse_hits)) / (queries * k);
    const exact = @as(f64, @floatFromInt(exact_hits)) / (queries * k);
    try testing.expect(exact >= 0.98);
    try testing.expect(exact > coarse);
}
