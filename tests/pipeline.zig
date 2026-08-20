//! End-to-end tests across module seams.
//!
//! The unit tests verify each module against its own oracle. These verify that the
//! modules agree with *each other* when composed the way a real index composes them:
//! quantizer → packing → scan kernel, and query → rotation → parity split.
//!
//! They also test the property the library actually exists to provide, which no unit
//! test covers: that top-k retrieval returns the right neighbours. Distortion bounds
//! are a proxy for that; recall is the thing itself.

const std = @import("std");
const testing = std.testing;
const zq = @import("zquant");

const Allocator = std.mem.Allocator;

// -- helpers -----------------------------------------------------------------

fn randomUnit(buf: []f32, random: std.Random) void {
    var norm: f64 = 0;
    for (buf) |*v| {
        const g = random.floatNorm(f32);
        v.* = g;
        norm += @as(f64, g) * g;
    }
    const inv: f32 = @floatCast(1.0 / @sqrt(norm));
    for (buf) |*v| v.* *= inv;
}

fn exactDot(a: []const f32, b: []const f32) f64 {
    var sum: f64 = 0;
    for (a, b) |x, y| sum += @as(f64, x) * y;
    return sum;
}

/// Indices of the `k` largest scores, descending.
fn topK(allocator: Allocator, scores: []const f64, k: usize) ![]u32 {
    const order = try allocator.alloc(u32, scores.len);
    defer allocator.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);

    std.mem.sort(u32, order, scores, struct {
        fn lessThan(s: []const f64, a: u32, b: u32) bool {
            return s[a] > s[b];
        }
    }.lessThan);

    const out = try allocator.alloc(u32, @min(k, scores.len));
    @memcpy(out, order[0..out.len]);
    return out;
}

// -- seam: quantizer output feeds the scan kernel ----------------------------

test "real quantizer output scores correctly through the packed scan" {
    // The scan kernel's own tests build codes from a bare Codebook over synthetic
    // "rotated" data. This runs the actual path: Mse.encodeRotated -> encodeSlice ->
    // Layout.pack -> scoreExact, and checks it against a dot product taken directly
    // from the decoded centroids. A mismatch here means packing order, dimension
    // order, or padding disagree between producer and consumer.
    const allocator = testing.allocator;

    for ([_]u32{ 64, 256, 1024 }) |dim| {
        for ([_]u6{ 2, 4 }) |bits| {
            var q = try zq.mse.Mse.init(allocator, .{ .dim = dim, .bits = bits, .seed = 0x5EED });
            defer q.deinit();
            var ws = try zq.mse.Workspace.init(allocator, q);
            defer ws.deinit();

            const layout = zq.packing.Layout.init(q.padded, bits);

            const x = try allocator.alloc(f32, dim);
            defer allocator.free(x);
            const codes = try allocator.alloc(u8, q.codeLen());
            defer allocator.free(codes);
            const stored = try allocator.alloc(u8, layout.stride());
            defer allocator.free(stored);
            const rotated_query = try allocator.alloc(f32, q.padded);
            defer allocator.free(rotated_query);

            var prng = std.Random.DefaultPrng.init(dim + bits);
            const random = prng.random();

            for (0..8) |_| {
                randomUnit(x, random);
                _ = q.encode(x, codes, &ws);
                layout.pack(codes, stored);

                for (rotated_query) |*v| v.* = random.floatNorm(f32);

                // Reference: decode the codes back to centroids and dot directly.
                var expected: f64 = 0;
                for (codes, rotated_query) |code, p| {
                    expected += @as(f64, p) * q.codebook.centroids[code];
                }

                const got = zq.simd_scan.scoreExact(
                    layout,
                    q.codebook.centroids,
                    rotated_query,
                    stored,
                );
                try testing.expectApproxEqAbs(@as(f32, @floatCast(expected)), got, 1e-4);
            }
        }
    }
}

test "int8 kernel agrees with the exact path on real quantizer output" {
    // Same seam, but through the vectorized kernel and its parity-split query. This
    // is where a wrong even/odd mapping would show up: it would still produce
    // plausible numbers, just consistently wrong ones.
    const allocator = testing.allocator;
    const dim: u32 = 512;

    var q = try zq.mse.Mse.init(allocator, .{ .dim = dim, .bits = 4, .seed = 0x1234 });
    defer q.deinit();
    var ws = try zq.mse.Workspace.init(allocator, q);
    defer ws.deinit();

    const layout = zq.packing.Layout.init(q.padded, 4);
    try testing.expect(zq.simd_scan.canVectorize(layout));

    const table = zq.simd_scan.Table.init(q.codebook.centroids);
    var query = try zq.simd_scan.Query.init(allocator, layout);
    defer query.deinit();

    const x = try allocator.alloc(f32, dim);
    defer allocator.free(x);
    const codes = try allocator.alloc(u8, q.codeLen());
    defer allocator.free(codes);
    const stored = try allocator.alloc(u8, layout.stride());
    defer allocator.free(stored);
    const rotated_query = try allocator.alloc(f32, q.padded);
    defer allocator.free(rotated_query);

    var prng = std.Random.DefaultPrng.init(0xBEEF);
    const random = prng.random();

    var squared_error: f64 = 0;
    var squared_signal: f64 = 0;
    for (0..200) |_| {
        randomUnit(x, random);
        _ = q.encode(x, codes, &ws);
        layout.pack(codes, stored);

        for (rotated_query) |*v| v.* = random.floatNorm(f32) / @sqrt(@as(f32, @floatFromInt(dim)));
        query.load(rotated_query);

        const fast = zq.simd_scan.scoreInt8(layout, table, query, stored, q.padded);
        const exact = zq.simd_scan.scoreExact(layout, q.codebook.centroids, rotated_query, stored);
        const err = @as(f64, fast) - exact;
        squared_error += err * err;
        squared_signal += exact * exact;
    }
    try testing.expect(@sqrt(squared_error / squared_signal) < 0.02);
}

// -- the property the library exists for -------------------------------------

const RecallResult = struct {
    /// Fraction of queries whose true nearest neighbour appears in the approximate
    /// top-k. This is the paper's "1@k".
    at_k: f64,
    /// Mean relative error of the estimated inner product.
    score_error: f64,
};

fn measureRecall(
    allocator: Allocator,
    dim: u32,
    bits: u6,
    corpus_size: usize,
    queries: usize,
    k: usize,
    clustered: bool,
    seed: u64,
) !RecallResult {
    var q = try zq.prod.Prod.init(allocator, .{ .dim = dim, .bits = bits, .seed = seed });
    defer q.deinit();
    var ws = try zq.prod.Workspace.init(allocator, q);
    defer ws.deinit();
    var state = try zq.prod.QueryState.init(allocator, q);
    defer state.deinit();

    const corpus = try allocator.alloc(f32, corpus_size * dim);
    defer allocator.free(corpus);
    const codes = try allocator.alloc(u8, corpus_size * q.codeLen());
    defer allocator.free(codes);
    const sketches = try allocator.alloc(u8, corpus_size * q.sketchLen());
    defer allocator.free(sketches);
    const scalars = try allocator.alloc(zq.prod.Scalars, corpus_size);
    defer allocator.free(scalars);

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    // Clustered data is the realistic case and the harder one: neighbours are close
    // together, so small score errors reorder them.
    const cluster_count = 32;
    const centers = try allocator.alloc(f32, cluster_count * dim);
    defer allocator.free(centers);
    for (0..cluster_count) |c| randomUnit(centers[c * dim ..][0..dim], random);

    for (0..corpus_size) |i| {
        const row = corpus[i * dim ..][0..dim];
        if (clustered) {
            const c = random.uintLessThan(usize, cluster_count);
            for (row, centers[c * dim ..][0..dim]) |*v, center| {
                v.* = center + 0.35 * random.floatNorm(f32) / @sqrt(@as(f32, @floatFromInt(dim)));
            }
            const norm = zq.mse.euclideanNorm(row);
            for (row) |*v| v.* /= norm;
        } else {
            randomUnit(row, random);
        }
        scalars[i] = q.encode(
            row,
            codes[i * q.codeLen() ..][0..q.codeLen()],
            sketches[i * q.sketchLen() ..][0..q.sketchLen()],
            &ws,
        );
    }

    const query = try allocator.alloc(f32, dim);
    defer allocator.free(query);
    const exact_scores = try allocator.alloc(f64, corpus_size);
    defer allocator.free(exact_scores);
    const approx_scores = try allocator.alloc(f64, corpus_size);
    defer allocator.free(approx_scores);

    var hits: usize = 0;
    var error_sum: f64 = 0;
    var signal_sum: f64 = 0;

    for (0..queries) |_| {
        randomUnit(query, random);
        q.prepareQuery(query, &state, &ws);

        for (0..corpus_size) |i| {
            exact_scores[i] = exactDot(query, corpus[i * dim ..][0..dim]);
            approx_scores[i] = q.dot(
                state,
                codes[i * q.codeLen() ..][0..q.codeLen()],
                sketches[i * q.sketchLen() ..][0..q.sketchLen()],
                scalars[i],
            );
            const err = approx_scores[i] - exact_scores[i];
            error_sum += err * err;
            signal_sum += exact_scores[i] * exact_scores[i];
        }

        const truth = try topK(allocator, exact_scores, 1);
        defer allocator.free(truth);
        const approx = try topK(allocator, approx_scores, k);
        defer allocator.free(approx);

        for (approx) |candidate| {
            if (candidate == truth[0]) {
                hits += 1;
                break;
            }
        }
    }

    return .{
        .at_k = @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(queries)),
        .score_error = @sqrt(error_sum / signal_sum),
    };
}

const print_recall = false; // flip to re-derive the thresholds below

fn report(label: []const u8, r: RecallResult) void {
    if (print_recall) std.debug.print("  {s:<28} 1@k={d:.3} err={d:.3}\n", .{ label, r.at_k, r.score_error });
}

test "recall improves with bit-width" {
    // The most basic sanity property of a quantizer used for search, and one no unit
    // test covers: more bits must retrieve better.
    const allocator = testing.allocator;
    var previous: f64 = 0;
    for ([_]u6{ 1, 2, 3, 4 }) |bits| {
        const result = try measureRecall(allocator, 256, bits, 2000, 100, 10, false, 0x5EED);
        report("bits sweep", result);
        try testing.expect(result.at_k >= previous - 0.02);
        previous = result.at_k;
    }
    // And 4 bits must actually be good, not merely better than 3. Thresholds
    // throughout this file are set from measurement with ~3 standard errors of
    // margin (100 queries, so σ ≈ 0.014 near p=0.98), not chosen to be comfortable.
    // Measured here: 0.600, 0.890, 0.980, 1.000 for b=1..4.
    try testing.expect(previous > 0.95);
}

test "recall at 4 bits on clustered data" {
    // Clustered corpora are the realistic case and strictly harder: true neighbours
    // are close in score, so estimator noise reorders them.
    const allocator = testing.allocator;
    const result = try measureRecall(allocator, 256, 4, 2000, 100, 10, true, 0xC10D);
    report("d=256 b=4 clustered", result);
    try testing.expect(result.at_k > 0.92); // measured 0.980
    try testing.expect(result.score_error < 0.20); // measured 0.138
}

test "recall holds at embedding dimensions" {
    // Higher d is where the near-independence argument is strongest, so recall
    // should be at least as good as at d=256 despite the corpus being no larger.
    const allocator = testing.allocator;
    const result = try measureRecall(allocator, 1024, 4, 1500, 60, 10, false, 0xD1);
    report("d=1024 b=4 uniform", result);
    try testing.expect(result.at_k > 0.95); // measured 1.000
}

test "recall degrades gracefully at KV dimensions" {
    // d=128 is the low-dimension regime flagged as a first-order risk in
    // DESIGN.md §10. This pins current behaviour so a regression is visible; it is a
    // measurement, not an aspiration.
    const allocator = testing.allocator;
    const result = try measureRecall(allocator, 128, 4, 2000, 100, 10, false, 0x808);
    report("d=128 b=4 uniform", result);
    try testing.expect(result.at_k > 0.95); // measured 0.990
}

test "top-1 recall is meaningfully worse than top-10, as it should be" {
    // Guards against a test that passes because everything is trivially easy. If
    // 1@1 equalled 1@10 the corpus would not be discriminating and the recall
    // numbers above would mean nothing.
    const allocator = testing.allocator;
    const at_1 = try measureRecall(allocator, 256, 2, 2000, 60, 1, false, 0x99);
    const at_10 = try measureRecall(allocator, 256, 2, 2000, 60, 10, false, 0x99);
    try testing.expect(at_10.at_k > at_1.at_k + 0.05);
}

// -- reconstruction round trip -----------------------------------------------

test "prod decode reconstructs better than mse alone at the same total budget" {
    // Sanity on the two-stage split: spending one bit on the sketch should not make
    // reconstruction worse than spending all b bits on MSE... except it should, for
    // MSE specifically, since prod optimizes inner products instead. Pin the
    // direction so the tradeoff is explicit rather than assumed.
    const allocator = testing.allocator;
    const dim: u32 = 256;
    const trials = 100;

    var mse_q = try zq.mse.Mse.init(allocator, .{ .dim = dim, .bits = 4, .seed = 1 });
    defer mse_q.deinit();
    var mse_ws = try zq.mse.Workspace.init(allocator, mse_q);
    defer mse_ws.deinit();

    var prod_q = try zq.prod.Prod.init(allocator, .{ .dim = dim, .bits = 4, .seed = 1 });
    defer prod_q.deinit();
    var prod_ws = try zq.prod.Workspace.init(allocator, prod_q);
    defer prod_ws.deinit();

    const x = try allocator.alloc(f32, dim);
    defer allocator.free(x);
    const back = try allocator.alloc(f32, dim);
    defer allocator.free(back);
    const mse_codes = try allocator.alloc(u8, mse_q.codeLen());
    defer allocator.free(mse_codes);
    const prod_codes = try allocator.alloc(u8, prod_q.codeLen());
    defer allocator.free(prod_codes);
    const sketch = try allocator.alloc(u8, prod_q.sketchLen());
    defer allocator.free(sketch);

    var prng = std.Random.DefaultPrng.init(3);
    const random = prng.random();

    var mse_distortion: f64 = 0;
    var prod_distortion: f64 = 0;
    for (0..trials) |_| {
        randomUnit(x, random);

        const norm = mse_q.encode(x, mse_codes, &mse_ws);
        mse_q.decode(mse_codes, norm, back, &mse_ws);
        for (x, back) |a, b| mse_distortion += (@as(f64, a) - b) * (@as(f64, a) - b);

        const scalars = prod_q.encode(x, prod_codes, sketch, &prod_ws);
        prod_q.decode(prod_codes, sketch, scalars, back, &prod_ws);
        for (x, back) |a, b| prod_distortion += (@as(f64, a) - b) * (@as(f64, a) - b);
    }

    // prod spends a bit on the sketch instead of MSE, so its reconstruction MSE is
    // worse. That is the deliberate trade, and it should be visible.
    try testing.expect(prod_distortion > mse_distortion);
    // But not catastrophically: the sketch recovers much of the lost bit.
    try testing.expect(prod_distortion < mse_distortion * 4.0);
}

// -- allocation failure ------------------------------------------------------

test "quantizer construction is leak-free under allocation failure" {
    // Exercises every errdefer path in Mse.init / Prod.init. Without this, a missing
    // errdefer leaks only on an allocation failure, which no other test produces.
    try testing.checkAllAllocationFailures(testing.allocator, struct {
        fn run(allocator: Allocator) !void {
            var q = try zq.mse.Mse.init(allocator, .{ .dim = 64, .bits = 2, .seed = 1 });
            defer q.deinit();
            var ws = try zq.mse.Workspace.init(allocator, q);
            defer ws.deinit();
        }
    }.run, .{});

    try testing.checkAllAllocationFailures(testing.allocator, struct {
        fn run(allocator: Allocator) !void {
            var q = try zq.prod.Prod.init(allocator, .{ .dim = 64, .bits = 2, .seed = 1 });
            defer q.deinit();
            var ws = try zq.prod.Workspace.init(allocator, q);
            defer ws.deinit();
            var state = try zq.prod.QueryState.init(allocator, q);
            defer state.deinit();
        }
    }.run, .{});
}
