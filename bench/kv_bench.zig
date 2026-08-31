//! KV-cache compression fidelity, on real attention tensors.
//!
//! A KV cache is not a retrieval problem and recall@10 says nothing about it. At each
//! generation step attention needs *every* score, because they go through a softmax, and
//! then a weighted sum of value rows. So there are two distinct questions:
//!
//!   keys   — how accurately is ⟨q, k⟩ estimated? An inner-product problem, which is what
//!            the index's estimator is built for.
//!   values — how accurately is a row *reconstructed*? A decode problem, which the
//!            estimator has nothing to do with.
//!
//! They are measured separately and together, because a combined number alone cannot say
//! which half to improve.
//!
//! Tensors come from `bench/py/dump_kv.py` running a real model, since a KV cache has
//! structure that is hard to guess: RoPE rotates keys position-dependently, a few key
//! channels carry outsized magnitude, and values look nothing like keys.
//!
//! Baselines are per-token int8 and int4 with a per-row scale, which is what KV
//! quantization in practice usually means.

const std = @import("std");
const zq = @import("zquant");

const K_POSITIONS = 64; // query positions sampled per head
const MIN_CONTEXT = 64; // skip the earliest positions, where attention is near-trivial

fn readFvecs(a: std.mem.Allocator, io: std.Io, path: []const u8) !struct { data: []f32, dim: u32, count: usize } {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30));
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

fn softmax(scores: []f32) void {
    var m: f32 = -std.math.inf(f32);
    for (scores) |s| m = @max(m, s);
    var sum: f32 = 0;
    for (scores) |*s| {
        s.* = @exp(s.* - m);
        sum += s.*;
    }
    for (scores) |*s| s.* /= sum;
}

/// Per-row symmetric quantization to `bits`, the usual KV-cache baseline.
fn quantizeRows(rows: []const f32, d: usize, bits: u8, out: []f32) void {
    const levels: f32 = @floatFromInt((@as(u32, 1) << @intCast(bits - 1)) - 1);
    var i: usize = 0;
    while (i < rows.len) : (i += d) {
        var amax: f32 = 0;
        for (rows[i..][0..d]) |v| amax = @max(amax, @abs(v));
        const scale = if (amax > 0) amax / levels else 1.0;
        for (rows[i..][0..d], out[i..][0..d]) |v, *o| {
            o.* = @round(v / scale) * scale;
        }
    }
}

const Stats = struct {
    score_rmse: f64 = 0,
    weight_tv: f64 = 0,
    out_rel: f64 = 0,
    n: usize = 0,

    fn add(self: *Stats, other: Stats) void {
        self.score_rmse += other.score_rmse;
        self.weight_tv += other.weight_tv;
        self.out_rel += other.out_rel;
        self.n += other.n;
    }
    fn mean(self: Stats) Stats {
        const f: f64 = @floatFromInt(@max(self.n, 1));
        return .{ .score_rmse = self.score_rmse / f, .weight_tv = self.weight_tv / f, .out_rel = self.out_rel / f, .n = self.n };
    }
};

/// One head: exact attention against attention over the supplied key scores and value
/// rows. `key_scores` of null means exact keys; `values` of null means exact values.
fn evaluate(
    a: std.mem.Allocator,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    seq: usize,
    d: usize,
    approx_scores: ?[]const f32, // [seq*seq] or null
    approx_values: ?[]const f32,
    random: std.Random,
) !Stats {
    const inv_sqrt_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
    const exact_p = try a.alloc(f32, seq);
    defer a.free(exact_p);
    const approx_p = try a.alloc(f32, seq);
    defer a.free(approx_p);
    const out_a = try a.alloc(f32, d);
    defer a.free(out_a);
    const out_b = try a.alloc(f32, d);
    defer a.free(out_b);

    var stats = Stats{};
    for (0..K_POSITIONS) |_| {
        const t = MIN_CONTEXT + random.uintLessThan(usize, seq - MIN_CONTEXT);
        const n = t + 1; // causal
        const qv = q[t * d ..][0..d];

        var score_sq: f64 = 0;
        for (0..n) |j| {
            var s: f32 = 0;
            for (qv, k[j * d ..][0..d]) |x, y| s += x * y;
            exact_p[j] = s * inv_sqrt_d;
            const approx = if (approx_scores) |as| as[t * seq + j] * inv_sqrt_d else exact_p[j];
            approx_p[j] = approx;
            const diff = @as(f64, exact_p[j] - approx);
            score_sq += diff * diff;
        }
        stats.score_rmse += @sqrt(score_sq / @as(f64, @floatFromInt(n)));

        softmax(exact_p[0..n]);
        softmax(approx_p[0..n]);
        var tv: f64 = 0;
        for (0..n) |j| tv += @abs(@as(f64, exact_p[j] - approx_p[j]));
        stats.weight_tv += 0.5 * tv;

        @memset(out_a, 0);
        @memset(out_b, 0);
        for (0..n) |j| {
            const src_a = v[j * d ..][0..d];
            const src_b = if (approx_values) |av| av[j * d ..][0..d] else src_a;
            for (out_a, src_a) |*o, x| o.* += exact_p[j] * x;
            for (out_b, src_b) |*o, x| o.* += approx_p[j] * x;
        }
        var num: f64 = 0;
        var den: f64 = 0;
        for (out_a, out_b) |x, y| {
            const e = @as(f64, x - y);
            num += e * e;
            den += @as(f64, x) * x;
        }
        stats.out_rel += @sqrt(num / @max(den, 1e-30));
        stats.n += 1;
    }
    return stats;
}

pub fn main() !void {
    const a = std.heap.smp_allocator;
    var threaded = std.Io.Threaded.init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const meta_raw = std.Io.Dir.cwd().readFileAlloc(io, "data/kv/meta.txt", a, .limited(4096)) catch {
        std.debug.print("no data/kv — run: python bench/py/dump_kv.py\n", .{});
        return;
    };
    defer a.free(meta_raw);

    var prng = std.Random.DefaultPrng.init(0x11FE);
    const random = prng.random();

    std.debug.print("\nKV-cache fidelity on real attention tensors ({d} query positions per head)\n", .{K_POSITIONS});
    std.debug.print("  score RMSE is on pre-softmax logits; weight TV is total variation between\n", .{});
    std.debug.print("  attention distributions; output error is relative L2 of the attention output.\n\n", .{});

    var lines = std.mem.tokenizeScalar(u8, meta_raw, '\n');
    while (lines.next()) |line| {
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const layer = try std.fmt.parseInt(usize, f.next().?, 10);
        const kv_heads = try std.fmt.parseInt(usize, f.next().?, 10);
        const q_heads = try std.fmt.parseInt(usize, f.next().?, 10);
        const seq = try std.fmt.parseInt(usize, f.next().?, 10);
        const d = try std.fmt.parseInt(usize, f.next().?, 10);

        const kp = try std.fmt.allocPrint(a, "data/kv/layer{d}_k.fvecs", .{layer});
        defer a.free(kp);
        const vp = try std.fmt.allocPrint(a, "data/kv/layer{d}_v.fvecs", .{layer});
        defer a.free(vp);
        const qp = try std.fmt.allocPrint(a, "data/kv/layer{d}_q.fvecs", .{layer});
        defer a.free(qp);
        const kf = try readFvecs(a, io, kp);
        defer a.free(kf.data);
        const vf = try readFvecs(a, io, vp);
        defer a.free(vf.data);
        const qf = try readFvecs(a, io, qp);
        defer a.free(qf.data);

        std.debug.print("layer {d}: seq={d} d_head={d} kv_heads={d}\n", .{ layer, seq, d, kv_heads });
        std.debug.print("  {s:>18} {s:>9} {s:>11} {s:>10} {s:>11}\n", .{ "scheme", "B/tok/hd", "score RMSE", "weight TV", "output err" });

        // fp16 reference line: exact, and the memory everything else is measured against.
        const fp16_bytes = 2 * 2 * d;
        std.debug.print("  {s:>18} {d:>9} {s:>11} {s:>10} {s:>11}\n", .{ "fp16 (reference)", fp16_bytes, "0", "0", "0" });

        const scores = try a.alloc(f32, seq * seq);
        defer a.free(scores);
        const approx_v = try a.alloc(f32, seq * d);
        defer a.free(approx_v);

        // --- per-row int8 and int4, the usual KV baselines.
        for ([_]u8{ 8, 4 }) |bits| {
            var agg = Stats{};
            const kq = try a.alloc(f32, seq * d);
            defer a.free(kq);
            for (0..kv_heads) |h| {
                const k = kf.data[h * seq * d ..][0 .. seq * d];
                const v = vf.data[h * seq * d ..][0 .. seq * d];
                const q = qf.data[(h * (q_heads / kv_heads)) * seq * d ..][0 .. seq * d];
                quantizeRows(k, d, bits, kq);
                quantizeRows(v, d, bits, approx_v);
                for (0..seq) |i| for (0..seq) |j| {
                    var s: f32 = 0;
                    for (q[i * d ..][0..d], kq[j * d ..][0..d]) |x, y| s += x * y;
                    scores[i * seq + j] = s;
                };
                agg.add(try evaluate(a, q, k, v, seq, d, scores, approx_v, random));
            }
            const m = agg.mean();
            const per = 2 * (d * bits / 8 + 4);
            const label = if (bits == 8) "int8 per-row" else "int4 per-row";
            std.debug.print("  {s:>18} {d:>9} {d:>11.4} {d:>10.4} {d:>11.4}\n", .{ label, per, m.score_rmse, m.weight_tv, m.out_rel });
        }

        // --- zquant, across the knobs that could plausibly matter here.
        //
        // Two are worth separating. Calibration is opt-in and was measured to *cost*
        // recall on low-rank zero-mean data, and a KV cache is strongly low-rank. And
        // the index's estimator carries a per-vector correction fitted for retrieval,
        // where int4 simply reconstructs and takes an exact dot — so scoring both ways
        // says whether any gap is in the codes or in the estimator.
        const Variant = struct { calibrate: bool, decode_dot: bool, label: []const u8 };
        for ([_]u6{ 2, 3, 4, 5 }) |bits| {
            // decode+dot across the range, since it is the configuration that works;
            // the estimator and calibration are shown at one width to record how much
            // they cost here rather than to sweep a losing option.
            const variants: []const Variant = if (bits == 5) &.{
                .{ .calibrate = false, .decode_dot = true, .label = "decode+dot" },
                .{ .calibrate = false, .decode_dot = false, .label = "estimator" },
                .{ .calibrate = true, .decode_dot = false, .label = "est +calib" },
            } else &.{
                .{ .calibrate = false, .decode_dot = true, .label = "decode+dot" },
            };
            for (variants) |variant| {
                var agg_k = Stats{};
                var agg_both = Stats{};

                for (0..kv_heads) |h| {
                    const k = kf.data[h * seq * d ..][0 .. seq * d];
                    const v = vf.data[h * seq * d ..][0 .. seq * d];
                    const q = qf.data[(h * (q_heads / kv_heads)) * seq * d ..][0 .. seq * d];

                    var mse = try zq.mse.Mse.init(a, .{ .dim = @intCast(d), .bits = bits, .seed = 0x5EED });
                    defer mse.deinit();
                    var ws = try zq.mse.Workspace.init(a, mse);
                    defer ws.deinit();
                    const codes = try a.alloc(u8, mse.codeLen());
                    defer a.free(codes);

                    if (variant.decode_dot) {
                        // Reconstruct the keys and take an exact dot, which is the
                        // protocol the int4 baseline uses.
                        const kq = try a.alloc(f32, seq * d);
                        defer a.free(kq);
                        for (0..seq) |i| {
                            const norm = mse.encode(k[i * d ..][0..d], codes, &ws);
                            mse.decode(codes, norm, kq[i * d ..][0..d], &ws);
                        }
                        for (0..seq) |i| for (0..seq) |j| {
                            var s: f32 = 0;
                            for (q[i * d ..][0..d], kq[j * d ..][0..d]) |x, y| s += x * y;
                            scores[i * seq + j] = s;
                        };
                    } else {
                        var index = try zq.flat.FlatIndex.init(a, .{
                            .dim = @intCast(d),
                            .bits = bits,
                            .metric = .inner_product,
                            .seed = 0x5EED,
                        });
                        defer index.deinit();
                        if (variant.calibrate) try index.calibrate(k);
                        try index.addBatch(k);
                        var searcher = try zq.flat.FlatIndex.Searcher.init(a, index, seq);
                        defer searcher.deinit();
                        for (0..seq) |i| {
                            const res = index.search(q[i * d ..][0..d], &searcher);
                            for (scores[i * seq ..][0..seq]) |*s| s.* = 0;
                            for (res) |e| scores[i * seq + e.id] = e.score;
                        }
                    }

                    for (0..seq) |i| {
                        const norm = mse.encode(v[i * d ..][0..d], codes, &ws);
                        mse.decode(codes, norm, approx_v[i * d ..][0..d], &ws);
                    }

                    agg_k.add(try evaluate(a, q, k, v, seq, d, scores, null, random));
                    agg_both.add(try evaluate(a, q, k, v, seq, d, scores, approx_v, random));
                }

                var probe = try zq.flat.FlatIndex.init(a, .{ .dim = @intCast(d), .bits = bits, .metric = .inner_product });
                defer probe.deinit();
                const per = 2 * probe.bytesPerVector();
                const mk = agg_k.mean();
                const mb = agg_both.mean();
                var buf: [40]u8 = undefined;
                std.debug.print("  {s:>18} {d:>9} {d:>11.4} {d:>10.4} {d:>11.4}\n", .{
                    try std.fmt.bufPrint(&buf, "b={d} {s} K", .{ bits, variant.label }),
                    per / 2,
                    mk.score_rmse,
                    mk.weight_tv,
                    mk.out_rel,
                });
                var buf2: [40]u8 = undefined;
                std.debug.print("  {s:>18} {d:>9} {d:>11.4} {d:>10.4} {d:>11.4}\n", .{
                    try std.fmt.bufPrint(&buf2, "b={d} {s} K+V", .{ bits, variant.label }),
                    per,
                    mb.score_rmse,
                    mb.weight_tv,
                    mb.out_rel,
                });
            }
        }
        std.debug.print("\n", .{});
    }
}
