//! Why is recall behind turbovec at matched storage? Check the assumptions the
//! codebook rests on, against real rotated SIFT data rather than synthetic vectors.
const std = @import("std");
const zq = @import("zquant");

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

pub fn main() !void {
    const a = std.heap.smp_allocator;
    var threaded = std.Io.Threaded.init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const base = try readFvecs(a, io, "data/sift-norm/base.fvecs");
    defer a.free(base.data);
    const d = base.dim;
    const n = @min(base.count, 3000);

    var q = try zq.mse.Mse.init(a, .{ .dim = d, .bits = 4, .seed = 0x5EED });
    defer q.deinit();
    var ws = try zq.mse.Workspace.init(a, q);
    defer ws.deinit();

    const rotated = try a.alloc(f32, q.padded);
    defer a.free(rotated);
    const staging = try a.alloc(f32, q.padded);
    defer a.free(staging);

    // 1. Do rotated real coordinates match the sphere density the codebook assumes?
    var sum2: f64 = 0;
    var sum4: f64 = 0;
    var max_abs: f64 = 0;
    var count: usize = 0;
    for (0..n) |i| {
        _ = q.encodeRotated(base.data[i * d ..][0..d], rotated, staging);
        for (rotated) |v| {
            const sq = @as(f64, v) * v;
            sum2 += sq;
            sum4 += sq * sq;
            max_abs = @max(max_abs, @abs(@as(f64, v)));
            count += 1;
        }
    }
    const fc: f64 = @floatFromInt(count);
    const fd: f64 = @floatFromInt(q.padded);
    std.debug.print("rotated SIFT coordinates (d={d}, n={d})\n", .{ d, n });
    std.debug.print("  E[y^2]  measured {e:.4}   theory 1/d = {e:.4}   ratio {d:.3}\n", .{ sum2 / fc, 1.0 / fd, (sum2 / fc) * fd });
    std.debug.print("  E[y^4]  measured {e:.4}   theory 3/(d(d+2)) = {e:.4}   ratio {d:.3}\n", .{ sum4 / fc, 3.0 / (fd * (fd + 2.0)), (sum4 / fc) / (3.0 / (fd * (fd + 2.0))) });
    std.debug.print("  kurtosis {d:.3} (3.0 = gaussian; higher = heavy tails)\n", .{(sum4 / fc) / ((sum2 / fc) * (sum2 / fc))});
    std.debug.print("  max|y| {d:.4} = {d:.2} sigma\n", .{ max_abs, max_abs / @sqrt(sum2 / fc) });

    // 1b. How much does the reconstruction's norm vary between vectors?
    //
    // The score is ||x|| * <p, y_hat>. MSE-optimal reconstruction shrinks y_hat
    // toward zero; a *uniform* shrinkage is harmless for ranking, since it scales
    // every score equally. Per-vector variation is not: it moves vectors up and
    // down the ranking for reasons unrelated to similarity.
    {
        const recon = try a.alloc(f32, q.padded);
        defer a.free(recon);
        const c = try a.alloc(u8, q.codeLen());
        defer a.free(c);

        var sum: f64 = 0;
        var sumsq: f64 = 0;
        for (0..n) |i| {
            _ = q.encodeRotated(base.data[i * d ..][0..d], rotated, staging);
            q.codebook.encodeSlice(rotated, c);
            q.decodeRotated(c, recon);
            var nn: f64 = 0;
            for (recon) |v| nn += @as(f64, v) * v;
            const len = @sqrt(nn);
            sum += len;
            sumsq += len * len;
        }
        const fn_: f64 = @floatFromInt(n);
        const mean = sum / fn_;
        const sd = @sqrt(sumsq / fn_ - mean * mean);
        std.debug.print("\n||y_hat|| across vectors (b=4): mean {d:.4} sd {d:.4} -> cv {d:.4}\n", .{ mean, sd, sd / mean });
        std.debug.print("  a uniform shrink is harmless for ranking; this cv is the part that is not\n", .{});
    }

    // 1c. Estimator error against the score gaps it must resolve.
    //
    // R@10 depends on separating the true 1st from the true 10th neighbour. If the
    // estimator's error is comparable to that gap, the top-10 set shuffles no matter
    // how good the reconstruction is.
    {
        const queries = try readFvecs(a, io, "data/sift-norm/query.fvecs");
        defer a.free(queries.data);

        var prod = try zq.prod.Prod.init(a, .{ .dim = d, .bits = 5, .seed = 0x5EED });
        defer prod.deinit();
        var pws = try zq.prod.Workspace.init(a, prod);
        defer pws.deinit();
        var state = try zq.prod.QueryState.init(a, prod);
        defer state.deinit();

        const pc = try a.alloc(u8, prod.codeLen());
        defer a.free(pc);
        const ps = try a.alloc(u8, prod.sketchLen());
        defer a.free(ps);

        const m = 400;
        const sims = try a.alloc(f64, m);
        defer a.free(sims);

        var err2: f64 = 0;
        var cnt: usize = 0;
        var gap_sum: f64 = 0;

        for (0..20) |qi| {
            const query = queries.data[qi * d ..][0..d];
            prod.prepareQuery(query, &state, &pws);
            for (0..m) |i| {
                const row = base.data[i * d ..][0..d];
                const sc = prod.encode(row, pc, ps, &pws);
                const est: f64 = prod.dot(state, pc, ps, sc);
                var truth: f64 = 0;
                for (query, row) |qv, xv| truth += @as(f64, qv) * xv;
                sims[i] = truth;
                err2 += (est - truth) * (est - truth);
                cnt += 1;
            }
            std.mem.sort(f64, sims, {}, std.sort.desc(f64));
            gap_sum += sims[0] - sims[9];
        }
        const rms = @sqrt(err2 / @as(f64, @floatFromInt(cnt)));
        const gap = gap_sum / 20.0;
        std.debug.print("\nestimator error vs score gaps (b=5)\n", .{});
        std.debug.print("  RMS estimate error      {d:.5}\n", .{rms});
        std.debug.print("  mean sim(1st)-sim(10th) {d:.5}  (over a 400-vector sample)\n", .{gap});
        std.debug.print("  error / gap             {d:.2}  -> top-10 ordering is {s}\n", .{ rms / gap, if (rms > gap) "dominated by noise" else "partially resolvable" });
    }

    // 1d. Estimator RMS measured EXACTLY as turbovec's was: a 400-vector index,
    // every vector scored, unconditioned. An earlier attempt measured only the
    // vectors the index chose to return, which is a high-scoring subset and made
    // the two numbers incomparable.
    {
        const queries = try readFvecs(a, io, "data/sift-norm/query.fvecs");
        defer a.free(queries.data);
        const probe = 400;

        std.debug.print("\nindex estimator RMS on a {d}-vector corpus, all scored\n", .{probe});
        for ([_]u6{ 2, 3, 4, 5 }) |bits| {
            var index = try zq.flat.FlatIndex.init(a, .{
                .dim = d,
                .bits = bits,
                .metric = .inner_product,
                .seed = 0x5EED,
            });
            defer index.deinit();
            try index.addBatch(base.data[0 .. probe * d]);

            var searcher = try zq.flat.FlatIndex.Searcher.init(a, index, probe);
            defer searcher.deinit();

            var err2: f64 = 0;
            var cnt: usize = 0;
            for (0..20) |qi| {
                const query = queries.data[qi * d ..][0..d];
                for (index.search(query, &searcher)) |e| {
                    var exact_ip: f64 = 0;
                    for (query, base.data[@as(usize, e.id) * d ..][0..d]) |qv, xv| {
                        exact_ip += @as(f64, qv) * xv;
                    }
                    const delta = @as(f64, e.score) - exact_ip;
                    err2 += delta * delta;
                    cnt += 1;
                }
            }
            std.debug.print("  bits={d} ({d}B): RMS {d:.5}\n", .{ bits, index.bytesPerVector(), @sqrt(err2 / @as(f64, @floatFromInt(cnt))) });
        }
        std.debug.print("  turbovec, same protocol: bits=2 0.02677  bits=3 0.01370  bits=4 0.00718\n", .{});
    }

    // 1e. Does per-coordinate structure survive the rotation?
    //
    // A Haar rotation makes every coordinate identically distributed, so a single
    // shared codebook is optimal. A *structured* rotation (RHT) on structured data
    // may not. If the per-coordinate standard deviations vary, a per-coordinate
    // scale would be worth fitting — that is what turbovec's calibrate() appears to
    // do, and it buys them +2.4 to +5.1 points.
    {
        const sigma = try a.alloc(f64, q.padded);
        defer a.free(sigma);
        @memset(sigma, 0);
        for (0..n) |i| {
            _ = q.encodeRotated(base.data[i * d ..][0..d], rotated, staging);
            for (rotated, sigma) |v, *acc| acc.* += @as(f64, v) * v;
        }
        var mean: f64 = 0;
        for (sigma) |*acc| {
            acc.* = @sqrt(acc.* / @as(f64, @floatFromInt(n)));
            mean += acc.*;
        }
        mean /= @floatFromInt(q.padded);
        var sd: f64 = 0;
        var lo: f64 = 1e30;
        var hi: f64 = 0;
        for (sigma) |v| {
            sd += (v - mean) * (v - mean);
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        sd = @sqrt(sd / @as(f64, @floatFromInt(q.padded)));
        std.debug.print("\nper-coordinate sigma after rotation (d={d})\n", .{q.padded});
        std.debug.print("  mean {e:.4}  sd {e:.4}  cv {d:.4}\n", .{ mean, sd, sd / mean });
        std.debug.print("  min/mean {d:.3}  max/mean {d:.3}\n", .{ lo / mean, hi / mean });
        std.debug.print("  -> a per-coordinate scale is worth fitting only if cv is well above 0\n", .{});
    }

    // 1f. Does a per-coordinate MEAN survive the rotation?
    //
    // The codebook is symmetric about zero. SIFT vectors are non-negative histograms
    // with a strong mean vector μ; after rotation, coordinate j has mean πⱼᵀμ, which
    // need not be small. If it is comparable to σ_j the codebook is centred in the
    // wrong place and half its levels are wasted.
    {
        const mean = try a.alloc(f64, q.padded);
        defer a.free(mean);
        const var_ = try a.alloc(f64, q.padded);
        defer a.free(var_);
        @memset(mean, 0);
        @memset(var_, 0);
        for (0..n) |i| {
            _ = q.encodeRotated(base.data[i * d ..][0..d], rotated, staging);
            for (rotated, mean, var_) |v, *m, *s2| {
                m.* += v;
                s2.* += @as(f64, v) * v;
            }
        }
        const fn_: f64 = @floatFromInt(n);
        var worst: f64 = 0;
        var mean_ratio: f64 = 0;
        for (mean, var_) |*m, *s2| {
            m.* /= fn_;
            const sd = @sqrt(s2.* / fn_ - m.* * m.*);
            const ratio = if (sd > 0) @abs(m.*) / sd else 0;
            worst = @max(worst, ratio);
            mean_ratio += ratio;
        }
        mean_ratio /= @floatFromInt(q.padded);
        std.debug.print("\nper-coordinate |mean|/sigma after rotation\n", .{});
        std.debug.print("  average {d:.4}   worst {d:.4}\n", .{ mean_ratio, worst });
        std.debug.print("  -> a symmetric codebook assumes this is ~0\n", .{});
    }

    // 2. Measured reconstruction distortion vs the paper's value.
    const codes = try a.alloc(u8, q.codeLen());
    defer a.free(codes);
    const back = try a.alloc(f32, d);
    defer a.free(back);
    const want = [_]f64{ 0.36, 0.117, 0.03, 0.009 };
    std.debug.print("\nD_mse on real SIFT vs paper (uniform-on-sphere theory)\n", .{});
    for (1..5) |bits| {
        var qq = try zq.mse.Mse.init(a, .{ .dim = d, .bits = @intCast(bits), .seed = 0x5EED });
        defer qq.deinit();
        var w2 = try zq.mse.Workspace.init(a, qq);
        defer w2.deinit();
        const c2 = try a.alloc(u8, qq.codeLen());
        defer a.free(c2);

        var dist: f64 = 0;
        for (0..n) |i| {
            const row = base.data[i * d ..][0..d];
            const norm = qq.encode(row, c2, &w2);
            qq.decode(c2, norm, back, &w2);
            for (row, back) |x, y| dist += (@as(f64, x) - y) * (@as(f64, x) - y);
        }
        dist /= @floatFromInt(n);
        std.debug.print("  b={d}: measured {d:.4}   paper {d:.4}   ratio {d:.2}x\n", .{ bits, dist, want[bits - 1], dist / want[bits - 1] });
    }
}
