const std = @import("std");
const zq = @import("zquant");

fn randomUnit(buf: []f32, random: std.Random) void {
    var n: f64 = 0;
    for (buf) |*v| { const g = random.floatNorm(f32); v.* = g; n += @as(f64,g)*g; }
    const inv: f32 = @floatCast(1.0/@sqrt(n));
    for (buf) |*v| v.* *= inv;
}
fn dot(a: []const f32, b: []const f32) f64 {
    var s: f64 = 0; for (a,b) |x,y| s += @as(f64,x)*y; return s;
}

fn recall(a: std.mem.Allocator, dim: u32, bits: u6, n: usize, nq: usize, ks: []const usize, clustered: bool) ![]f64 {
    var q = try zq.prod.Prod.init(a, .{ .dim = dim, .bits = bits, .seed = 0x5EED });
    defer q.deinit();
    var ws = try zq.prod.Workspace.init(a, q); defer ws.deinit();
    var st = try zq.prod.QueryState.init(a, q); defer st.deinit();

    const corpus = try a.alloc(f32, n*dim); defer a.free(corpus);
    const codes = try a.alloc(u8, n*q.codeLen()); defer a.free(codes);
    const sk = try a.alloc(u8, n*q.sketchLen()); defer a.free(sk);
    const sc = try a.alloc(zq.prod.Scalars, n); defer a.free(sc);

    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const nc = 32;
    const centers = try a.alloc(f32, nc*dim); defer a.free(centers);
    for (0..nc) |c| randomUnit(centers[c*dim..][0..dim], rnd);

    for (0..n) |i| {
        const row = corpus[i*dim..][0..dim];
        if (clustered) {
            const c = rnd.uintLessThan(usize, nc);
            for (row, centers[c*dim..][0..dim]) |*v, ce| v.* = ce + 0.35*rnd.floatNorm(f32)/@sqrt(@as(f32,@floatFromInt(dim)));
            const nm = zq.mse.euclideanNorm(row);
            for (row) |*v| v.* /= nm;
        } else randomUnit(row, rnd);
        sc[i] = q.encode(row, codes[i*q.codeLen()..][0..q.codeLen()], sk[i*q.sketchLen()..][0..q.sketchLen()], &ws);
    }

    const query = try a.alloc(f32, dim); defer a.free(query);
    const es = try a.alloc(f64, n); defer a.free(es);
    const as_ = try a.alloc(f64, n); defer a.free(as_);
    const order = try a.alloc(u32, n); defer a.free(order);
    const hits = try a.alloc(f64, ks.len);
    @memset(hits, 0);

    for (0..nq) |_| {
        randomUnit(query, rnd);
        q.prepareQuery(query, &st, &ws);
        var best: u32 = 0;
        for (0..n) |i| {
            es[i] = dot(query, corpus[i*dim..][0..dim]);
            if (es[i] > es[best]) best = @intCast(i);
            as_[i] = q.dot(st, codes[i*q.codeLen()..][0..q.codeLen()], sk[i*q.sketchLen()..][0..q.sketchLen()], sc[i]);
        }
        for (order, 0..) |*o, i| o.* = @intCast(i);
        std.mem.sort(u32, order, as_, struct {
            fn lt(s: []const f64, x: u32, y: u32) bool { return s[x] > s[y]; }
        }.lt);
        for (ks, hits) |k, *h| {
            for (order[0..@min(k,n)]) |c| if (c == best) { h.* += 1; break; };
        }
    }
    for (hits) |*h| h.* /= @as(f64,@floatFromInt(nq));
    return hits;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const a = gpa.allocator();
    const ks = [_]usize{ 1, 5, 10, 20, 50, 100 };

    std.debug.print("\nrecall 1@k, 10k corpus, 200 queries\n", .{});
    std.debug.print("{s:>6} {s:>5} {s:>10}", .{ "d", "bits", "data" });
    for (ks) |k| std.debug.print("  1@{d:<4}", .{k});
    std.debug.print("\n", .{});

    for ([_]u32{ 1024 }) |dim| {
        for ([_]u6{ 2, 3, 4, 5, 6 }) |bits| {
            for ([_]bool{ false, true }) |cl| {
                const r = try recall(a, dim, bits, 10_000, 200, &ks, cl);
                defer a.free(r);
                std.debug.print("{d:>6} {d:>5} {s:>10}", .{ dim, bits, if (cl) "clustered" else "uniform" });
                _ = &cl;
                for (r) |v| std.debug.print("  {d:>5.3}", .{v});
                std.debug.print("\n", .{});
            }
        }
    }
}
