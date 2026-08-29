//! Where the low-bit-rate loss actually is.
//!
//! FAISS PQ leads below 25 B/vector — 0.511 against 0.357 at bits=2 on SIFT — and the
//! two candidate explanations call for very different work. If the int8 query
//! quantization or the estimator is losing the points, that is fixable inside the
//! current design. If an exact f32 scan over the same codes scores no better, the
//! codes themselves are the limit and only a learned or vector quantizer closes it.
//!
//! Runs each low bit width under: the shipped configuration, an exact f32 scan of the
//! same codes, and the paper's QJL sketch correction in place of the scalar one.

const std = @import("std");
const zq = @import("zquant");

const K = 10;

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

    const name_raw = try std.Io.Dir.cwd().readFileAlloc(io, "data/dataset.txt", a, .limited(256));
    defer a.free(name_raw);
    const name = std.mem.trim(u8, name_raw, " \n\r\t");
    const dir = try std.fmt.allocPrint(a, "data/{s}", .{name});
    defer a.free(dir);
    const bp = try std.fmt.allocPrint(a, "{s}/base.fvecs", .{dir});
    defer a.free(bp);
    const qp = try std.fmt.allocPrint(a, "{s}/query.fvecs", .{dir});
    defer a.free(qp);
    const tp = try std.fmt.allocPrint(a, "{s}/groundtruth.ivecs", .{dir});
    defer a.free(tp);

    const base = try readFvecs(a, io, bp);
    defer a.free(base.data);
    const queries = try readFvecs(a, io, qp);
    defer a.free(queries.data);
    const truth = try readIvecs(a, io, tp);
    defer a.free(truth.data);

    const d = base.dim;
    const nq = queries.count;
    std.debug.print("low-bit study on {s}: {d}x{d}, {d} queries\n", .{ dir, base.count, d, nq });
    std.debug.print("{s:>5} {s:>18} {s:>6} {s:>8}\n", .{ "bits", "config", "B/vec", "R@10" });

    const Variant = struct { name: []const u8, exact: bool, correction: zq.flat.Correction, sketch: bool };
    const variants = [_]Variant{
        .{ .name = "shipped (int8)", .exact = false, .correction = .scalar, .sketch = true },
        .{ .name = "exact f32 scan", .exact = true, .correction = .scalar, .sketch = true },
        .{ .name = "qjl sketch", .exact = false, .correction = .qjl_sketch, .sketch = true },
    };

    // Fewer dimensions at higher precision, same bytes.
    //
    // A random rotation makes coordinates exchangeable, so keeping only the first m of
    // them is a Johnson-Lindenstrauss projection: it discards signal, but it buys bits
    // for the coordinates that remain. At d=128, 64 dims at 2 bits costs the same 16 B
    // of codes as 128 dims at 1 bit. Whether that trade pays is an empirical question
    // and nothing measured so far answers it.
    {
        var rot = try zq.rotation.Rotation.init(a, d, .hadamard, 0xB0A7, .rht_signs);
        defer rot.deinit();
        const padded = rot.padded;
        const rbuf = try a.alloc(f32, padded);
        defer a.free(rbuf);

        for ([_]u32{ 2, 4 }) |shrink| {
            const m: u32 = @intCast(padded / shrink);
            const proj_base = try a.alloc(f32, base.count * m);
            defer a.free(proj_base);
            for (0..base.count) |i| {
                rot.apply(base.data[i * d ..][0..d], rbuf);
                @memcpy(proj_base[i * m ..][0..m], rbuf[0..m]);
            }
            const proj_q = try a.alloc(f32, nq * m);
            defer a.free(proj_q);
            for (0..nq) |i| {
                rot.apply(queries.data[i * d ..][0..d], rbuf);
                @memcpy(proj_q[i * m ..][0..m], rbuf[0..m]);
            }

            for ([_]u6{ 3, 4, 5 }) |bits| {
                var index = try zq.flat.FlatIndex.init(a, .{
                    .dim = m,
                    .bits = bits,
                    .metric = .inner_product,
                    .seed = 0x5EED,
                    .correction = .scalar,
                });
                defer index.deinit();
                try index.calibrate(proj_base);
                try index.addBatch(proj_base);

                var searcher = try zq.flat.FlatIndex.Searcher.init(a, index, K);
                defer searcher.deinit();
                var recall: f64 = 0;
                for (0..nq) |i| {
                    const res = index.search(proj_q[i * m ..][0..m], &searcher);
                    const gt = truth.data[i * truth.width ..][0..truth.width];
                    var hit: usize = 0;
                    for (res) |e| {
                        for (gt[0..K]) |g| {
                            if (e.id == g) {
                                hit += 1;
                                break;
                            }
                        }
                    }
                    recall += @as(f64, @floatFromInt(hit)) / @as(f64, K);
                }
                var label: [32]u8 = undefined;
                const nm = try std.fmt.bufPrint(&label, "d/{d} at {d}b", .{ shrink, bits });
                std.debug.print("{d:>5} {s:>18} {d:>6} {d:>8.3}\n", .{
                    bits, nm, index.bytesPerVector(), recall / @as(f64, @floatFromInt(nq)),
                });
            }
        }
    }

    for ([_]u6{ 2, 3 }) |bits| {
        for (variants) |vr| {
            var index = try zq.flat.FlatIndex.init(a, .{
                .dim = d,
                .bits = bits,
                .metric = .inner_product,
                .seed = 0x5EED,
                .exact_scan = vr.exact,
                .use_sketch = vr.sketch,
                .correction = vr.correction,
            });
            defer index.deinit();
            try index.calibrate(base.data);
            try index.addBatch(base.data);

            var searcher = try zq.flat.FlatIndex.Searcher.init(a, index, K);
            defer searcher.deinit();

            var recall: f64 = 0;
            for (0..nq) |i| {
                const res = index.search(queries.data[i * d ..][0..d], &searcher);
                const gt = truth.data[i * truth.width ..][0..truth.width];
                var hit: usize = 0;
                for (res) |e| {
                    for (gt[0..K]) |g| {
                        if (e.id == g) {
                            hit += 1;
                            break;
                        }
                    }
                }
                recall += @as(f64, @floatFromInt(hit)) / @as(f64, K);
            }
            std.debug.print("{d:>5} {s:>18} {d:>6} {d:>8.3}\n", .{
                bits, vr.name, index.bytesPerVector(), recall / @as(f64, @floatFromInt(nq)),
            });
        }
    }
}
