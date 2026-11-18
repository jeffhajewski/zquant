//! Which vector x query tile the batched scan should use on this CPU.
//!
//! `FlatIndex.searchBatch` scores a tile of V stored vectors against Q queries at a
//! time, holding V*Q accumulators in registers. It uses 4x4 everywhere. That suits
//! aarch64's 32 vector registers. x86 below AVX-512 has 16, and its dot product needs
//! three temporaries `sdot` does not, so 4x4 may spill there: the x86_64_v3
//! disassembly of the 4x4 tile showed hundreds of stack accesses in its loop.
//!
//! This times every compact and expanded kernel the index uses at several tile
//! shapes, scoring the same codes against the same 32 queries, and reports each
//! shape's throughput against 4x4. It is a means of choosing the shape; quickbench,
//! which goes through the public API, is what confirms a choice.

const std = @import("std");
const builtin = @import("builtin");
const zq = @import("zquant");
const Timer = @import("timer.zig").Timer;

const Shape = struct { v: usize, q: usize };
const shapes = [_]Shape{
    .{ .v = 4, .q = 4 },
    .{ .v = 4, .q = 2 },
    .{ .v = 2, .q = 4 },
    .{ .v = 2, .q = 2 },
    .{ .v = 1, .q = 4 },
    .{ .v = 4, .q = 1 },
    .{ .v = 3, .q = 3 },
};

const dim: u32 = 256;
const n: usize = 20_000;
const nq: usize = 36; // divisible by every Q above
const trials = 9;

pub fn main() !void {
    const a = std.heap.smp_allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();

    std.debug.print("\ntile_bench: {s}, cpu {s}, {d} vectors x {d} dims, {d} queries, best of {d}\n", .{
        @tagName(builtin.cpu.arch), builtin.cpu.model.name, n, dim, nq, trials,
    });
    std.debug.print("{s:<22}", .{"kernel"});
    inline for (shapes) |s| std.debug.print("{s:>9}", .{std.fmt.comptimePrint("{d}x{d}", .{ s.v, s.q })});
    std.debug.print("   (M scores/s; then relative to 4x4)\n", .{});

    const sigma = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
    const rotated = try a.alloc(f32, dim);
    defer a.free(rotated);
    const raw = try a.alloc(f32, dim);
    defer a.free(raw);
    const codes = try a.alloc(u8, dim);
    defer a.free(codes);
    const out = try a.alloc(f32, 4 * nq);
    defer a.free(out);

    // Code bits 1..4: index bits 2..5. Bit 3 is the bit-plane layout.
    inline for ([_]u6{ 1, 2, 3, 4 }) |bits| {
        var cb = try zq.codebook.Codebook.init(a, zq.density.Density.sphereCoord(dim), bits);
        defer cb.deinit();
        const table = zq.simd_scan.Table.init(cb.centroids);
        const layout = zq.packing.Layout.init(dim, bits);
        const stride = layout.stride();
        const store = try a.alloc(u8, n * stride);
        defer a.free(store);
        const expanded = try a.alloc(i8, n * dim);
        defer a.free(expanded);
        const values: [16]i8 = table.values;
        for (0..n) |i| {
            for (raw) |*v| v.* = random.floatNorm(f32) * sigma;
            cb.encodeSlice(raw, codes);
            layout.pack(codes, store[i * stride ..][0..stride]);
            for (expanded[i * dim ..][0..dim], codes) |*dst, c| dst.* = values[c];
        }

        var queries: [nq]zq.simd_scan.Query = undefined;
        var seq: [nq]zq.simd_scan.Query = undefined;
        for (0..nq) |i| {
            queries[i] = try zq.simd_scan.Query.init(a, layout);
            seq[i] = try zq.simd_scan.Query.initSequential(a, dim);
            for (rotated) |*v| v.* = random.floatNorm(f32) * sigma;
            queries[i].load(rotated);
            seq[i].load(rotated);
        }
        defer for (0..nq) |i| {
            queries[i].deinit();
            seq[i].deinit();
        };

        // The compact kernels, where the layout has a tiled form.
        if (zq.simd_scan.canTile(layout)) {
            var rates: [shapes.len]f64 = undefined;
            inline for (shapes, 0..) |s, si| {
                var best: u64 = std.math.maxInt(u64);
                var sink: f32 = 0;
                for (0..trials) |_| {
                    var t = Timer.start();
                    var v0: usize = 0;
                    while (v0 + s.v <= n) : (v0 += s.v) {
                        var i: usize = 0;
                        while (i + s.q <= nq) : (i += s.q) {
                            zq.simd_scan.scoreInt8Tiled(s.v, s.q, layout, table, queries[i..], store.ptr + v0 * stride, stride, dim, out[i..], nq);
                        }
                        sink += out[0];
                    }
                    best = @min(best, t.read());
                }
                std.mem.doNotOptimizeAway(sink);
                rates[si] = @as(f64, @floatFromInt(n * nq)) / @as(f64, @floatFromInt(best)) * 1e3;
            }
            report(std.fmt.comptimePrint("compact, {d}-bit codes", .{bits}), &rates);
        }

        // The expanded kernel is the same for every width; time it once.
        if (bits == 4) {
            var rates: [shapes.len]f64 = undefined;
            inline for (shapes, 0..) |s, si| {
                var best: u64 = std.math.maxInt(u64);
                var sink: f32 = 0;
                for (0..trials) |_| {
                    var t = Timer.start();
                    var v0: usize = 0;
                    while (v0 + s.v <= n) : (v0 += s.v) {
                        var i: usize = 0;
                        while (i + s.q <= nq) : (i += s.q) {
                            zq.simd_scan.scoreExpandedTiled(s.v, s.q, expanded.ptr + v0 * dim, dim, seq[i..], table.scale, dim, out[i..], nq);
                        }
                        sink += out[0];
                    }
                    best = @min(best, t.read());
                }
                std.mem.doNotOptimizeAway(sink);
                rates[si] = @as(f64, @floatFromInt(n * nq)) / @as(f64, @floatFromInt(best)) * 1e3;
            }
            report("expanded", &rates);
        }
    }
}

fn report(name: []const u8, rates: []const f64) void {
    std.debug.print("{s:<22}", .{name});
    for (rates) |r| std.debug.print("{d:>9.0}", .{r});
    std.debug.print("\n{s:<22}", .{""});
    for (rates) |r| std.debug.print("{d:>9.2}", .{r / rates[0]});
    std.debug.print("\n", .{});
}
