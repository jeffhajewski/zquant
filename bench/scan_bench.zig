const std = @import("std");
const zq = @import("zquant");
const Timer = @import("timer.zig").Timer;

pub fn main() !void {
    // smp_allocator: 0.16 dropped GeneralPurposeAllocator, and benchmarks want
    // throughput rather than leak tracking.
    const a = std.heap.smp_allocator;

    const dim: u32 = 1024;
    const n: usize = 200_000;
    const bits: u6 = 4;

    const layout = zq.packing.Layout.init(dim, bits);
    var cb = try zq.codebook.Codebook.init(a, zq.density.Density.sphereCoord(dim), bits);
    defer cb.deinit();

    const storage = try a.alignedAlloc(u8, .@"64", layout.totalBytes(n));
    defer a.free(storage);
    const raw = try a.alloc(f32, dim);
    defer a.free(raw);
    const codes = try a.alloc(u8, dim);
    defer a.free(codes);

    var prng = std.Random.DefaultPrng.init(1);
    const sigma = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
    for (0..n) |i| {
        for (raw) |*v| v.* = prng.random().floatNorm(f32) * sigma;
        cb.encodeSlice(raw, codes);
        layout.pack(codes, layout.vectorSliceMut(storage, i));
    }

    const rotated = try a.alloc(f32, dim);
    defer a.free(rotated);
    for (rotated) |*v| v.* = prng.random().floatNorm(f32) * sigma;

    var query = try zq.simd_scan.Query.init(a, layout);
    defer query.deinit();
    query.load(rotated);
    const table = zq.simd_scan.Table.init(cb.centroids);

    const bytes = layout.totalBytes(n);
    std.debug.print("d={d} n={d} b={d} | corpus {d:.1} MB ({d} B/vector)\n",
        .{ dim, n, bits, @as(f64, @floatFromInt(bytes)) / 1e6, layout.stride() });

    // --- int8 SIMD kernel
    var sink: f32 = 0;
    var t = Timer.start();
    const reps = 5;
    for (0..reps) |_| {
        for (0..n) |i| sink += zq.simd_scan.scoreInt8(layout, table, query, layout.vectorSlice(storage, i), dim);
    }
    const simd_ns = t.read() / reps;

    // --- exact f32 reference
    t.reset();
    for (0..n) |i| sink += zq.simd_scan.scoreExact(layout, cb.centroids, rotated, layout.vectorSlice(storage, i));
    const exact_ns = t.read();

    // --- unquantized f32 brute force, for scale
    const flat = try a.alloc(f32, 20_000 * dim);
    defer a.free(flat);
    for (flat) |*v| v.* = prng.random().floatNorm(f32) * sigma;
    t.reset();
    for (0..20_000) |i| {
        var acc: f32 = 0;
        const row = flat[i * dim ..][0..dim];
        for (row, rotated) |x, p| acc += x * p;
        sink += acc;
    }
    const flat_ns = t.read();

    const simd_vps = @as(f64, @floatFromInt(n)) / (@as(f64, @floatFromInt(simd_ns)) / 1e9);
    const exact_vps = @as(f64, @floatFromInt(n)) / (@as(f64, @floatFromInt(exact_ns)) / 1e9);
    const flat_vps = 20_000.0 / (@as(f64, @floatFromInt(flat_ns)) / 1e9);

    std.debug.print("  int8 simd : {d:>9.0} vec/s  {d:>6.1} GB/s  {d:.2} ns/vector\n",
        .{ simd_vps, simd_vps * @as(f64, @floatFromInt(layout.stride())) / 1e9,
           @as(f64, @floatFromInt(simd_ns)) / @as(f64, @floatFromInt(n)) });
    std.debug.print("  exact f32 : {d:>9.0} vec/s  ({d:.1}x slower)\n", .{ exact_vps, simd_vps / exact_vps });
    std.debug.print("  f32 brute : {d:>9.0} vec/s  ({d:.1}x slower than int8, {d:.1} GB/s)\n",
        .{ flat_vps, simd_vps / flat_vps, flat_vps * @as(f64, dim) * 4.0 / 1e9 });
    std.debug.print("  [sink {d}]\n", .{sink});
}
