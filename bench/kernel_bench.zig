//! Kernel-level timing, with no index overhead in the path.
//!
//! An end-to-end measurement showed compact and expanded scans at the same speed, and
//! it would be easy to read that as "unpacking is free". It could equally mean the
//! per-vector overhead around the kernel is large enough to hide the difference.
//! Timing the kernels alone separates those.
//!
//! **Reports the minimum of several trials, not a single one.** Run-to-run spread on
//! this machine reaches ±15%, which is larger than most of the differences worth
//! measuring: the packed/expanded ratio at 4 bits came out anywhere from 1.01× to
//! 1.53× across four identical runs. Noise only ever *adds* time — scheduling,
//! frequency, cache eviction — so the minimum is the least contaminated sample and
//! the right estimator. The spread is printed alongside so a difference can be
//! checked against it rather than assumed significant.
const std = @import("std");
const zq = @import("zquant");
const Timer = @import("timer.zig").Timer;

pub fn main() !void {
    const a = std.heap.smp_allocator;
    const dim: u32 = 256;
    const n: usize = 50_000;
    const reps = 10;
    const trials = 7;

    var prng = std.Random.DefaultPrng.init(1);
    const random = prng.random();

    for ([_]u6{ 2, 3, 4 }) |bits| {
        var cb = try zq.codebook.Codebook.init(a, zq.density.Density.sphereCoord(dim), bits);
        defer cb.deinit();
        const table = zq.simd_scan.Table.init(cb.centroids);
        const layout = zq.packing.Layout.init(dim, bits);

        // Packed corpus.
        const packed_stride = layout.stride();
        const packed_store = try a.alloc(u8, n * packed_stride);
        defer a.free(packed_store);
        // Expanded corpus: one int8 per coordinate.
        const expanded_store = try a.alloc(i8, n * dim);
        defer a.free(expanded_store);

        const codes = try a.alloc(u8, dim);
        defer a.free(codes);
        const raw = try a.alloc(f32, dim);
        defer a.free(raw);
        const sigma = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
        const values: [16]i8 = table.values;
        for (0..n) |i| {
            for (raw) |*v| v.* = random.floatNorm(f32) * sigma;
            cb.encodeSlice(raw, codes);
            layout.pack(codes, packed_store[i * packed_stride ..][0..packed_stride]);
            for (expanded_store[i * dim ..][0..dim], codes) |*dst, c| dst.* = values[c];
        }

        var packed_query = try zq.simd_scan.Query.init(a, layout);
        defer packed_query.deinit();
        var flat_query = try zq.simd_scan.Query.initSequential(a, dim);
        defer flat_query.deinit();
        const rotated = try a.alloc(f32, dim);
        defer a.free(rotated);
        for (rotated) |*v| v.* = random.floatNorm(f32) * sigma;
        packed_query.load(rotated);
        flat_query.load(rotated);

        var sink: f32 = 0;

        var packed_ns: u64 = std.math.maxInt(u64);
        var packed_max: u64 = 0;
        var expanded_ns: u64 = std.math.maxInt(u64);
        var expanded_max: u64 = 0;

        for (0..trials) |_| {
            var t = Timer.start();
            for (0..reps) |_| {
                for (0..n) |i| {
                    sink += zq.simd_scan.scoreInt8(
                        layout,
                        table,
                        packed_query,
                        packed_store[i * packed_stride ..][0..packed_stride],
                        dim,
                    );
                }
            }
            const took = t.read() / reps;
            packed_ns = @min(packed_ns, took);
            packed_max = @max(packed_max, took);

            t.reset();
            for (0..reps) |_| {
                for (0..n) |i| {
                    sink += zq.simd_scan.scoreExpanded(
                        expanded_store[i * dim ..][0..dim],
                        flat_query,
                        table.scale,
                        dim,
                    );
                }
            }
            const took2 = t.read() / reps;
            expanded_ns = @min(expanded_ns, took2);
            expanded_max = @max(expanded_max, took2);
        }

        const fp: f64 = @floatFromInt(packed_ns);
        const fe: f64 = @floatFromInt(expanded_ns);
        const fnn: f64 = @floatFromInt(n);
        const spread_p = (@as(f64, @floatFromInt(packed_max)) - fp) / fp * 100.0;
        const spread_e = (@as(f64, @floatFromInt(expanded_max)) - fe) / fe * 100.0;
        std.debug.print(
            "b={d}: packed {d:>6.2} ns/vec (+{d:>4.1}%)  expanded {d:>6.2} ns/vec (+{d:>4.1}%)  ratio {d:.2}x\n",
            .{ bits, fp / fnn, spread_p, fe / fnn, spread_e, fp / fe },
        );
        std.mem.doNotOptimizeAway(sink);
    }
}
