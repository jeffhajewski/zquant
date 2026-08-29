//! Isolates the multi-query expanded kernel from the index.
//!
//! The scan sits at a flat ~48-65 G dim/s single-threaded regardless of dimension,
//! dataset, or residency, which is the signature of a kernel pinned on load-issue
//! rather than on memory. The one-query kernel spends two loads per SDOT. Scoring Q
//! queries in one pass over the vector should drop that toward (1+Q)/Q.
//!
//! Reported in G dim/s of *useful work* — query-dimensions scored per second — so the
//! Q=1 and Q=8 rows are directly comparable.

const std = @import("std");
const zq = @import("zquant");
const Timer = @import("timer.zig").Timer;

pub fn main() !void {
    const a = std.heap.smp_allocator;
    const n: usize = 100_000;
    const trials = 7;

    var prng = std.Random.DefaultPrng.init(1);
    const random = prng.random();

    std.debug.print("\nMulti-query expanded kernel, n={d} vectors, min of {d} trials\n", .{ n, trials });
    std.debug.print("{s:>6} {s:>4} {s:>12} {s:>10} {s:>8}\n", .{ "dim", "Q", "G dim/s", "vs Q=1", "spread" });

    inline for ([_]u32{ 256, 784 }) |dim| {
        const cb = try zq.codebook.Codebook.init(a, zq.density.Density.sphereCoord(dim), 5);
        defer @constCast(&cb).deinit();
        const table = zq.simd_scan.Table.init(cb.centroids);

        const store = try a.alloc(i8, n * dim);
        defer a.free(store);
        for (store) |*v| v.* = @intCast(@as(i32, random.intRangeAtMost(i8, -100, 100)));

        const max_q = 8;
        var queries: [max_q]zq.simd_scan.Query = undefined;
        for (0..max_q) |i| queries[i] = try zq.simd_scan.Query.initSequential(a, dim);
        defer for (0..max_q) |i| queries[i].deinit();
        const rotated = try a.alloc(f32, dim);
        defer a.free(rotated);
        const sigma = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
        for (0..max_q) |i| {
            for (rotated) |*v| v.* = random.floatNorm(f32) * sigma;
            queries[i].load(rotated);
        }

        var out: [max_q]f32 = undefined;
        var sink: f32 = 0;
        var base: f64 = 0;

        inline for ([_]usize{ 1, 2, 4, 8 }) |Q| {
            // Fixed work per trial, not fixed iterations: at Q=8 one pass over the
            // corpus is tens of microseconds, which is inside timer noise. The first
            // version of this bench reported 147% spread and an ordering that
            // reversed between runs.
            const per_pass: u64 = @as(u64, n) * dim * Q;
            const reps: u64 = @max(1, 400_000_000 / per_pass);
            var best: u64 = std.math.maxInt(u64);
            var worst: u64 = 0;
            for (0..trials) |_| {
                var t = Timer.start();
                for (0..reps) |_| for (0..n) |i| {
                    zq.simd_scan.scoreExpandedMulti(
                        Q,
                        store[i * dim ..][0..dim],
                        queries[0..],
                        table.scale,
                        dim,
                        out[0..],
                    );
                    sink += out[0];
                };
                const took = t.read() / reps;
                best = @min(best, took);
                worst = @max(worst, took);
            }
            // Useful work: Q queries x dim coordinates, per stored vector.
            const work: f64 = @floatFromInt(per_pass);
            const gdims = work / @as(f64, @floatFromInt(best));
            if (Q == 1) base = gdims;
            const spread = (@as(f64, @floatFromInt(worst - best)) / @as(f64, @floatFromInt(best))) * 100.0;
            std.debug.print("{d:>6} {d:>4} {d:>12.1} {d:>9.2}x {d:>7.0}%\n", .{ dim, Q, gdims, gdims / base, spread });
        }
        // Vector tiling: Q query chunks stay in registers across V stored vectors.
        inline for ([_]usize{ 2, 4 }) |V| {
            inline for ([_]usize{ 4, 8 }) |Q| {
                var tout: [4 * 8]f32 = undefined;
                const per_pass: u64 = @as(u64, n) * dim * Q;
                const reps: u64 = @max(1, 400_000_000 / per_pass);
                var best: u64 = std.math.maxInt(u64);
                var worst: u64 = 0;
                for (0..trials) |_| {
                    var t = Timer.start();
                    for (0..reps) |_| {
                        var i: usize = 0;
                        while (i + V <= n) : (i += V) {
                            zq.simd_scan.scoreExpandedTiled(
                                V,
                                Q,
                                store.ptr + i * dim,
                                dim,
                                queries[0..],
                                table.scale,
                                dim,
                                tout[0..],
                                Q,
                            );
                            sink += tout[0];
                        }
                    }
                    const took = t.read() / reps;
                    best = @min(best, took);
                    worst = @max(worst, took);
                }
                const gdims = @as(f64, @floatFromInt(per_pass)) / @as(f64, @floatFromInt(best));
                const spread = (@as(f64, @floatFromInt(worst - best)) / @as(f64, @floatFromInt(best))) * 100.0;
                std.debug.print("{d:>6} {s:>4} {d:>12.1} {d:>9.2}x {d:>7.0}%\n", .{ dim, std.fmt.comptimePrint("{d}x{d}", .{ V, Q }), gdims, gdims / base, spread });
            }
        }

        std.mem.doNotOptimizeAway(sink);
    }
}
