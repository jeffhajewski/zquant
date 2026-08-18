const std = @import("std");
const zq = @import("zquant");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const a = gpa.allocator();

    const n = 1 << 22; // 4M coordinates
    const src = try a.alloc(f32, n);
    defer a.free(src);
    const dst = try a.alloc(u8, n);
    defer a.free(dst);
    var prng = std.Random.DefaultPrng.init(1);
    for (src) |*v| v.* = (prng.random().float(f32) - 0.5) * 6.0;

    for ([_]u6{ 2, 4 }) |bits| {
        var cb = try zq.codebook.Codebook.init(a, zq.density.Density.gauss(1.0), bits);
        defer cb.deinit();

        var t = try std.time.Timer.start();
        for (0..5) |_| cb.encodeSliceScalar(src, dst);
        const scalar_ns = t.read() / 5;

        t.reset();
        for (0..5) |_| cb.encodeSlice(src, dst);
        const simd_ns = t.read() / 5;

        const scalar_rate = @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(scalar_ns));
        const simd_rate = @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(simd_ns));
        std.debug.print("b={d}: scalar {d:.2} Gcoord/s | simd {d:.2} Gcoord/s | {d:.2}x\n",
            .{ bits, scalar_rate, simd_rate, simd_rate / scalar_rate });
    }
}
