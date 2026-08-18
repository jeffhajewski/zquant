//! A scalar codebook: the Lloyd-Max solution, narrowed to f32 for the kernels.
//!
//! Solved at runtime for now. Precomputing these as comptime tables is a P1 concern
//! (docs/DESIGN.md §3); it changes construction cost, not results.

const std = @import("std");
const Allocator = std.mem.Allocator;
const lloyd_max = @import("../math/lloyd_max.zig");
const Density = @import("../math/density.zig").Density;
const simd_encode = @import("../simd/encode.zig");

pub const max_bits: u6 = 8;

pub const Codebook = struct {
    bits: u6,
    /// Reconstruction levels, ascending. `1 << bits` of them.
    centroids: []f32,
    /// Midpoints between consecutive centroids. `(1 << bits) - 1` of them.
    thresholds: []f32,
    /// Distortion relative to the density's variance: the paper's D_mse.
    normalized_distortion: f64,

    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        density: Density,
        bits: u6,
    ) (lloyd_max.Error)!Codebook {
        std.debug.assert(bits <= max_bits);

        // Zero bits: a single level at the density's mean, which is zero for both
        // symmetric densities here. Not a degenerate case to guard against but a
        // real configuration — `prod` at b=1 spends its whole budget on the QJL
        // sketch and needs an MSE stage that reconstructs nothing.
        if (bits == 0) {
            const centroids = try allocator.alloc(f32, 1);
            errdefer allocator.free(centroids);
            const thresholds = try allocator.alloc(f32, 0);
            centroids[0] = 0;
            return .{
                .bits = 0,
                .centroids = centroids,
                .thresholds = thresholds,
                // Reconstructing zero leaves the entire variance as error.
                .normalized_distortion = 1.0,
                .allocator = allocator,
            };
        }

        var solution = try lloyd_max.solveBits(allocator, density, bits, .{});
        defer solution.deinit();

        const centroids = try allocator.alloc(f32, solution.centroids.len);
        errdefer allocator.free(centroids);
        const thresholds = try allocator.alloc(f32, solution.thresholds.len);
        errdefer allocator.free(thresholds);

        for (centroids, solution.centroids) |*dst, src| dst.* = @floatCast(src);
        // Narrow the centroids first, then rebuild thresholds from the narrowed
        // values. Rounding the f64 thresholds independently could place one off the
        // true f32 midpoint, so `encode` would disagree with a direct argmin over
        // the f32 centroids for inputs sitting in the gap.
        for (thresholds, 0..) |*t, i| {
            t.* = 0.5 * (centroids[i] + centroids[i + 1]);
        }

        return .{
            .bits = bits,
            .centroids = centroids,
            .thresholds = thresholds,
            .normalized_distortion = solution.normalized_distortion,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Codebook) void {
        self.allocator.free(self.centroids);
        self.allocator.free(self.thresholds);
        self.* = undefined;
    }

    pub fn levels(self: Codebook) usize {
        return self.centroids.len;
    }

    /// Nearest centroid, as the count of thresholds `v` exceeds.
    ///
    /// The reference form of what the SIMD encoder must reproduce bit-for-bit. Kept
    /// as a plain scan rather than a binary search so it stays obviously correct;
    /// the fast variants are benchmarked against it (docs/DESIGN.md §4.1).
    pub fn encode(self: Codebook, v: f32) u8 {
        var idx: u8 = 0;
        for (self.thresholds) |t| {
            if (v > t) idx += 1 else break;
        }
        return idx;
    }

    pub fn decode(self: Codebook, code: u8) f32 {
        return self.centroids[code];
    }

    /// Vectorized encode. Verified bit-for-bit against `encode` above, which is in
    /// turn verified against a direct argmin.
    pub fn encodeSlice(self: Codebook, src: []const f32, dst: []u8) void {
        std.debug.assert(src.len == dst.len);
        simd_encode.encodeSlice(self.bits, self.thresholds, src, dst);
    }

    /// The scalar path, kept callable so tests can compare against it directly.
    pub fn encodeSliceScalar(self: Codebook, src: []const f32, dst: []u8) void {
        std.debug.assert(src.len == dst.len);
        for (src, dst) |v, *c| c.* = self.encode(v);
    }

    pub fn decodeSlice(self: Codebook, src: []const u8, dst: []f32) void {
        std.debug.assert(src.len == dst.len);
        for (src, dst) |c, *v| v.* = self.decode(c);
    }
};

// -- tests -------------------------------------------------------------------

const testing = std.testing;

test "encode agrees with a direct argmin over the f32 centroids" {
    // The property that matters after narrowing to f32: threshold-counting and
    // nearest-centroid must not disagree anywhere.
    for (1..6) |bits| {
        var cb = try Codebook.init(testing.allocator, Density.gauss(1.0), @intCast(bits));
        defer cb.deinit();

        var v: f32 = -5.0;
        while (v <= 5.0) : (v += 0.0007) {
            const got = cb.encode(v);
            var best: u8 = 0;
            var best_dist = @abs(v - cb.centroids[0]);
            for (cb.centroids, 0..) |c, i| {
                const dist = @abs(v - c);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = @intCast(i);
                }
            }
            try testing.expectEqual(best, got);
        }
    }
}

test "codebook shape and ordering" {
    for (1..7) |bits| {
        var cb = try Codebook.init(testing.allocator, Density.sphereCoord(128), @intCast(bits));
        defer cb.deinit();
        try testing.expectEqual(@as(usize, 1) << @intCast(bits), cb.levels());
        try testing.expectEqual(cb.levels() - 1, cb.thresholds.len);
        for (1..cb.levels()) |i| try testing.expect(cb.centroids[i] > cb.centroids[i - 1]);
    }
}

test "round trip through encode and decode lands on a centroid" {
    var cb = try Codebook.init(testing.allocator, Density.gauss(1.0), 4);
    defer cb.deinit();
    for (cb.centroids) |c| {
        try testing.expectEqual(c, cb.decode(cb.encode(c)));
    }
}

test "slice helpers match the scalar path" {
    var cb = try Codebook.init(testing.allocator, Density.gauss(1.0), 3);
    defer cb.deinit();

    const src = [_]f32{ -3.0, -0.4, 0.0, 0.25, 1.9, 7.0 };
    var codes: [6]u8 = undefined;
    var out: [6]f32 = undefined;
    cb.encodeSlice(&src, &codes);
    cb.decodeSlice(&codes, &out);

    var scalar_codes: [6]u8 = undefined;
    cb.encodeSliceScalar(&src, &scalar_codes);
    try testing.expectEqualSlices(u8, &scalar_codes, &codes);

    for (src, codes, out) |v, code, decoded| {
        try testing.expectEqual(cb.encode(v), code);
        try testing.expectEqual(cb.centroids[code], decoded);
    }
}

test "zero bits yields a single zero level" {
    var cb = try Codebook.init(testing.allocator, Density.sphereCoord(256), 0);
    defer cb.deinit();
    try testing.expectEqual(@as(usize, 1), cb.levels());
    try testing.expectEqual(@as(f32, 0), cb.centroids[0]);
    try testing.expectEqual(@as(usize, 0), cb.thresholds.len);
    try testing.expectEqual(@as(f64, 1.0), cb.normalized_distortion);
    for ([_]f32{ -9, 0, 9 }) |v| try testing.expectEqual(@as(u8, 0), cb.encode(v));
}

test "codes always index a valid level" {
    var cb = try Codebook.init(testing.allocator, Density.gauss(1.0), 2);
    defer cb.deinit();
    for ([_]f32{ -1e30, -1, 0, 1, 1e30 }) |v| {
        try testing.expect(cb.encode(v) < cb.levels());
    }
}
