//! TurboQuant_mse — Algorithm 1 of the paper.
//!
//!     Quant(x):     y   = Π · (x/‖x‖);  idx_j = argmin_k |y_j − c_k|
//!     DeQuant(idx): ỹ_j = c_{idx_j};    x̃ = ‖x‖ · Πᵀ · ỹ
//!
//! Codes are one byte per coordinate here. Bit-packing to b bits is a separate
//! concern (docs/DESIGN.md §3, `quant/packing.zig`) and does not change any result.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Density = @import("../math/density.zig").Density;
const Rotation = @import("../math/rotation.zig").Rotation;
const RotationKind = @import("../math/rotation.zig").Kind;
const rng = @import("../math/rng.zig");
const Codebook = @import("codebook.zig").Codebook;

pub const Params = struct {
    dim: u32,
    bits: u6,
    seed: u64 = 0,
    rotation: RotationKind = .hadamard,
    /// Use the exact sphere-coordinate density rather than its N(0,1/d) limit.
    ///
    /// Defaults on because it is never worse and matters at the dimensions the KV
    /// path uses (docs/DESIGN.md §8.2). The Gaussian limit is available mainly to
    /// quantify what choosing it would cost.
    exact_density: bool = true,
};

/// Per-thread scratch. Held by the caller so encode/decode allocate nothing and are
/// safe to call concurrently from several threads against one quantizer.
pub const Workspace = struct {
    /// Two buffers, because the rotation stages through its destination and then
    /// works in place: source and destination must not overlap.
    staging: []f32,
    rotated: []f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, q: Mse) Allocator.Error!Workspace {
        const buf = try allocator.alloc(f32, 2 * @as(usize, q.padded));
        return .{
            .staging = buf[0..q.padded],
            .rotated = buf[q.padded..],
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Workspace) void {
        // One allocation, split in two.
        self.allocator.free(self.staging.ptr[0 .. self.staging.len + self.rotated.len]);
        self.* = undefined;
    }
};

pub const Mse = struct {
    dim: u32,
    /// Working dimension: `dim` rounded up to a power of two. One code is stored per
    /// padded coordinate, so a poorly-sized `dim` costs bits.
    padded: u32,
    bits: u6,
    rotation: Rotation,
    codebook: Codebook,

    pub fn init(allocator: Allocator, params: Params) !Mse {
        std.debug.assert(params.dim >= 1);
        std.debug.assert(params.bits >= 1);

        var rotation = try Rotation.init(
            allocator,
            params.dim,
            params.rotation,
            params.seed,
            .rht_signs,
        );
        errdefer rotation.deinit();

        // The density is parameterized by the *padded* dimension: after rotation the
        // vector is uniform on the sphere of that dimension, zero-padding included.
        const density = if (params.exact_density)
            Density.sphereCoord(rotation.padded)
        else
            Density.gauss(1.0 / @sqrt(@as(f64, @floatFromInt(rotation.padded))));

        var codebook = try Codebook.init(allocator, density, params.bits);
        errdefer codebook.deinit();

        return .{
            .dim = params.dim,
            .padded = rotation.padded,
            .bits = params.bits,
            .rotation = rotation,
            .codebook = codebook,
        };
    }

    pub fn deinit(self: *Mse) void {
        self.rotation.deinit();
        self.codebook.deinit();
        self.* = undefined;
    }

    /// Codes produced per vector.
    pub fn codeLen(self: Mse) usize {
        return self.padded;
    }

    /// Encode `x`, returning its Euclidean norm.
    ///
    /// The norm is stored separately because the algorithm quantizes directions: the
    /// coordinate density is that of a point on the *unit* sphere.
    pub fn encode(self: Mse, x: []const f32, codes: []u8, ws: *Workspace) f32 {
        std.debug.assert(x.len == self.dim);
        std.debug.assert(codes.len == self.codeLen());

        const norm = self.encodeRotated(x, ws.rotated, ws.staging);
        self.codebook.encodeSlice(ws.rotated, codes);
        return norm;
    }

    /// Reconstruct into `out`, which is `dim` long.
    pub fn decode(self: Mse, codes: []const u8, norm: f32, out: []f32, ws: *Workspace) void {
        std.debug.assert(codes.len == self.codeLen());
        std.debug.assert(out.len == self.dim);

        self.decodeRotated(codes, ws.rotated);
        for (ws.rotated) |*v| v.* *= norm;
        self.rotation.applyInverse(ws.rotated, ws.staging);
        @memcpy(out, ws.staging[0..self.dim]);
    }

    /// `ỹ` — the reconstruction in the rotated basis, before scaling by the norm.
    ///
    /// Exposed because `prod` forms its residual here rather than in the original
    /// basis: staying rotated is what lets a search rotate only the query
    /// (docs/DESIGN.md §1.3).
    pub fn decodeRotated(self: Mse, codes: []const u8, out: []f32) void {
        std.debug.assert(codes.len == self.codeLen());
        std.debug.assert(out.len == self.padded);
        self.codebook.decodeSlice(codes, out);
    }

    /// `y` — the unit-norm rotated vector, written to `out`. Returns ‖x‖.
    ///
    /// `staging` must be `padded` long and disjoint from `out`.
    pub fn encodeRotated(self: Mse, x: []const f32, out: []f32, staging: []f32) f32 {
        std.debug.assert(x.len == self.dim);
        std.debug.assert(out.len == self.padded);
        std.debug.assert(staging.len == self.padded);

        const norm = euclideanNorm(x);
        if (norm == 0) {
            // No direction to preserve. Encodes to the zero vector, which decodes
            // back to zero once scaled by the zero norm.
            @memset(out, 0);
            return 0;
        }
        const inverse = 1.0 / norm;
        for (staging[0..self.dim], x) |*dst, v| dst.* = v * inverse;
        self.rotation.apply(staging[0..self.dim], out);
        return norm;
    }
};

pub fn euclideanNorm(x: []const f32) f32 {
    var sum: f64 = 0;
    for (x) |v| sum += @as(f64, v) * v;
    return @floatCast(@sqrt(sum));
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;

fn randomUnit(buf: []f32, stream: *rng.Stream) void {
    var norm: f64 = 0;
    for (buf) |*v| {
        const g = stream.nextGaussian();
        v.* = @floatCast(g);
        norm += g * g;
    }
    const inv: f32 = @floatCast(1.0 / @sqrt(norm));
    for (buf) |*v| v.* *= inv;
}

/// Measured E‖x − x̃‖² over random unit vectors: the paper's D_mse.
fn measureDistortion(q: Mse, trials: usize, seed: u64) !f64 {
    var ws = try Workspace.init(testing.allocator, q);
    defer ws.deinit();

    const x = try testing.allocator.alloc(f32, q.dim);
    defer testing.allocator.free(x);
    const back = try testing.allocator.alloc(f32, q.dim);
    defer testing.allocator.free(back);
    const codes = try testing.allocator.alloc(u8, q.codeLen());
    defer testing.allocator.free(codes);

    var stream = rng.Philox.init(seed).stream(.testing);
    var total: f64 = 0;
    for (0..trials) |_| {
        randomUnit(x, &stream);
        const norm = q.encode(x, codes, &ws);
        q.decode(codes, norm, back, &ws);
        for (x, back) |a, b| {
            const diff = @as(f64, a) - @as(f64, b);
            total += diff * diff;
        }
    }
    return total / @as(f64, @floatFromInt(trials));
}

test "measured distortion matches the paper's values" {
    // Paper, Theorem 1: D_mse ≈ 0.36, 0.117, 0.03, 0.009 for b = 1..4. This is the
    // end-to-end check — rotation, codebook, encode, and decode together — as
    // opposed to the solver test, which only checks the codebook in isolation.
    const expected = [_]f64{ 0.36, 0.117, 0.03, 0.009 };
    for (expected, 1..) |want, bits| {
        var q = try Mse.init(testing.allocator, .{
            .dim = 512,
            .bits = @intCast(bits),
            .seed = 0x5EED,
        });
        defer q.deinit();

        const measured = try measureDistortion(q, 300, 0xA11CE);
        try testing.expectApproxEqAbs(want, measured, 0.02);
    }
}

test "measured distortion respects the Panter-Dite bound" {
    const bound_constant = @sqrt(3.0) * std.math.pi / 2.0;
    for (1..6) |bits| {
        var q = try Mse.init(testing.allocator, .{ .dim = 256, .bits = @intCast(bits) });
        defer q.deinit();
        const measured = try measureDistortion(q, 200, 0xB0B);
        const bound = bound_constant / std.math.pow(f64, 4.0, @floatFromInt(bits));
        try testing.expect(measured <= bound);
    }
}

test "the dense reference and the hadamard fast path distort equally" {
    // The §7.2 gate: the RHT is an approximation of the paper's Haar rotation, and
    // is only acceptable if it costs nothing measurable in distortion.
    for ([_]u6{ 2, 4 }) |bits| {
        var fast = try Mse.init(testing.allocator, .{
            .dim = 256,
            .bits = bits,
            .seed = 0x11,
            .rotation = .hadamard,
        });
        defer fast.deinit();
        var reference = try Mse.init(testing.allocator, .{
            .dim = 256,
            .bits = bits,
            .seed = 0x11,
            .rotation = .dense,
        });
        defer reference.deinit();

        const fast_distortion = try measureDistortion(fast, 200, 0xCAFE);
        const reference_distortion = try measureDistortion(reference, 200, 0xCAFE);
        try testing.expectApproxEqRel(reference_distortion, fast_distortion, 0.06);
    }
}

test "norm is preserved and reconstruction is scale-equivariant" {
    var q = try Mse.init(testing.allocator, .{ .dim = 64, .bits = 4 });
    defer q.deinit();
    var ws = try Workspace.init(testing.allocator, q);
    defer ws.deinit();

    var stream = rng.Philox.init(5).stream(.testing);
    var x: [64]f32 = undefined;
    randomUnit(&x, &stream);

    var codes: [64]u8 = undefined;
    var codes_scaled: [64]u8 = undefined;
    var scaled: [64]f32 = undefined;
    for (&scaled, x) |*d, v| d.* = v * 37.5;

    const n1 = q.encode(&x, &codes, &ws);
    const n2 = q.encode(&scaled, &codes_scaled, &ws);

    try testing.expectApproxEqRel(@as(f32, 1.0), n1, 1e-5);
    try testing.expectApproxEqRel(@as(f32, 37.5), n2, 1e-5);
    // Direction is all that is quantized, so scaling must not change the codes.
    try testing.expectEqualSlices(u8, &codes, &codes_scaled);
}

test "zero vectors round trip" {
    var q = try Mse.init(testing.allocator, .{ .dim = 32, .bits = 3 });
    defer q.deinit();
    var ws = try Workspace.init(testing.allocator, q);
    defer ws.deinit();

    const x = [_]f32{0} ** 32;
    var codes: [32]u8 = undefined;
    var back: [32]f32 = undefined;

    const norm = q.encode(&x, &codes, &ws);
    try testing.expectEqual(@as(f32, 0), norm);
    q.decode(&codes, norm, &back, &ws);
    for (back) |v| try testing.expectEqual(@as(f32, 0), v);
}

test "non-power-of-two dimensions work end to end" {
    for ([_]u32{ 200, 300, 768, 1000 }) |dim| {
        var q = try Mse.init(testing.allocator, .{ .dim = dim, .bits = 4, .seed = 3 });
        defer q.deinit();
        try testing.expect(q.padded >= dim);
        try testing.expect(std.math.isPowerOfTwo(q.padded));

        const measured = try measureDistortion(q, 60, 0xD1);
        // Padding costs bits, not accuracy: the reconstruction is still good.
        try testing.expect(measured < 0.02);
    }
}

test "encoding is deterministic across instances with the same seed" {
    // The cross-language conformance property (docs/DESIGN.md §7.4) in-process.
    var stream = rng.Philox.init(77).stream(.testing);
    var x: [128]f32 = undefined;
    randomUnit(&x, &stream);

    var codes_a: [128]u8 = undefined;
    var codes_b: [128]u8 = undefined;

    var a = try Mse.init(testing.allocator, .{ .dim = 128, .bits = 4, .seed = 0xABCDEF });
    defer a.deinit();
    var wsa = try Workspace.init(testing.allocator, a);
    defer wsa.deinit();
    _ = a.encode(&x, &codes_a, &wsa);

    var b = try Mse.init(testing.allocator, .{ .dim = 128, .bits = 4, .seed = 0xABCDEF });
    defer b.deinit();
    var wsb = try Workspace.init(testing.allocator, b);
    defer wsb.deinit();
    _ = b.encode(&x, &codes_b, &wsb);

    try testing.expectEqualSlices(u8, &codes_a, &codes_b);
}

test "different seeds give different codes" {
    var stream = rng.Philox.init(78).stream(.testing);
    var x: [128]f32 = undefined;
    randomUnit(&x, &stream);

    var codes_a: [128]u8 = undefined;
    var codes_b: [128]u8 = undefined;

    var a = try Mse.init(testing.allocator, .{ .dim = 128, .bits = 4, .seed = 1 });
    defer a.deinit();
    var wsa = try Workspace.init(testing.allocator, a);
    defer wsa.deinit();
    _ = a.encode(&x, &codes_a, &wsa);

    var b = try Mse.init(testing.allocator, .{ .dim = 128, .bits = 4, .seed = 2 });
    defer b.deinit();
    var wsb = try Workspace.init(testing.allocator, b);
    defer wsb.deinit();
    _ = b.encode(&x, &codes_b, &wsb);

    try testing.expect(!std.mem.eql(u8, &codes_a, &codes_b));
}

test "exact density beats the gaussian limit at KV dimensions" {
    // Quantifies what §8.2 is protecting against. At d=128 the exact codebook should
    // be at least as good as the Gaussian-limit one; the point is that the choice is
    // measurable rather than cosmetic.
    var exact = try Mse.init(testing.allocator, .{
        .dim = 128,
        .bits = 4,
        .seed = 0x9,
        .exact_density = true,
    });
    defer exact.deinit();
    var limit = try Mse.init(testing.allocator, .{
        .dim = 128,
        .bits = 4,
        .seed = 0x9,
        .exact_density = false,
    });
    defer limit.deinit();

    // The codebooks must actually differ, or the comparison is vacuous.
    try testing.expect(!std.mem.eql(f32, exact.codebook.centroids, limit.codebook.centroids));

    const exact_distortion = try measureDistortion(exact, 400, 0x1CE);
    const limit_distortion = try measureDistortion(limit, 400, 0x1CE);
    try testing.expect(exact_distortion <= limit_distortion * 1.001);
}
