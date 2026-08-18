//! TurboQuant_prod — Algorithm 2 of the paper: an *unbiased* inner-product quantizer.
//!
//! MSE-optimal quantizers are biased for inner products (at b=1 the bias is exactly
//! 2/π). The fix spends b−1 bits on the MSE stage and the last bit on a 1-bit QJL
//! sketch of the residual, which corrects the bias in expectation.
//!
//!     Quant(x):  idx = Quant_mse(x);  r = x − DeQuant_mse(idx)
//!                qjl = sign(S·r);     γ = ‖r‖₂
//!
//! Everything here works in the rotated basis (docs/DESIGN.md §1.3). Because
//! `S' = S·Πᵀ` has the same distribution as `S`, the residual can be formed as
//! `u = y − ỹ` without ever rotating back, and a query is rotated exactly once:
//!
//!     ⟨q, x̃⟩ = ‖x‖ · [ ⟨p, ỹ⟩ + γ·scale·⟨S'p, qjl⟩ ],   p = Π·q
//!
//! ## Deviation: an orthogonal sketch, with an exact constant
//!
//! The paper draws S with i.i.d. N(0,1) entries and normalizes by √(π/2)/d. We use
//! an orthogonal S' (a second `Rotation`), for the same O(d log d) and
//! nothing-to-store reasons as Π itself.
//!
//! That change permits a *better* constant. The rows of a Haar-orthogonal matrix are
//! uniform unit vectors, and for such a row r and unit x,
//!
//!     E[ sign(rᵀx) · (rᵀy) ] = c_m · ⟨x, y⟩,    c_m = E|rᵀx|
//!
//! exactly — decompose y into its component along x plus a perpendicular part, whose
//! contribution vanishes by symmetry. And `c_m` is precisely the mean absolute value
//! of the sphere-coordinate density, which `density.moment` already gives in closed
//! form. So the estimator is unbiased at finite m, whereas √(π/2)/√m is only its
//! large-m limit. That matters at the dimensions the KV path uses.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Density = @import("../math/density.zig").Density;
const Rotation = @import("../math/rotation.zig").Rotation;
const RotationKind = @import("../math/rotation.zig").Kind;
const rng = @import("../math/rng.zig");
const mse_mod = @import("mse.zig");
const Mse = mse_mod.Mse;

pub const Params = struct {
    dim: u32,
    /// Total budget per coordinate: `bits − 1` code bits plus the 1 sketch bit.
    /// `bits = 1` means no MSE stage at all — the whole budget is the sketch.
    bits: u6,
    seed: u64 = 0,
    rotation: RotationKind = .hadamard,
    exact_density: bool = true,
};

/// Per-vector outputs that are not the code arrays.
pub const Scalars = struct {
    /// ‖x‖.
    norm: f32,
    /// ‖y − ỹ‖: the residual norm, in the rotated basis.
    gamma: f32,
};

/// Query-side state. Built once per query, then reused across the whole corpus —
/// this is the entire point of staying in the rotated basis.
pub const QueryState = struct {
    /// p = Π·q
    rotated: []f32,
    /// S'·p
    sketched: []f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, q: Prod) Allocator.Error!QueryState {
        const buf = try allocator.alloc(f32, 2 * @as(usize, q.padded()));
        return .{
            .rotated = buf[0..q.padded()],
            .sketched = buf[q.padded()..],
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *QueryState) void {
        self.allocator.free(self.rotated.ptr[0 .. self.rotated.len + self.sketched.len]);
        self.* = undefined;
    }
};

pub const Workspace = struct {
    staging: []f32,
    rotated: []f32,
    reconstruction: []f32,
    residual: []f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, q: Prod) Allocator.Error!Workspace {
        const n = q.padded();
        const buf = try allocator.alloc(f32, 4 * @as(usize, n));
        return .{
            .staging = buf[0..n],
            .rotated = buf[n .. 2 * n],
            .reconstruction = buf[2 * n .. 3 * n],
            .residual = buf[3 * n .. 4 * n],
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Workspace) void {
        self.allocator.free(self.staging.ptr[0 .. 4 * self.staging.len]);
        self.* = undefined;
    }
};

pub const Prod = struct {
    bits: u6,
    mse: Mse,
    sketch: Rotation,
    /// 1 / (m · c_m). Folded into one constant so the hot path multiplies once.
    sketch_scale: f32,

    pub fn init(allocator: Allocator, params: Params) !Prod {
        std.debug.assert(params.bits >= 1);

        // The MSE stage gets one fewer bit; the last one buys the sketch.
        var mse = try Mse.init(allocator, .{
            .dim = params.dim,
            .bits = params.bits - 1,
            .seed = params.seed,
            .rotation = params.rotation,
            .exact_density = params.exact_density,
        });
        errdefer mse.deinit();

        // A different RNG purpose, so S' is independent of Π under the same seed.
        var sketch = try Rotation.init(
            allocator,
            mse.padded,
            params.rotation,
            params.seed,
            .sketch_signs,
        );
        errdefer sketch.deinit();

        const m = mse.padded;
        // c_m = E|rᵀx| for a uniform unit row r and unit x, which is the mean
        // absolute value of the sphere-coordinate density: 2·∫₀¹ t·f(t) dt.
        const mean_abs = 2.0 * Density.sphereCoord(m).moment(0.0, 1.0);
        const scale = 1.0 / (@as(f64, @floatFromInt(m)) * mean_abs);

        return .{
            .bits = params.bits,
            .mse = mse,
            .sketch = sketch,
            .sketch_scale = @floatCast(scale),
        };
    }

    pub fn deinit(self: *Prod) void {
        self.mse.deinit();
        self.sketch.deinit();
        self.* = undefined;
    }

    pub fn dim(self: Prod) u32 {
        return self.mse.dim;
    }

    pub fn padded(self: Prod) u32 {
        return self.mse.padded;
    }

    /// Codes per vector, one byte each (bit-packing is a separate concern).
    pub fn codeLen(self: Prod) usize {
        return self.mse.codeLen();
    }

    /// Bytes of sketch per vector: one bit per padded coordinate.
    pub fn sketchLen(self: Prod) usize {
        return (@as(usize, self.padded()) + 7) / 8;
    }

    pub fn encode(
        self: Prod,
        x: []const f32,
        codes: []u8,
        sketch: []u8,
        ws: *Workspace,
    ) Scalars {
        std.debug.assert(x.len == self.dim());
        std.debug.assert(codes.len == self.codeLen());
        std.debug.assert(sketch.len == self.sketchLen());

        // y = Π·(x/‖x‖)
        const norm = self.mse.encodeRotated(x, ws.rotated, ws.staging);

        // idx, then ỹ, both in the rotated basis.
        self.mse.codebook.encodeSlice(ws.rotated, codes);
        self.mse.decodeRotated(codes, ws.reconstruction);

        // u = y − ỹ
        var sum: f64 = 0;
        for (ws.residual, ws.rotated, ws.reconstruction) |*u, y, y_hat| {
            u.* = y - y_hat;
            sum += @as(f64, u.*) * u.*;
        }
        const gamma: f32 = @floatCast(@sqrt(sum));

        // qjl = sign(S'·u). Scale-invariant, so u need not be normalized first.
        self.sketch.apply(ws.residual, ws.staging);
        packSigns(ws.staging, sketch);

        return .{ .norm = norm, .gamma = gamma };
    }

    /// Full reconstruction x̃. Not on the search path — `dot` avoids it entirely —
    /// but needed for MSE-style use and for testing the estimator against it.
    pub fn decode(
        self: Prod,
        codes: []const u8,
        sketch: []const u8,
        scalars: Scalars,
        out: []f32,
        ws: *Workspace,
    ) void {
        std.debug.assert(out.len == self.dim());

        self.mse.decodeRotated(codes, ws.reconstruction);

        // u_est = γ·scale·S'ᵀ·qjl
        unpackSigns(sketch, ws.residual);
        self.sketch.applyInverse(ws.residual, ws.staging);

        const weight = scalars.gamma * self.sketch_scale;
        for (ws.rotated, ws.reconstruction, ws.staging) |*dst, y_hat, correction| {
            dst.* = (y_hat + weight * correction) * scalars.norm;
        }

        self.mse.rotation.applyInverse(ws.rotated, ws.staging);
        @memcpy(out, ws.staging[0..self.dim()]);
    }

    /// Rotate and sketch a query once, for reuse across the corpus.
    pub fn prepareQuery(self: Prod, q: []const f32, state: *QueryState, ws: *Workspace) void {
        std.debug.assert(q.len == self.dim());
        // No normalization: the estimator is linear in q, so an unnormalized query
        // yields the true unnormalized inner product.
        @memcpy(ws.staging[0..self.dim()], q);
        self.mse.rotation.apply(ws.staging[0..self.dim()], state.rotated);
        self.sketch.apply(state.rotated, state.sketched);
    }

    /// Estimate ⟨q, x⟩ without reconstructing anything.
    ///
    ///     ‖x‖ · [ ⟨p, ỹ⟩ + γ·scale·⟨S'p, qjl⟩ ]
    ///
    /// The reference form of the §4.2 scan kernel: the first term is a per-coordinate
    /// gather from a tiny codebook, the second a sign-flip-and-sum.
    pub fn dot(
        self: Prod,
        state: QueryState,
        codes: []const u8,
        sketch: []const u8,
        scalars: Scalars,
    ) f32 {
        const centroids = self.mse.codebook.centroids;

        var mse_term: f64 = 0;
        for (codes, state.rotated) |code, p| {
            mse_term += @as(f64, p) * centroids[code];
        }

        var sketch_term: f64 = 0;
        for (state.sketched, 0..) |w, i| {
            const bit = (sketch[i >> 3] >> @intCast(i & 7)) & 1;
            sketch_term += if (bit == 1) @as(f64, w) else -@as(f64, w);
        }

        const total = mse_term + @as(f64, scalars.gamma) * self.sketch_scale * sketch_term;
        return @floatCast(total * scalars.norm);
    }
};

fn packSigns(values: []const f32, bits: []u8) void {
    @memset(bits, 0);
    for (values, 0..) |v, i| {
        // Zero maps to +1. It has probability zero for continuous inputs, and any
        // consistent choice is unbiased.
        if (v >= 0) bits[i >> 3] |= @as(u8, 1) << @intCast(i & 7);
    }
}

fn unpackSigns(bits: []const u8, values: []f32) void {
    for (values, 0..) |*v, i| {
        const bit = (bits[i >> 3] >> @intCast(i & 7)) & 1;
        v.* = if (bit == 1) 1.0 else -1.0;
    }
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

fn exactDot(a: []const f32, b: []const f32) f64 {
    var sum: f64 = 0;
    for (a, b) |x, y| sum += @as(f64, x) * y;
    return sum;
}

const Measurement = struct {
    /// Least-squares slope of estimate against truth. Unbiased means 1.
    ///
    /// A slope rather than a mean of per-trial ratios: for random unit vectors in
    /// dimension d the true inner product is about ±1/√d, so individual ratios have
    /// a near-zero denominator and their mean is numerically worthless. The first
    /// version of this test measured 0.19 where the answer was 0.64 for exactly
    /// that reason.
    slope: f64,
    /// E[(⟨q,x̃⟩ − ⟨q,x⟩)²]: the paper's D_prod.
    distortion: f64,
};

fn measure(q: Prod, trials: usize, seed: u64) !Measurement {
    var ws = try Workspace.init(testing.allocator, q);
    defer ws.deinit();
    var state = try QueryState.init(testing.allocator, q);
    defer state.deinit();

    const x = try testing.allocator.alloc(f32, q.dim());
    defer testing.allocator.free(x);
    const query = try testing.allocator.alloc(f32, q.dim());
    defer testing.allocator.free(query);
    const codes = try testing.allocator.alloc(u8, q.codeLen());
    defer testing.allocator.free(codes);
    const sketch = try testing.allocator.alloc(u8, q.sketchLen());
    defer testing.allocator.free(sketch);

    var stream = rng.Philox.init(seed).stream(.testing);
    var squared: f64 = 0;
    var cross: f64 = 0;
    var truth_squared: f64 = 0;

    for (0..trials) |_| {
        randomUnit(x, &stream);
        randomUnit(query, &stream);

        const scalars = q.encode(x, codes, sketch, &ws);
        q.prepareQuery(query, &state, &ws);
        const estimate: f64 = q.dot(state, codes, sketch, scalars);
        const truth = exactDot(query, x);

        const err = estimate - truth;
        squared += err * err;
        cross += estimate * truth;
        truth_squared += truth * truth;
    }
    return .{
        .slope = cross / truth_squared,
        .distortion = squared / @as(f64, @floatFromInt(trials)),
    };
}

/// The paper's construction verbatim: an i.i.d. N(0,1) sketch matrix with the
/// √(π/2)/m normalization, at one bit (no MSE stage).
///
/// Present so the orthogonal sketch has something to be compared against that is
/// known-correct, and so a claim about beating the paper rests on running the
/// paper's own method in this harness rather than on quoting its numbers.
fn measureGaussianSketch(
    comptime m: u32,
    trials: usize,
    seed: u64,
) !f64 {
    var gaussian = rng.Philox.init(seed).stream(.dense_rotation);
    const S = try testing.allocator.alloc(f32, @as(usize, m) * m);
    defer testing.allocator.free(S);
    for (S) |*v| v.* = @floatCast(gaussian.nextGaussian());

    var stream = rng.Philox.init(seed ^ 0x9E37).stream(.testing);
    var x: [m]f32 = undefined;
    var query: [m]f32 = undefined;
    const scale = @sqrt(std.math.pi / 2.0) / @as(f64, m);

    var squared: f64 = 0;
    for (0..trials) |_| {
        randomUnit(&x, &stream);
        randomUnit(&query, &stream);

        var acc: f64 = 0;
        for (0..m) |i| {
            const row = S[i * m ..][0..m];
            var row_query: f64 = 0;
            var row_x: f64 = 0;
            for (row, query, x) |sv, qv, xv| {
                row_query += @as(f64, sv) * qv;
                row_x += @as(f64, sv) * xv;
            }
            acc += if (row_x >= 0) row_query else -row_query;
        }
        const err = scale * acc - exactDot(&query, &x);
        squared += err * err;
    }
    return squared / @as(f64, @floatFromInt(trials));
}

test "inner product estimate is unbiased" {
    // The entire reason TurboQuant_prod exists. An MSE quantizer alone has a
    // multiplicative bias here (2/π at one bit); the QJL residual removes it.
    const trials = 6000;
    for ([_]u6{ 1, 2, 3, 4 }) |bits| {
        var q = try Prod.init(testing.allocator, .{
            .dim = 256,
            .bits = bits,
            .seed = 0x5EED,
        });
        defer q.deinit();

        const m = try measure(q, trials, 0xA11CE);
        try testing.expectApproxEqAbs(@as(f64, 1.0), m.slope, 0.03);
    }
}

test "the paper's gaussian sketch reproduces the paper's distortion" {
    // Validates the measurement harness against a published number before the
    // harness is used to claim anything. Paper, Theorem 2: D_prod ≈ 1.57/d at b=1,
    // and 1.57 ≈ π/2 is the variance of the Gaussian-sketch estimator.
    const m: u32 = 128;
    const distortion = try measureGaussianSketch(m, 1500, 0x5151);
    try testing.expectApproxEqRel(
        std.math.pi / 2.0,
        distortion * @as(f64, m),
        0.12,
    );
}

test "the orthogonal sketch beats the paper's gaussian sketch" {
    // An orthogonal S' is not merely a cheaper stand-in for a Gaussian one: it is
    // measurably more accurate. Orthonormal rows fix Σ(rᵢᵀq)² = ‖q‖² exactly instead
    // of letting it fluctuate, and that removed variance shows up directly in the
    // estimator. Measured at roughly 2.7× lower distortion, consistently across
    // bit-widths.
    const m: u32 = 128;
    const gaussian = try measureGaussianSketch(m, 1500, 0x5151);

    var q = try Prod.init(testing.allocator, .{ .dim = m, .bits = 1, .seed = 0x5151 });
    defer q.deinit();
    const ours = try measure(q, 4000, 0xBEEF);

    try testing.expect(ours.distortion < gaussian);
    try testing.expect(gaussian / ours.distortion > 2.0);
}

test "inner product distortion by bit-width" {
    // Regression pins on measured values. These are ours, not the paper's: the
    // orthogonal sketch lands about 2.7× below the published Gaussian-sketch
    // figures {1.57, 0.56, 0.18, 0.047}/d, so asserting those would fail for the
    // right reason. Each value must also stay under the paper's, which is the
    // claim that matters.
    const dim: u32 = 256;
    const paper = [_]f64{ 1.57, 0.56, 0.18, 0.047 };
    const measured = [_]f64{ 0.567, 0.207, 0.068, 0.020 };

    var previous: f64 = std.math.inf(f64);
    for (paper, measured, 1..) |published, want, bits| {
        var q = try Prod.init(testing.allocator, .{
            .dim = dim,
            .bits = @intCast(bits),
            .seed = 0x1234,
        });
        defer q.deinit();

        const m = try measure(q, 4000, 0xBEEF);
        const scaled = m.distortion * @as(f64, @floatFromInt(q.padded()));

        try testing.expectApproxEqRel(want, scaled, 0.15);
        try testing.expect(scaled < published);
        try testing.expect(scaled < previous); // strictly improving with more bits
        previous = scaled;
    }
}

test "the mse-only estimator is biased where prod is not" {
    // Demonstrates the problem being solved rather than asserting it. At one MSE bit
    // the paper predicts a multiplicative bias of 2/π ≈ 0.6366 in ⟨q, x̃_mse⟩.
    const dim: u32 = 512;
    var q = try Prod.init(testing.allocator, .{ .dim = dim, .bits = 2, .seed = 7 });
    defer q.deinit();

    var ws = try Workspace.init(testing.allocator, q);
    defer ws.deinit();
    var state = try QueryState.init(testing.allocator, q);
    defer state.deinit();

    const x = try testing.allocator.alloc(f32, dim);
    defer testing.allocator.free(x);
    const query = try testing.allocator.alloc(f32, dim);
    defer testing.allocator.free(query);
    const codes = try testing.allocator.alloc(u8, q.codeLen());
    defer testing.allocator.free(codes);
    const sketch = try testing.allocator.alloc(u8, q.sketchLen());
    defer testing.allocator.free(sketch);

    var stream = rng.Philox.init(0xDEAD).stream(.testing);
    var cross: f64 = 0;
    var truth_squared: f64 = 0;
    const trials = 4000;
    for (0..trials) |_| {
        randomUnit(x, &stream);
        randomUnit(query, &stream);
        const scalars = q.encode(x, codes, sketch, &ws);
        q.prepareQuery(query, &state, &ws);

        // Just the MSE half of the estimator.
        var mse_term: f64 = 0;
        for (codes, state.rotated) |code, p| {
            mse_term += @as(f64, p) * q.mse.codebook.centroids[code];
        }
        const truth = exactDot(query, x);
        cross += mse_term * scalars.norm * truth;
        truth_squared += truth * truth;
    }
    try testing.expectApproxEqAbs(2.0 / std.math.pi, cross / truth_squared, 0.03);
}

test "the exact constant beats the asymptotic one at small dimensions" {
    // Justifies deriving c_m from the density instead of using √(π/2)/√m. At small m
    // the two differ, and only the exact one is unbiased.
    for ([_]u32{ 64, 128 }) |m| {
        const mean_abs = 2.0 * Density.sphereCoord(m).moment(0.0, 1.0);
        const exact = 1.0 / (@as(f64, @floatFromInt(m)) * mean_abs);
        const asymptotic = @sqrt(std.math.pi / 2.0) / @sqrt(@as(f64, @floatFromInt(m)));
        // They agree to within a couple of percent but are not equal, and the gap
        // grows as m shrinks.
        try testing.expect(@abs(exact - asymptotic) / exact > 0.001);
        try testing.expect(@abs(exact - asymptotic) / exact < 0.05);
    }
}

test "decode agrees with the dot estimator" {
    // `dot` is an algebraic shortcut past `decode`; they must not drift apart.
    const dim: u32 = 128;
    var q = try Prod.init(testing.allocator, .{ .dim = dim, .bits = 4, .seed = 0x33 });
    defer q.deinit();

    var ws = try Workspace.init(testing.allocator, q);
    defer ws.deinit();
    var state = try QueryState.init(testing.allocator, q);
    defer state.deinit();

    var x: [dim]f32 = undefined;
    var query: [dim]f32 = undefined;
    var back: [dim]f32 = undefined;
    const codes = try testing.allocator.alloc(u8, q.codeLen());
    defer testing.allocator.free(codes);
    const sketch = try testing.allocator.alloc(u8, q.sketchLen());
    defer testing.allocator.free(sketch);

    var stream = rng.Philox.init(0x99).stream(.testing);
    for (0..20) |_| {
        randomUnit(&x, &stream);
        randomUnit(&query, &stream);

        const scalars = q.encode(&x, codes, sketch, &ws);
        q.decode(codes, sketch, scalars, &back, &ws);
        q.prepareQuery(&query, &state, &ws);

        const via_dot = q.dot(state, codes, sketch, scalars);
        const via_decode = exactDot(&query, &back);
        try testing.expectApproxEqAbs(via_decode, @as(f64, via_dot), 1e-4);
    }
}

test "estimator is linear in the query" {
    // Confirms no hidden query normalization, which would silently break any caller
    // that cares about unnormalized inner products.
    const dim: u32 = 64;
    var q = try Prod.init(testing.allocator, .{ .dim = dim, .bits = 3, .seed = 1 });
    defer q.deinit();

    var ws = try Workspace.init(testing.allocator, q);
    defer ws.deinit();
    var state = try QueryState.init(testing.allocator, q);
    defer state.deinit();
    var scaled_state = try QueryState.init(testing.allocator, q);
    defer scaled_state.deinit();

    var x: [dim]f32 = undefined;
    var query: [dim]f32 = undefined;
    var scaled: [dim]f32 = undefined;
    const codes = try testing.allocator.alloc(u8, q.codeLen());
    defer testing.allocator.free(codes);
    const sketch = try testing.allocator.alloc(u8, q.sketchLen());
    defer testing.allocator.free(sketch);

    var stream = rng.Philox.init(0x77).stream(.testing);
    randomUnit(&x, &stream);
    randomUnit(&query, &stream);
    for (&scaled, query) |*d, v| d.* = v * -3.25;

    const scalars = q.encode(&x, codes, sketch, &ws);
    q.prepareQuery(&query, &state, &ws);
    q.prepareQuery(&scaled, &scaled_state, &ws);

    const base = q.dot(state, codes, sketch, scalars);
    const got = q.dot(scaled_state, codes, sketch, scalars);
    try testing.expectApproxEqRel(base * -3.25, got, 1e-4);
}

test "hadamard sketch stays unbiased against the dense reference" {
    // The §7.2 gate for S'. The exact-c_m argument is proven for Haar-orthogonal
    // rows; the RHT only approximates them, so the bias it introduces has to be
    // measured rather than assumed away.
    const trials = 6000;
    for ([_]RotationKind{ .dense, .hadamard }) |kind| {
        var q = try Prod.init(testing.allocator, .{
            .dim = 256,
            .bits = 3,
            .seed = 0x4242,
            .rotation = kind,
        });
        defer q.deinit();

        const m = try measure(q, trials, 0xF00D);
        try testing.expectApproxEqAbs(@as(f64, 1.0), m.slope, 0.03);
    }
}

test "bit budget accounting" {
    for ([_]u6{ 1, 2, 4 }) |bits| {
        var q = try Prod.init(testing.allocator, .{ .dim = 128, .bits = bits });
        defer q.deinit();
        // b−1 code bits...
        try testing.expectEqual(bits - 1, q.mse.bits);
        // ...plus exactly one sketch bit per padded coordinate.
        try testing.expectEqual(@as(usize, 128 / 8), q.sketchLen());
    }
}

test "one bit means sketch only" {
    var q = try Prod.init(testing.allocator, .{ .dim = 128, .bits = 1 });
    defer q.deinit();
    try testing.expectEqual(@as(usize, 1), q.mse.codebook.levels());

    var ws = try Workspace.init(testing.allocator, q);
    defer ws.deinit();
    var x: [128]f32 = undefined;
    var stream = rng.Philox.init(2).stream(.testing);
    randomUnit(&x, &stream);

    const codes = try testing.allocator.alloc(u8, q.codeLen());
    defer testing.allocator.free(codes);
    const sketch = try testing.allocator.alloc(u8, q.sketchLen());
    defer testing.allocator.free(sketch);

    const scalars = q.encode(&x, codes, sketch, &ws);
    // With nothing reconstructed, the residual is the whole unit vector.
    try testing.expectApproxEqAbs(@as(f32, 1.0), scalars.gamma, 1e-4);
    for (codes) |c| try testing.expectEqual(@as(u8, 0), c);
}

test "sign packing round trips" {
    var values: [37]f32 = undefined;
    var stream = rng.Philox.init(11).stream(.testing);
    for (&values) |*v| v.* = @floatCast(stream.nextGaussian());

    var bits: [5]u8 = undefined;
    var unpacked: [37]f32 = undefined;
    packSigns(&values, &bits);
    unpackSigns(&bits, &unpacked);

    for (values, unpacked) |original, sign| {
        try testing.expectEqual(@as(f32, if (original >= 0) 1.0 else -1.0), sign);
    }
}

test "encoding is deterministic for a given seed" {
    var stream = rng.Philox.init(123).stream(.testing);
    var x: [256]f32 = undefined;
    randomUnit(&x, &stream);

    var results: [2][]u8 = undefined;
    var sketches: [2][]u8 = undefined;
    defer for (results) |r| testing.allocator.free(r);
    defer for (sketches) |s| testing.allocator.free(s);

    for (0..2) |i| {
        var q = try Prod.init(testing.allocator, .{ .dim = 256, .bits = 4, .seed = 0xFEED });
        defer q.deinit();
        var ws = try Workspace.init(testing.allocator, q);
        defer ws.deinit();

        results[i] = try testing.allocator.alloc(u8, q.codeLen());
        sketches[i] = try testing.allocator.alloc(u8, q.sketchLen());
        _ = q.encode(&x, results[i], sketches[i], &ws);
    }
    try testing.expectEqualSlices(u8, results[0], results[1]);
    try testing.expectEqualSlices(u8, sketches[0], sketches[1]);
}
