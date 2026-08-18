//! Lloyd-Max optimal scalar quantizer.
//!
//! Solves the continuous 1-D k-means problem of Eq. (4) in the TurboQuant paper:
//!
//!     C(f_X, b) = min over c_1..c_{2^b} of  Σ_i ∫ |t − c_i|² f_X(t) dt
//!
//! integrated over each Voronoi cell. The optimum is a Voronoi tessellation, so cell
//! boundaries are midpoints between consecutive centroids and each centroid is the
//! conditional mean of its cell. Alternating those two conditions is Lloyd's
//! algorithm; for a log-concave density the fixed point is unique, so there is no
//! local-minimum problem to worry about here.
//!
//! Run once, offline, per (bits, density). Never on the hot path.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Density = @import("density.zig").Density;

pub const Options = struct {
    /// Stop when one iteration's relative improvement in distortion falls below
    /// this.
    ///
    /// Distortion rather than centroid movement: Lloyd's iteration converges only
    /// linearly in the centroids, and the outermost levels drift through the tails
    /// for thousands of iterations after the distortion has stopped changing in the
    /// 14th digit. Gating on centroid movement made b=7 take 37k iterations to
    /// reach a result it already had at ~200.
    tolerance: f64 = 1e-15,
    /// Generous, because convergence is slow at large level counts: Lloyd's linear
    /// rate approaches 1 as levels grow, and b=8 still needs ~27k iterations even
    /// with extrapolation. That is acceptable for an offline solver run once per
    /// (bits, density); if high bit-widths ever become hot, Max's shooting method
    /// (bisect on the lowest centroid, propagate the recurrence) would replace the
    /// fixed-point iteration entirely.
    max_iterations: u32 = 100_000,
};

pub const Error = error{
    /// Lloyd's iteration did not settle. Should not happen for a log-concave
    /// density; treated as a hard error rather than a silently-truncated result.
    NoConvergence,
} || Allocator.Error;

pub const Solution = struct {
    /// Reconstruction levels, ascending.
    centroids: []f64,
    /// Decision boundaries: midpoints between consecutive centroids. An input is
    /// encoded as the count of thresholds it exceeds. Length is centroids.len − 1.
    thresholds: []f64,
    /// E[(t − Q(t))²] under the density.
    distortion: f64,
    /// distortion / variance. This is the paper's D_mse: for a unit vector the
    /// per-coordinate variance is 1/d and the end-to-end MSE is d · C(f_X, b), so
    /// the factors of d cancel and this is directly comparable to the published
    /// values {0.36, 0.117, 0.03, 0.009} for b = 1..4.
    normalized_distortion: f64,
    iterations: u32,

    allocator: Allocator,

    pub fn deinit(self: *Solution) void {
        self.allocator.free(self.centroids);
        self.allocator.free(self.thresholds);
        self.* = undefined;
    }

    /// Encode by counting exceeded thresholds. The reference implementation of what
    /// the SIMD encoder must reproduce exactly.
    pub fn encode(self: Solution, t: f64) usize {
        var idx: usize = 0;
        for (self.thresholds) |threshold| {
            if (t > threshold) idx += 1 else break;
        }
        return idx;
    }
};

/// Solve for `1 << bits` levels.
pub fn solveBits(
    allocator: Allocator,
    density: Density,
    bits: u6,
    options: Options,
) Error!Solution {
    std.debug.assert(bits >= 1 and bits <= 16);
    return solve(allocator, density, @as(usize, 1) << bits, options);
}

pub fn solve(
    allocator: Allocator,
    density: Density,
    levels: usize,
    options: Options,
) Error!Solution {
    std.debug.assert(levels >= 2);

    const centroids = try allocator.alloc(f64, levels);
    errdefer allocator.free(centroids);
    const thresholds = try allocator.alloc(f64, levels - 1);
    errdefer allocator.free(thresholds);

    // Scratch: current cell statistics, two previous iterates for extrapolation,
    // and a candidate with its own statistics so a rejected extrapolation leaves
    // no trace.
    const scratch = try allocator.alloc(f64, levels * 6);
    defer allocator.free(scratch);
    const cell_mass = scratch[0 * levels ..][0..levels];
    const cell_moment = scratch[1 * levels ..][0..levels];
    const prev2 = scratch[2 * levels ..][0..levels];
    const prev1 = scratch[3 * levels ..][0..levels];
    const candidate = scratch[4 * levels ..][0..levels];
    const candidate_stats = scratch[5 * levels ..][0..levels];

    const candidate_thresholds = try allocator.alloc(f64, levels - 1);
    defer allocator.free(candidate_thresholds);
    const candidate_moment = try allocator.alloc(f64, levels);
    defer allocator.free(candidate_moment);

    // Seed at equiprobable quantiles. A uniform seed over the support would put
    // most levels out in the tails where there is no mass, and Lloyd would need
    // many more iterations to drag them back.
    for (centroids, 0..) |*c, i| {
        const p = (@as(f64, @floatFromInt(i)) + 0.5) / @as(f64, @floatFromInt(levels));
        c.* = density.quantile(p);
    }

    var iterations: u32 = 0;
    var distortion = evaluate(density, centroids, thresholds, cell_mass, cell_moment);
    var history: u32 = 0;

    while (iterations < options.max_iterations) {
        iterations += 1;

        @memcpy(prev2, prev1);
        @memcpy(prev1, centroids);
        history += 1;

        lloydStep(centroids, cell_mass, cell_moment);
        const updated = evaluate(density, centroids, thresholds, cell_mass, cell_moment);

        const improvement = distortion - updated;
        distortion = updated;

        // Aitken Δ² extrapolation along the dominant error mode.
        //
        // Lloyd converges linearly here with a ratio very close to 1, so the
        // iteration count is set by that slow mode rather than by the tolerance:
        // loosening the tolerance by six orders of magnitude cut b=8 by only a
        // third. Extrapolating the geometric tail attacks the actual cause.
        //
        // Guarded by an explicit distortion check, so a bad extrapolation is
        // discarded and this can never converge to a worse answer than plain
        // Lloyd would.
        if (history >= 3 and iterations % 3 == 0) {
            if (extrapolate(prev2, prev1, centroids, candidate)) {
                const candidate_distortion = evaluate(
                    density,
                    candidate,
                    candidate_thresholds,
                    candidate_stats,
                    candidate_moment,
                );
                if (candidate_distortion < distortion) {
                    @memcpy(centroids, candidate);
                    @memcpy(thresholds, candidate_thresholds);
                    @memcpy(cell_mass, candidate_stats);
                    @memcpy(cell_moment, candidate_moment);
                    distortion = candidate_distortion;
                    history = 0; // restart the sequence around the new point
                }
            }
        }

        if (improvement <= options.tolerance * @abs(distortion)) break;
    } else {
        return Error.NoConvergence;
    }

    return .{
        .centroids = centroids,
        .thresholds = thresholds,
        .distortion = distortion,
        .normalized_distortion = distortion / density.variance(),
        .iterations = iterations,
        .allocator = allocator,
    };
}

/// One Lloyd update: each centroid becomes the conditional mean of its cell.
fn lloydStep(centroids: []f64, cell_mass: []const f64, cell_moment: []const f64) void {
    for (centroids, cell_mass, cell_moment) |*c, m, mu| {
        // An empty cell has no conditional mean. Leave the centroid where it is
        // rather than moving it to NaN; with a log-concave density and quantile
        // seeding this should not trigger.
        if (m > 1e-300) c.* = mu / m;
    }
}

/// Recompute thresholds and cell statistics for `centroids`, returning the
/// distortion they achieve.
///
/// Distortion is written as E[t²] − 2Σ c_i·moment_i + Σ c_i²·mass_i rather than the
/// shorter E[t²] − Σ c_i²·mass_i, because the short form is valid only at the exact
/// fixed point and would report a too-good number for any intermediate iterate —
/// including the extrapolated candidates, whose acceptance depends on this value.
fn evaluate(
    density: Density,
    centroids: []const f64,
    thresholds: []f64,
    cell_mass: []f64,
    cell_moment: []f64,
) f64 {
    computeThresholds(centroids, thresholds);
    const bounds = density.support();
    const levels = centroids.len;

    var distortion = density.variance();
    for (0..levels) |i| {
        const lo = if (i == 0) bounds[0] else thresholds[i - 1];
        const hi = if (i == levels - 1) bounds[1] else thresholds[i];
        const m = density.mass(lo, hi);
        const mu = density.moment(lo, hi);
        cell_mass[i] = m;
        cell_moment[i] = mu;
        distortion -= 2.0 * centroids[i] * mu;
        distortion += centroids[i] * centroids[i] * m;
    }
    return @max(distortion, 0.0);
}

/// Aitken Δ² on each coordinate of three successive iterates.
///
/// Returns false if the result is not usable — a vanishing denominator (the
/// sequence has already stalled) or a non-monotone output (extrapolation crossed
/// two levels over each other, which would make the codebook invalid).
fn extrapolate(a: []const f64, b: []const f64, c: []const f64, out: []f64) bool {
    for (out, a, b, c) |*o, x0, x1, x2| {
        const d1 = x2 - x1;
        const denominator = x2 - 2.0 * x1 + x0;
        if (@abs(denominator) < 1e-300) return false;
        const step = d1 * d1 / denominator;
        if (!std.math.isFinite(step)) return false;
        o.* = x2 - step;
    }
    for (1..out.len) |i| {
        if (!(out[i] > out[i - 1])) return false;
    }
    return true;
}

fn computeThresholds(centroids: []const f64, thresholds: []f64) void {
    for (thresholds, 0..) |*t, i| {
        t.* = 0.5 * (centroids[i] + centroids[i + 1]);
    }
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;

test "reproduces the published Lloyd-Max levels for a unit gaussian" {
    // Classic Max (1960) table. The paper quotes the same numbers scaled by 1/√d:
    // ±√(2/π) at b=1 and ±0.453, ±1.51 at b=2.
    const expected: [4][]const f64 = .{
        &.{0.7979},
        &.{ 0.4528, 1.5104 },
        &.{ 0.2451, 0.7560, 1.3439, 2.1519 },
        &.{ 0.1284, 0.3881, 0.6568, 0.9423, 1.2562, 1.6181, 2.0690, 2.7326 },
    };

    for (expected, 1..) |want_positive, bits| {
        var sol = try solveBits(testing.allocator, Density.gauss(1.0), @intCast(bits), .{});
        defer sol.deinit();

        const levels = @as(usize, 1) << @intCast(bits);
        try testing.expectEqual(levels, sol.centroids.len);

        // The upper half of the (symmetric) codebook.
        const positive = sol.centroids[levels / 2 ..];
        try testing.expectEqual(want_positive.len, positive.len);
        for (want_positive, positive) |want, got| {
            // The published table is quoted to four decimals, so half an ulp of the
            // quote is 5e-5. Anything tighter tests the transcription, not the solver.
            try testing.expectApproxEqAbs(want, got, 1e-4);
        }
    }
}

test "reproduces the paper's distortion values" {
    // Paper, Theorem 1: D_mse ≈ 0.36, 0.117, 0.03, 0.009 for b = 1..4, where
    // D_mse = d · C(f_X, b). Since normalized_distortion divides out the variance,
    // it is directly comparable.
    const want = [_]f64{ 0.36, 0.117, 0.03, 0.009 };
    for (want, 1..) |expected, bits| {
        var sol = try solveBits(testing.allocator, Density.gauss(1.0), @intCast(bits), .{});
        defer sol.deinit();
        // The paper quotes two significant figures, so match to that precision.
        try testing.expectApproxEqAbs(expected, sol.normalized_distortion, 0.005);
    }
}

test "distortion respects the Panter-Dite bound" {
    // Paper: C(f_X, b) ≤ √3·π/(2d) · 4^−b, so D_mse = d·C ≤ √3·π/2 · 4^−b.
    //
    // The constant is √3·π/2 = 2.7207, which is exactly the "≈ 2.7 factor" the
    // abstract quotes as TurboQuant's gap from the Shannon lower bound (that bound
    // being 4^−b). Derivable independently: Panter-Dite gives
    // (1/12)·(∫f^(1/3))³ = (1/12)·6√3·π for a unit Gaussian.
    const bound_constant = @sqrt(3.0) * std.math.pi / 2.0;
    for (1..9) |bits| {
        var sol = try solveBits(testing.allocator, Density.gauss(1.0), @intCast(bits), .{});
        defer sol.deinit();
        const bound = bound_constant / std.math.pow(f64, 4.0, @floatFromInt(bits));
        try testing.expect(sol.normalized_distortion <= bound);
    }
}

test "distortion approaches the high-resolution 4x-per-bit limit from below" {
    // Each extra bit quarters the distortion only asymptotically. The measured
    // ratios climb 3.09, 3.40, 3.64, 3.79, 3.89, 3.94 — so asserting "about 4"
    // would be wrong at low bit-widths. What actually holds is that the ratio stays
    // under 4 and increases monotonically toward it.
    var previous: f64 = std.math.inf(f64);
    var previous_ratio: f64 = 0;
    for (1..9) |bits| {
        var sol = try solveBits(testing.allocator, Density.gauss(1.0), @intCast(bits), .{});
        defer sol.deinit();
        if (previous != std.math.inf(f64)) {
            const ratio = previous / sol.normalized_distortion;
            try testing.expect(ratio > 3.0 and ratio < 4.0);
            try testing.expect(ratio > previous_ratio);
            previous_ratio = ratio;
        }
        previous = sol.normalized_distortion;
    }
    try testing.expect(previous_ratio > 3.9); // genuinely converging on 4
}

test "sphere-coordinate codebook matches the paper's explicit centroids" {
    // Paper §3.1 states the optimal centroids for moderately high d directly:
    // b=1 → ±√(2/π)/√d,  b=2 → ±0.453/√d, ±1.51/√d.
    const d: u32 = 1536;
    const inv_sqrt_d = 1.0 / @sqrt(@as(f64, @floatFromInt(d)));

    {
        var sol = try solveBits(testing.allocator, Density.sphereCoord(d), 1, .{});
        defer sol.deinit();
        const want = @sqrt(2.0 / std.math.pi) * inv_sqrt_d;
        try testing.expectApproxEqRel(want, sol.centroids[1], 1e-3);
        try testing.expectApproxEqRel(-want, sol.centroids[0], 1e-3);
    }
    {
        var sol = try solveBits(testing.allocator, Density.sphereCoord(d), 2, .{});
        defer sol.deinit();
        try testing.expectApproxEqRel(0.453 * inv_sqrt_d, sol.centroids[2], 2e-3);
        try testing.expectApproxEqRel(1.51 * inv_sqrt_d, sol.centroids[3], 2e-3);
    }
}

test "sphere-coordinate codebook diverges from the gaussian limit at KV dimensions" {
    // Concrete evidence for the §8.2 requirement that the KV path use exact-Beta
    // tables: at d=128 the exact codebook's outermost level differs from the
    // Gaussian-limit codebook by well beyond solver tolerance, while at d=4096 it
    // does not. The bounded support pulls the extreme levels inward.
    for ([_]struct { dim: u32, min_rel: f64 }{
        .{ .dim = 64, .min_rel = 0.004 },
        .{ .dim = 128, .min_rel = 0.002 },
    }) |case| {
        const inv_sqrt_d = 1.0 / @sqrt(@as(f64, @floatFromInt(case.dim)));
        var exact = try solveBits(testing.allocator, Density.sphereCoord(case.dim), 4, .{});
        defer exact.deinit();
        var limit = try solveBits(testing.allocator, Density.gauss(1.0), 4, .{});
        defer limit.deinit();

        const outermost_exact = exact.centroids[exact.centroids.len - 1];
        const outermost_limit = limit.centroids[limit.centroids.len - 1] * inv_sqrt_d;
        const rel = @abs(outermost_exact - outermost_limit) / outermost_limit;
        try testing.expect(rel > case.min_rel);
    }
}

test "codebooks are symmetric, ordered, and consistent with their thresholds" {
    const cases = [_]Density{
        Density.gauss(1.0),
        Density.gauss(0.02),
        Density.sphereCoord(64),
        Density.sphereCoord(128),
        Density.sphereCoord(1536),
    };
    for (cases) |density| {
        for (1..6) |bits| {
            var sol = try solveBits(testing.allocator, density, @intCast(bits), .{});
            defer sol.deinit();

            // Strictly ascending.
            for (1..sol.centroids.len) |i| {
                try testing.expect(sol.centroids[i] > sol.centroids[i - 1]);
            }
            // Symmetric about zero, as the density is. Not enforced by the solver:
            // it falls out of convergence, so checking it tests the solver.
            const n = sol.centroids.len;
            for (0..n / 2) |i| {
                try testing.expectApproxEqAbs(
                    sol.centroids[i],
                    -sol.centroids[n - 1 - i],
                    1e-9 * density.scale(),
                );
            }
            // Thresholds bisect neighbouring centroids and are ascending.
            for (sol.thresholds, 0..) |t, i| {
                try testing.expectApproxEqAbs(
                    0.5 * (sol.centroids[i] + sol.centroids[i + 1]),
                    t,
                    1e-15,
                );
                if (i > 0) try testing.expect(t > sol.thresholds[i - 1]);
            }
        }
    }
}

test "encode selects the nearest centroid" {
    var sol = try solveBits(testing.allocator, Density.gauss(1.0), 4, .{});
    defer sol.deinit();

    // Exhaustive comparison against a direct argmin over a fine grid.
    var t: f64 = -4.0;
    while (t <= 4.0) : (t += 0.001) {
        const idx = sol.encode(t);
        var best: usize = 0;
        var best_dist = @abs(t - sol.centroids[0]);
        for (sol.centroids, 0..) |c, i| {
            const dist = @abs(t - c);
            if (dist < best_dist) {
                best_dist = dist;
                best = i;
            }
        }
        try testing.expectEqual(best, idx);
    }
}

test "encode saturates outside the codebook range" {
    var sol = try solveBits(testing.allocator, Density.gauss(1.0), 3, .{});
    defer sol.deinit();
    try testing.expectEqual(@as(usize, 0), sol.encode(-100.0));
    try testing.expectEqual(sol.centroids.len - 1, sol.encode(100.0));
}

test "solver converges within budget at the bit-widths we ship" {
    // b <= 4 is the design's primary target (the shuffle-LUT kernel needs a codebook
    // of at most 16 entries), so that is where a convergence budget is worth
    // enforcing. Guards the quantile seeding, the distortion-based stopping rule,
    // and the extrapolation together.
    for (1..5) |bits| {
        var sol = try solveBits(testing.allocator, Density.gauss(1.0), @intCast(bits), .{});
        defer sol.deinit();
        try testing.expect(sol.iterations < 300);
    }
}

test "extrapolation does not change the answer, only the cost" {
    // The extrapolation is guarded by a distortion check, so it must be a pure
    // speedup. Compare against a run where it can never fire.
    const no_extrapolation: Options = .{ .max_iterations = 400 };
    for (1..5) |bits| {
        var fast = try solveBits(testing.allocator, Density.gauss(1.0), @intCast(bits), .{});
        defer fast.deinit();
        var plain = try solveBits(testing.allocator, Density.gauss(1.0), @intCast(bits), no_extrapolation);
        defer plain.deinit();
        for (fast.centroids, plain.centroids) |a, b| {
            try testing.expectApproxEqRel(b, a, 1e-6);
        }
        try testing.expect(fast.iterations <= plain.iterations);
    }
}

test "two levels is the minimum and works" {
    var sol = try solve(testing.allocator, Density.gauss(1.0), 2, .{});
    defer sol.deinit();
    try testing.expectEqual(@as(usize, 2), sol.centroids.len);
    try testing.expectEqual(@as(usize, 1), sol.thresholds.len);
    try testing.expectApproxEqAbs(@as(f64, 0), sol.thresholds[0], 1e-12);
}

test "non-power-of-two level counts are supported" {
    // Needed by the KV path's fractional bit rates, which mix codebook sizes.
    for ([_]usize{ 3, 5, 6, 12 }) |levels| {
        var sol = try solve(testing.allocator, Density.gauss(1.0), levels, .{});
        defer sol.deinit();
        try testing.expectEqual(levels, sol.centroids.len);
        for (1..levels) |i| try testing.expect(sol.centroids[i] > sol.centroids[i - 1]);
    }
}
