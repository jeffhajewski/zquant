//! The scalar densities that TurboQuant quantizes against.
//!
//! The algorithm's central reduction: for x on the unit sphere and a Haar-random
//! rotation Π, each coordinate of Πx is distributed as
//!
//!     f_X(t) = Γ(d/2) / (√π · Γ((d−1)/2)) · (1 − t²)^((d−3)/2),   t ∈ [−1, 1]
//!
//! and distinct coordinates are near-independent in high d. So the d-dimensional
//! quantization problem collapses to one scalar quantizer applied d times.
//!
//! `f_X` converges to N(0, 1/d). Both are provided because that convergence is not
//! uniform in d: at the embedding dimensions used for vector search the Gaussian
//! limit is fine, but the KV-cache path runs at d = 64..128 where the difference is
//! measurable (see docs/DESIGN.md §8.2). Codebooks there must use `sphereCoord`.

const std = @import("std");
const quadrature = @import("quadrature.zig");

/// Container-scope, so the Newton solve for the nodes runs once at comptime rather
/// than on every `mass` call. It was being rebuilt inside Lloyd's inner loop, which
/// dominated the offline solver's runtime.
const rule = blk: {
    @setEvalBranchQuota(200_000);
    break :blk quadrature.default.init();
};

pub const Density = union(enum) {
    /// N(0, sigma²).
    gaussian: struct { sigma: f64 },
    /// Distribution of a single coordinate of a uniform point on the unit sphere
    /// S^(dim−1). Requires dim >= 3, where the exponent (dim−3)/2 is non-negative
    /// and the density is bounded.
    sphere_coord: struct { dim: u32 },

    pub fn gauss(sigma: f64) Density {
        std.debug.assert(sigma > 0);
        return .{ .gaussian = .{ .sigma = sigma } };
    }

    pub fn sphereCoord(dim: u32) Density {
        std.debug.assert(dim >= 3);
        return .{ .sphere_coord = .{ .dim = dim } };
    }

    /// Standard deviation. Used to size quadrature panels and to normalize
    /// convergence tolerances, so it must be exact rather than approximate.
    pub fn scale(self: Density) f64 {
        return @sqrt(self.variance());
    }

    pub fn variance(self: Density) f64 {
        return switch (self) {
            .gaussian => |g| g.sigma * g.sigma,
            // A uniform point on S^(d−1) has E[t_j²] = 1/d, since the coordinates
            // are exchangeable and sum of squares is 1.
            .sphere_coord => |s| 1.0 / @as(f64, @floatFromInt(s.dim)),
        };
    }

    /// True support. Unbounded for the Gaussian; truncated at a point where the
    /// remaining tail mass is far below f64 precision.
    pub fn support(self: Density) [2]f64 {
        return switch (self) {
            .gaussian => |g| .{ -40.0 * g.sigma, 40.0 * g.sigma },
            .sphere_coord => .{ -1.0, 1.0 },
        };
    }

    pub fn logPdf(self: Density, t: f64) f64 {
        switch (self) {
            .gaussian => |g| {
                const z = t / g.sigma;
                return -0.5 * z * z - @log(g.sigma) - 0.5 * @log(2.0 * std.math.pi);
            },
            .sphere_coord => |s| {
                if (t <= -1.0 or t >= 1.0) return -std.math.inf(f64);
                const d: f64 = @floatFromInt(s.dim);
                const p = (d - 3.0) * 0.5;
                // log1p keeps precision when t is small, which is where essentially
                // all the mass lives once d is large.
                return logNorm(s.dim) + p * std.math.log1p(-t * t);
            },
        }
    }

    pub fn pdf(self: Density, t: f64) f64 {
        return @exp(self.logPdf(t));
    }

    /// ∫_a^b t · f(t) dt, in closed form for both densities.
    ///
    /// Closed form matters: this is what Lloyd's update divides by the cell mass, so
    /// quadrature error here would bias every centroid rather than merely blur it.
    pub fn moment(self: Density, a: f64, b: f64) f64 {
        if (b <= a) return 0;
        switch (self) {
            .gaussian => |g| {
                // ∫ t·φ(t) dt = −σ²·φ(t)
                return g.sigma * g.sigma * (self.pdf(a) - self.pdf(b));
            },
            .sphere_coord => |s| {
                // ∫ t(1−t²)^p dt = −(1−t²)^(p+1) / (2(p+1))
                const d: f64 = @floatFromInt(s.dim);
                const p = (d - 3.0) * 0.5;
                const q = p + 1.0;
                const lo = clampUnit(a);
                const hi = clampUnit(b);
                const term_a = @exp(logNorm(s.dim) + q * std.math.log1p(-lo * lo));
                const term_b = @exp(logNorm(s.dim) + q * std.math.log1p(-hi * hi));
                return (term_a - term_b) / (2.0 * q);
            },
        }
    }

    /// ∫_a^b f(t) dt, by composite Gauss-Legendre.
    ///
    /// No elementary closed form exists for `sphere_coord` (it is a regularized
    /// incomplete beta), and there is no `erf` in std, so both go through quadrature
    /// with panels sized to the density's own scale.
    pub fn mass(self: Density, a: f64, b: f64) f64 {
        if (b <= a) return 0;
        const bounds = self.support();
        const lo = @max(a, bounds[0]);
        const hi = @min(b, bounds[1]);
        if (hi <= lo) return 0;

        return rule.integrate(lo, hi, self.panelsFor(hi - lo), self, evalPdf);
    }

    fn panelsFor(self: Density, width: f64) usize {
        // Four panels per standard deviation resolves the peak; a 16-point rule per
        // panel then leaves plenty of margin.
        const want = @ceil(width / (0.25 * self.scale()));
        if (!std.math.isFinite(want) or want < 1) return 1;
        return @min(@as(usize, @intFromFloat(want)), 200_000);
    }

    fn evalPdf(self: Density, t: f64) f64 {
        return self.pdf(t);
    }

    /// Inverse CDF by bisection. Only used to seed Lloyd's iteration, so robustness
    /// matters more than speed.
    pub fn quantile(self: Density, p: f64) f64 {
        std.debug.assert(p > 0 and p < 1);
        const bounds = self.support();
        var lo = bounds[0];
        var hi = bounds[1];
        for (0..200) |_| {
            const mid = 0.5 * (lo + hi);
            if (self.mass(bounds[0], mid) < p) lo = mid else hi = mid;
            if (hi - lo < 1e-15 * self.scale()) break;
        }
        return 0.5 * (lo + hi);
    }
};

/// log of the normalizing constant Γ(d/2) / (√π · Γ((d−1)/2)).
///
/// Computed via lgamma because the ratio of gammas overflows f64 well before the
/// dimensions we care about: Γ(768) is far past the f64 range, while its log is not.
fn logNorm(dim: u32) f64 {
    const d: f64 = @floatFromInt(dim);
    return std.math.lgamma(f64, d * 0.5) -
        std.math.lgamma(f64, (d - 1.0) * 0.5) -
        0.5 * @log(std.math.pi);
}

fn clampUnit(t: f64) f64 {
    return @min(@max(t, -1.0), 1.0);
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;

test "densities integrate to one" {
    const cases = [_]Density{
        Density.gauss(1.0),
        Density.gauss(0.05),
        Density.sphereCoord(3),
        Density.sphereCoord(8),
        Density.sphereCoord(64),
        Density.sphereCoord(128),
        Density.sphereCoord(1536),
        Density.sphereCoord(3072),
    };
    for (cases) |d| {
        const bounds = d.support();
        try testing.expectApproxEqAbs(@as(f64, 1.0), d.mass(bounds[0], bounds[1]), 1e-10);
    }
}

test "second moment matches the stated variance" {
    // Independent check on both the normalization and the quadrature: integrate
    // t²·f numerically and compare against the closed-form variance.
    const cases = [_]Density{
        Density.gauss(1.0),
        Density.sphereCoord(16),
        Density.sphereCoord(128),
        Density.sphereCoord(1536),
    };
    for (cases) |d| {
        const bounds = d.support();
        const sq = struct {
            fn call(dens: Density, t: f64) f64 {
                return t * t * dens.pdf(t);
            }
        }.call;
        const got = rule.integrate(bounds[0], bounds[1], 20_000, d, sq);
        try testing.expectApproxEqRel(d.variance(), got, 1e-8);
    }
}

test "closed-form moment agrees with quadrature" {
    // The moment integral is the one place a closed form is load-bearing, so verify
    // it against an independent numerical evaluation.
    const cases = [_]Density{
        Density.gauss(1.0),
        Density.sphereCoord(9),
        Density.sphereCoord(128),
        Density.sphereCoord(1536),
    };
    for (cases) |d| {
        const s = d.scale();
        const intervals = [_][2]f64{
            .{ -0.5 * s, 0.5 * s },
            .{ 0.2 * s, 3.0 * s },
            .{ -4.0 * s, -1.0 * s },
            .{ 0.0, 8.0 * s },
        };
        const tf = struct {
            fn call(dens: Density, t: f64) f64 {
                return t * dens.pdf(t);
            }
        }.call;
        for (intervals) |iv| {
            const a = @max(iv[0], d.support()[0]);
            const b = @min(iv[1], d.support()[1]);
            const want = rule.integrate(a, b, 20_000, d, tf);
            const got = d.moment(a, b);
            try testing.expectApproxEqAbs(want, got, 1e-12 + 1e-8 * @abs(want));
        }
    }
}

test "sphere coordinate converges to the gaussian limit as d grows" {
    // The premise behind using N(0,1/d) tables at embedding dimensions.
    var prev: f64 = std.math.inf(f64);
    for ([_]u32{ 16, 64, 256, 1024, 4096 }) |d| {
        const sphere = Density.sphereCoord(d);
        const limit = Density.gauss(1.0 / @sqrt(@as(f64, @floatFromInt(d))));
        // Compare mass over a fixed number of standard deviations.
        const s = sphere.scale();
        const err = @abs(sphere.mass(-s, s) - limit.mass(-s, s));
        try testing.expect(err < prev); // strictly improving
        prev = err;
    }
    try testing.expect(prev < 1e-4);
}

test "sphere coordinate differs measurably from the gaussian limit at KV dimensions" {
    // This is the concrete justification for §8.2 requiring exact-Beta codebooks on
    // the KV path rather than reusing the Gaussian tables. If this ever stops being
    // true, that requirement can be revisited.
    for ([_]u32{ 64, 128 }) |d| {
        const sphere = Density.sphereCoord(d);
        const limit = Density.gauss(1.0 / @sqrt(@as(f64, @floatFromInt(d))));
        const s = sphere.scale();
        // Tail mass beyond 2 sigma is where the bounded support bites.
        const sphere_tail = sphere.mass(2.0 * s, sphere.support()[1]);
        const limit_tail = limit.mass(2.0 * s, limit.support()[1]);
        const rel = @abs(sphere_tail - limit_tail) / limit_tail;
        try testing.expect(rel > 0.005);
    }
}

test "densities are symmetric" {
    for ([_]Density{ Density.gauss(2.0), Density.sphereCoord(5), Density.sphereCoord(200) }) |d| {
        const s = d.scale();
        for ([_]f64{ 0.1, 0.5, 1.0, 2.0 }) |k| {
            try testing.expectApproxEqRel(d.pdf(k * s), d.pdf(-k * s), 1e-14);
            try testing.expectApproxEqRel(
                d.mass(0, k * s),
                d.mass(-k * s, 0),
                1e-10,
            );
        }
    }
}

test "sphere coordinate has zero density outside [-1, 1]" {
    const d = Density.sphereCoord(32);
    try testing.expectEqual(@as(f64, 0), d.pdf(1.0));
    try testing.expectEqual(@as(f64, 0), d.pdf(-1.0));
    try testing.expectEqual(@as(f64, 0), d.pdf(1.5));
}

test "dim 3 is exactly uniform" {
    // (1 − t²)^0 = 1, so the density is uniform on [−1, 1] with pdf 1/2.
    const d = Density.sphereCoord(3);
    try testing.expectApproxEqAbs(@as(f64, 0.5), d.pdf(0.0), 1e-14);
    try testing.expectApproxEqAbs(@as(f64, 0.5), d.pdf(0.9), 1e-14);
    try testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), d.variance(), 1e-14);
}

test "quantile inverts mass" {
    for ([_]Density{ Density.gauss(1.0), Density.sphereCoord(128) }) |d| {
        for ([_]f64{ 0.05, 0.25, 0.5, 0.75, 0.95 }) |p| {
            const t = d.quantile(p);
            try testing.expectApproxEqAbs(p, d.mass(d.support()[0], t), 1e-9);
        }
        try testing.expectApproxEqAbs(@as(f64, 0), d.quantile(0.5), 1e-12 + 1e-9 * d.scale());
    }
}
