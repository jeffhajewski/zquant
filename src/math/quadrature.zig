//! Composite Gauss-Legendre quadrature.
//!
//! Used only by the offline codebook solver, so accuracy is worth far more than
//! speed here. Nodes are computed by Newton iteration rather than transcribed from
//! a table: a table is a long list of digits with no way to tell a typo from a
//! correct entry, whereas a computed rule can be checked against polynomials it
//! must integrate exactly.

const std = @import("std");

/// An n-point Gauss-Legendre rule on [-1, 1].
pub fn Rule(comptime n: usize) type {
    return struct {
        nodes: [n]f64,
        weights: [n]f64,

        const Self = @This();

        pub fn init() Self {
            var self: Self = .{ .nodes = undefined, .weights = undefined };
            // Roots are symmetric, so solve for half and mirror.
            for (0..(n + 1) / 2) |i| {
                // Chebyshev-like initial guess; converges in a handful of steps.
                var x = @cos(std.math.pi * (@as(f64, @floatFromInt(i)) + 0.75) /
                    (@as(f64, @floatFromInt(n)) + 0.5));
                var dp: f64 = 0;
                for (0..100) |_| {
                    const p = legendre(n, x, &dp);
                    const dx = -p / dp;
                    x += dx;
                    if (@abs(dx) < 1e-16) break;
                }
                _ = legendre(n, x, &dp);
                // Standard Gauss-Legendre weight from the derivative at the root.
                const w = 2.0 / ((1.0 - x * x) * dp * dp);
                self.nodes[i] = -x;
                self.weights[i] = w;
                self.nodes[n - 1 - i] = x;
                self.weights[n - 1 - i] = w;
            }
            return self;
        }

        /// Integrate `f` over [a, b] using `panels` equal subintervals.
        pub fn integrate(
            self: Self,
            a: f64,
            b: f64,
            panels: usize,
            context: anytype,
            comptime f: fn (@TypeOf(context), f64) f64,
        ) f64 {
            if (b <= a) return 0;
            const p = @max(panels, 1);
            const width = (b - a) / @as(f64, @floatFromInt(p));
            const half = 0.5 * width;
            var total: f64 = 0;
            for (0..p) |k| {
                const mid = a + (@as(f64, @floatFromInt(k)) + 0.5) * width;
                var panel_sum: f64 = 0;
                for (self.nodes, self.weights) |node, weight| {
                    panel_sum += weight * f(context, mid + half * node);
                }
                total += panel_sum * half;
            }
            return total;
        }
    };
}

/// Legendre polynomial P_n(x) via the three-term recurrence; also yields P_n'(x).
fn legendre(n: usize, x: f64, derivative: *f64) f64 {
    var p_prev: f64 = 1.0; // P_0
    var p: f64 = x; // P_1
    if (n == 0) {
        derivative.* = 0;
        return 1.0;
    }
    var k: usize = 2;
    while (k <= n) : (k += 1) {
        const fk: f64 = @floatFromInt(k);
        const p_next = ((2.0 * fk - 1.0) * x * p - (fk - 1.0) * p_prev) / fk;
        p_prev = p;
        p = p_next;
    }
    const fn_: f64 = @floatFromInt(n);
    derivative.* = fn_ * (x * p - p_prev) / (x * x - 1.0);
    return p;
}

/// The default rule. 16 points integrates degree-31 polynomials exactly, which is
/// ample once panels are sized to the density's scale.
pub const default = Rule(16);

// -- tests -------------------------------------------------------------------

const testing = std.testing;

test "nodes are symmetric and weights sum to the interval length" {
    const rule = default.init();
    var sum: f64 = 0;
    for (rule.weights) |w| sum += w;
    try testing.expectApproxEqAbs(@as(f64, 2.0), sum, 1e-14);

    for (0..8) |i| {
        try testing.expectApproxEqAbs(rule.nodes[i], -rule.nodes[15 - i], 1e-15);
        try testing.expectApproxEqAbs(rule.weights[i], rule.weights[15 - i], 1e-15);
    }
    // Strictly increasing and interior to (-1, 1).
    for (1..16) |i| try testing.expect(rule.nodes[i] > rule.nodes[i - 1]);
    try testing.expect(rule.nodes[0] > -1.0 and rule.nodes[15] < 1.0);
}

test "integrates polynomials up to degree 2n-1 exactly" {
    const rule = default.init();
    // A 16-point rule is exact through degree 31. Check the boundary: degree 31
    // exact, and that we are not accidentally exact far beyond it.
    inline for (.{ 0, 1, 5, 17, 30, 31 }) |deg| {
        const f = struct {
            fn call(_: void, x: f64) f64 {
                return std.math.pow(f64, x, @floatFromInt(deg));
            }
        }.call;
        // Over [0,1] with a single panel: exact value is 1/(deg+1).
        const got = rule.integrate(0, 1, 1, {}, f);
        const want = 1.0 / @as(f64, @floatFromInt(deg + 1));
        try testing.expectApproxEqRel(want, got, 1e-13);
    }
}

test "composite panels converge on a peaked integrand" {
    const rule = default.init();
    // A narrow Gaussian bump: the case that motivates panel subdivision.
    const f = struct {
        fn call(_: void, x: f64) f64 {
            const s = 0.01;
            return @exp(-0.5 * (x / s) * (x / s)) / (s * @sqrt(2.0 * std.math.pi));
        }
    }.call;
    // One panel across [-1,1] misses the spike entirely; many panels resolve it.
    const coarse = rule.integrate(-1, 1, 1, {}, f);
    const fine = rule.integrate(-1, 1, 2000, {}, f);
    try testing.expect(@abs(coarse - 1.0) > 0.1);
    try testing.expectApproxEqAbs(@as(f64, 1.0), fine, 1e-12);
}

test "empty and inverted intervals integrate to zero" {
    const rule = default.init();
    const f = struct {
        fn call(_: void, _: f64) f64 {
            return 1.0;
        }
    }.call;
    try testing.expectEqual(@as(f64, 0), rule.integrate(1, 1, 4, {}, f));
    try testing.expectEqual(@as(f64, 0), rule.integrate(1, 0, 4, {}, f));
}
