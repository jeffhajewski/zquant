//! Random rotations.
//!
//! TurboQuant's first step is `y = Π·x` for a random rotation Π. Everything after it
//! depends on that step actually producing a uniformly-distributed direction: the
//! Beta coordinate density, the near-independence of coordinates, and therefore the
//! optimality of the per-coordinate scalar quantizer.
//!
//! Two implementations:
//!
//!   - `.dense` — QR of a Gaussian matrix, as the paper specifies. Exactly Haar, and
//!     O(d²) in both time and storage. Kept permanently as the reference oracle
//!     against which the fast path is measured (docs/DESIGN.md §7.2).
//!
//!   - `.hadamard` — a Randomized Hadamard Transform, `H·D_{R−1} ⋯ H·D_1·H·D_0` for
//!     random ±1 diagonals D_i. O(d log d), and described entirely by R·d sign bits
//!     derived from a seed, so nothing needs storing or serializing.
//!
//! The design sketch also interleaved a random permutation between rounds. It is not
//! implemented, because FWHT already makes every output coordinate depend on every
//! input coordinate, so the permutation adds little mixing that another sign-flip
//! round does not — and dropping it keeps `apply` allocation-free and thread-safe,
//! since permuting in place otherwise needs scratch. The mixing tests below are what
//! justify the omission; if they ever fail, add rounds first.

const std = @import("std");
const Allocator = std.mem.Allocator;
const rng = @import("rng.zig");

pub const Kind = enum {
    /// Fast path: randomized Hadamard transform.
    hadamard,
    /// Reference path: exactly Haar-distributed, via QR of a Gaussian matrix.
    dense,
};

/// Three rounds. One round leaves a standard basis vector with every coordinate at
/// identical magnitude (see the adversarial test below), which is precisely the
/// input distribution the scalar quantizer is not designed for. Two fixes it; three
/// leaves margin.
pub const default_rounds: u32 = 3;

pub const Rotation = struct {
    /// Logical input dimension.
    dim: u32,
    /// Working dimension: `dim` rounded up to a power of two.
    padded: u32,
    kind: Kind,
    rounds: u32,

    /// `.hadamard`: rounds × padded sign flips. `.dense`: padded × padded, row-major.
    data: []f32,
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        dim: u32,
        kind: Kind,
        seed: u64,
        purpose: rng.Purpose,
    ) Allocator.Error!Rotation {
        std.debug.assert(dim >= 1);
        const padded = std.math.ceilPowerOfTwoAssert(u32, dim);
        const rounds = default_rounds;

        const data = switch (kind) {
            .hadamard => try allocator.alloc(f32, rounds * padded),
            .dense => try allocator.alloc(f32, @as(usize, padded) * padded),
        };
        errdefer allocator.free(data);

        var stream = rng.Philox.init(seed).stream(purpose);
        switch (kind) {
            .hadamard => stream.fillSigns(data),
            .dense => buildDense(data, padded, &stream),
        }

        return .{
            .dim = dim,
            .padded = padded,
            .kind = kind,
            .rounds = rounds,
            .data = data,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Rotation) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }

    /// `dst = Π · src`, zero-padding `src` out to `padded`.
    ///
    /// `src` may be shorter than `padded` (normally exactly `dim`); `dst` must be
    /// exactly `padded` long.
    pub fn apply(self: Rotation, src: []const f32, dst: []f32) void {
        std.debug.assert(src.len <= self.padded);
        std.debug.assert(dst.len == self.padded);

        switch (self.kind) {
            .hadamard => {
                @memcpy(dst[0..src.len], src);
                @memset(dst[src.len..], 0);
                for (0..self.rounds) |round| {
                    applySigns(dst, self.signsFor(round));
                    walshHadamard(dst);
                }
            },
            .dense => {
                // The padded tail is zero, so it contributes nothing and the inner
                // loop can stop at src.len.
                const n = self.padded;
                for (0..n) |i| {
                    const row = self.data[i * n ..][0..n];
                    var sum: f32 = 0;
                    for (row[0..src.len], src) |m, x| sum += m * x;
                    dst[i] = sum;
                }
            },
        }
    }

    /// `dst = Πᵀ · src`. Both slices are `padded` long; the caller reads `dst[0..dim]`.
    pub fn applyInverse(self: Rotation, src: []const f32, dst: []f32) void {
        std.debug.assert(src.len == self.padded);
        std.debug.assert(dst.len == self.padded);

        switch (self.kind) {
            .hadamard => {
                @memcpy(dst, src);
                // Π = H·D_{R−1} ⋯ H·D_0, so Πᵀ = D_0·H ⋯ D_{R−1}·H: reverse the
                // rounds and swap the order within each.
                var round = self.rounds;
                while (round > 0) {
                    round -= 1;
                    walshHadamard(dst);
                    applySigns(dst, self.signsFor(round));
                }
            },
            .dense => {
                const n = self.padded;
                @memset(dst, 0);
                for (0..n) |i| {
                    const row = self.data[i * n ..][0..n];
                    const s = src[i];
                    for (dst, row) |*d, m| d.* += m * s;
                }
            },
        }
    }

    fn signsFor(self: Rotation, round: usize) []const f32 {
        return self.data[round * self.padded ..][0..self.padded];
    }
};

fn applySigns(x: []f32, signs: []const f32) void {
    for (x, signs) |*v, s| v.* *= s;
}

/// In-place fast Walsh-Hadamard transform, normalized so the operator is orthogonal.
///
/// The unnormalized transform satisfies H² = n·I; scaling by 1/√n makes H/√n both
/// orthogonal and symmetric, hence its own inverse. `applyInverse` relies on that.
pub fn walshHadamard(x: []f32) void {
    const n = x.len;
    std.debug.assert(std.math.isPowerOfTwo(n));

    var len: usize = 1;
    while (len < n) : (len <<= 1) {
        var base: usize = 0;
        while (base < n) : (base += len << 1) {
            for (0..len) |j| {
                const a = x[base + j];
                const b = x[base + j + len];
                x[base + j] = a + b;
                x[base + j + len] = a - b;
            }
        }
    }

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(n)));
    for (x) |*v| v.* *= scale;
}

/// Haar-random orthogonal matrix by modified Gram-Schmidt on Gaussian columns.
///
/// Modified rather than classical Gram-Schmidt: the classical form loses
/// orthogonality badly at the dimensions used here, and this is the oracle the fast
/// path is judged against, so its own error needs to stay far below the effect being
/// measured.
fn buildDense(matrix: []f32, n: u32, stream: *rng.Stream) void {
    for (matrix) |*v| v.* = @floatCast(stream.nextGaussian());

    // Rows are the basis vectors being orthonormalized.
    for (0..n) |i| {
        const row = matrix[i * n ..][0..n];
        for (0..i) |j| {
            const done = matrix[j * n ..][0..n];
            var dot: f64 = 0;
            for (row, done) |a, b| dot += @as(f64, a) * @as(f64, b);
            const d: f32 = @floatCast(dot);
            for (row, done) |*a, b| a.* -= d * b;
        }
        var norm: f64 = 0;
        for (row) |a| norm += @as(f64, a) * @as(f64, a);
        const inv: f32 = @floatCast(1.0 / @sqrt(norm));
        for (row) |*a| a.* *= inv;
    }
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;

fn randomUnitVector(buf: []f32, stream: *rng.Stream) void {
    var norm: f64 = 0;
    for (buf) |*v| {
        const g = stream.nextGaussian();
        v.* = @floatCast(g);
        norm += g * g;
    }
    const inv: f32 = @floatCast(1.0 / @sqrt(norm));
    for (buf) |*v| v.* *= inv;
}

test "walsh-hadamard matches a naive hadamard matrix product" {
    const n = 16;
    var x = [_]f32{ 3, -1, 4, 1, -5, 9, 2, 6, -5, 3, 5, -8, 9, 7, -9, 3 };
    const original = x;

    walshHadamard(&x);

    // H[i][j] = (-1)^popcount(i & j), scaled by 1/sqrt(n).
    const scale = 1.0 / @sqrt(@as(f32, n));
    for (0..n) |i| {
        var expected: f32 = 0;
        for (0..n) |j| {
            const sign: f32 = if (@popCount(i & j) % 2 == 0) 1 else -1;
            expected += sign * original[j];
        }
        try testing.expectApproxEqAbs(expected * scale, x[i], 1e-4);
    }
}

test "normalized walsh-hadamard is its own inverse" {
    var x = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const original = x;
    walshHadamard(&x);
    walshHadamard(&x);
    for (original, x) |want, got| try testing.expectApproxEqAbs(want, got, 1e-5);
}

test "walsh-hadamard preserves norm" {
    var stream = rng.Philox.init(1).stream(.testing);
    var x: [64]f32 = undefined;
    randomUnitVector(&x, &stream);

    var before: f64 = 0;
    for (x) |v| before += @as(f64, v) * v;
    walshHadamard(&x);
    var after: f64 = 0;
    for (x) |v| after += @as(f64, v) * v;

    try testing.expectApproxEqAbs(before, after, 1e-6);
}

test "both rotations preserve norm" {
    var stream = rng.Philox.init(42).stream(.testing);
    for ([_]Kind{ .hadamard, .dense }) |kind| {
        for ([_]u32{ 8, 64, 128 }) |dim| {
            var rot = try Rotation.init(testing.allocator, dim, kind, 0x5EED, .rht_signs);
            defer rot.deinit();

            const src = try testing.allocator.alloc(f32, dim);
            defer testing.allocator.free(src);
            const dst = try testing.allocator.alloc(f32, rot.padded);
            defer testing.allocator.free(dst);

            randomUnitVector(src, &stream);
            rot.apply(src, dst);

            var norm: f64 = 0;
            for (dst) |v| norm += @as(f64, v) * v;
            try testing.expectApproxEqAbs(@as(f64, 1.0), norm, 1e-5);
        }
    }
}

test "applyInverse undoes apply" {
    var stream = rng.Philox.init(7).stream(.testing);
    for ([_]Kind{ .hadamard, .dense }) |kind| {
        for ([_]u32{ 4, 32, 256 }) |dim| {
            var rot = try Rotation.init(testing.allocator, dim, kind, 0xC0FFEE, .rht_signs);
            defer rot.deinit();

            const src = try testing.allocator.alloc(f32, dim);
            defer testing.allocator.free(src);
            const rotated = try testing.allocator.alloc(f32, rot.padded);
            defer testing.allocator.free(rotated);
            const back = try testing.allocator.alloc(f32, rot.padded);
            defer testing.allocator.free(back);

            randomUnitVector(src, &stream);
            rot.apply(src, rotated);
            rot.applyInverse(rotated, back);

            for (src, back[0..dim]) |want, got| {
                try testing.expectApproxEqAbs(want, got, 1e-5);
            }
            // The padded tail must come back as zero, not as leaked energy.
            for (back[dim..]) |v| try testing.expectApproxEqAbs(@as(f32, 0), v, 1e-5);
        }
    }
}

test "rotations preserve inner products" {
    // The property the whole estimator rests on: <Πa, Πb> = <a, b>.
    var stream = rng.Philox.init(99).stream(.testing);
    for ([_]Kind{ .hadamard, .dense }) |kind| {
        var rot = try Rotation.init(testing.allocator, 128, kind, 0xBEEF, .rht_signs);
        defer rot.deinit();

        var a: [128]f32 = undefined;
        var b: [128]f32 = undefined;
        var ra: [128]f32 = undefined;
        var rb: [128]f32 = undefined;
        randomUnitVector(&a, &stream);
        randomUnitVector(&b, &stream);
        rot.apply(&a, &ra);
        rot.apply(&b, &rb);

        var before: f64 = 0;
        var after: f64 = 0;
        for (a, b) |x, y| before += @as(f64, x) * y;
        for (ra, rb) |x, y| after += @as(f64, x) * y;
        try testing.expectApproxEqAbs(before, after, 1e-5);
    }
}

test "dense rotation rows are orthonormal" {
    var rot = try Rotation.init(testing.allocator, 64, .dense, 0x1234, .dense_rotation);
    defer rot.deinit();
    const n = rot.padded;

    for (0..n) |i| {
        for (i..n) |j| {
            var dot: f64 = 0;
            for (0..n) |k| dot += @as(f64, rot.data[i * n + k]) * rot.data[j * n + k];
            const want: f64 = if (i == j) 1.0 else 0.0;
            try testing.expectApproxEqAbs(want, dot, 1e-5);
        }
    }
}

test "one round leaves a basis vector unmixed; more rounds fix it" {
    // The concrete reason default_rounds > 1. A single H·D maps e_0 to ±(1/√n)
    // across every coordinate: all magnitudes identical, which is the opposite of
    // the Beta-distributed spread the scalar quantizer is built for. It is also the
    // worst case for a real corpus, since one-hot-ish and axis-aligned vectors do
    // occur.
    const n: u32 = 256;
    var src = [_]f32{0} ** n;
    src[0] = 1.0;

    var dst: [n]f32 = undefined;

    {
        var rot = try Rotation.init(testing.allocator, n, .hadamard, 0xAAAA, .rht_signs);
        defer rot.deinit();
        rot.rounds = 1;
        rot.apply(&src, &dst);

        // Every magnitude is exactly 1/√n: zero spread.
        const expected = 1.0 / @sqrt(@as(f32, n));
        for (dst) |v| try testing.expectApproxEqAbs(expected, @abs(v), 1e-6);
    }

    {
        var rot = try Rotation.init(testing.allocator, n, .hadamard, 0xAAAA, .rht_signs);
        defer rot.deinit();
        try testing.expectEqual(@as(u32, 3), rot.rounds);
        rot.apply(&src, &dst);

        // Now the magnitudes vary. For a uniform point on the sphere the mean of
        // |y_j| is about 0.798/√n and the spread is comparable, so requiring the
        // sample deviation to be a decent fraction of the mean is a weak but
        // decisive separation from the degenerate case above.
        var mean: f64 = 0;
        for (dst) |v| mean += @abs(v);
        mean /= n;
        var variance: f64 = 0;
        for (dst) |v| {
            const dev = @abs(v) - mean;
            variance += dev * dev;
        }
        variance /= n;
        try testing.expect(@sqrt(variance) / mean > 0.3);
    }
}

test "hadamard rotation reproduces the sphere-coordinate moments" {
    // The statistical claim the whole algorithm depends on, checked against the
    // exact density rather than against intuition: coordinates of Πx should have
    // variance 1/d and the fourth moment 3/(d(d+2)) that the sphere coordinate
    // density has (versus 3/d² for a true Gaussian).
    const n: u32 = 64;
    const trials = 4000;

    var stream = rng.Philox.init(0xD09).stream(.testing);
    var rot = try Rotation.init(testing.allocator, n, .hadamard, 0x51DE, .rht_signs);
    defer rot.deinit();

    var src: [n]f32 = undefined;
    var dst: [n]f32 = undefined;
    var sum2: f64 = 0;
    var sum4: f64 = 0;

    for (0..trials) |_| {
        randomUnitVector(&src, &stream);
        rot.apply(&src, &dst);
        for (dst) |v| {
            const sq = @as(f64, v) * v;
            sum2 += sq;
            sum4 += sq * sq;
        }
    }

    const count: f64 = @floatFromInt(trials * n);
    const d: f64 = @floatFromInt(n);
    try testing.expectApproxEqRel(1.0 / d, sum2 / count, 0.02);
    try testing.expectApproxEqRel(3.0 / (d * (d + 2.0)), sum4 / count, 0.05);
}

test "hadamard and dense rotations agree statistically" {
    // The §7.2 gate in miniature: the fast path must produce the same coordinate
    // distribution as the Haar reference, since that is the only property the
    // quantizer's optimality argument uses.
    const n: u32 = 64;
    const trials = 3000;

    var moments: [2]f64 = undefined;
    for ([_]Kind{ .hadamard, .dense }, 0..) |kind, ki| {
        var stream = rng.Philox.init(0x1111).stream(.testing);
        var rot = try Rotation.init(testing.allocator, n, kind, 0x2222, .rht_signs);
        defer rot.deinit();

        var src: [n]f32 = undefined;
        var dst: [n]f32 = undefined;
        var sum4: f64 = 0;
        for (0..trials) |_| {
            randomUnitVector(&src, &stream);
            rot.apply(&src, &dst);
            for (dst) |v| {
                const sq = @as(f64, v) * v;
                sum4 += sq * sq;
            }
        }
        moments[ki] = sum4 / @as(f64, @floatFromInt(trials * n));
    }
    try testing.expectApproxEqRel(moments[1], moments[0], 0.05);
}

test "non-power-of-two dimensions are padded" {
    for ([_][2]u32{ .{ 200, 256 }, .{ 768, 1024 }, .{ 1536, 2048 }, .{ 3, 4 } }) |case| {
        var rot = try Rotation.init(testing.allocator, case[0], .hadamard, 1, .rht_signs);
        defer rot.deinit();
        try testing.expectEqual(case[1], rot.padded);
    }
}

test "exact powers of two are not padded further" {
    for ([_]u32{ 1, 2, 64, 128, 1024 }) |dim| {
        var rot = try Rotation.init(testing.allocator, dim, .hadamard, 1, .rht_signs);
        defer rot.deinit();
        try testing.expectEqual(dim, rot.padded);
    }
}

test "different seeds and purposes give different rotations" {
    var a = try Rotation.init(testing.allocator, 32, .hadamard, 1, .rht_signs);
    defer a.deinit();
    var b = try Rotation.init(testing.allocator, 32, .hadamard, 2, .rht_signs);
    defer b.deinit();
    var c = try Rotation.init(testing.allocator, 32, .hadamard, 1, .sketch_signs);
    defer c.deinit();

    try testing.expect(!std.mem.eql(f32, a.data, b.data));
    // Independence of Π and S' rests on this: same seed, different purpose.
    try testing.expect(!std.mem.eql(f32, a.data, c.data));
}

test "the same seed reproduces the same rotation" {
    var a = try Rotation.init(testing.allocator, 128, .hadamard, 0xFEEDFACE, .rht_signs);
    defer a.deinit();
    var b = try Rotation.init(testing.allocator, 128, .hadamard, 0xFEEDFACE, .rht_signs);
    defer b.deinit();
    try testing.expectEqualSlices(f32, a.data, b.data);
}
