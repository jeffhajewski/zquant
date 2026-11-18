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
//!
//! ## Dimensions that are not a power of two
//!
//! FWHT needs a power of two, and zero-padding up to one wastes up to half the codes:
//! the rotation spreads the input over every padded coordinate, so all of them must be
//! stored. The common embedding sizes are the worst case — 384, 768, 1536 and 3072 all
//! pad by a third, so d=768 cost exactly what d=1024 did.
//!
//! Instead the working dimension is `m·B` for a block size `B = 2^a ≥ 128` and a small
//! odd `m`, which covers those sizes exactly (768 = 3·256). Each round is then
//!
//!     x ← (Q_r ⊗ H_B) · D_r · x
//!
//! — signs, a Hadamard transform within each of the m blocks, and a Haar-random m×m
//! orthogonal matrix Q_r mixing the blocks column by column. The Kronecker product is
//! orthogonal, a single round already makes every output depend on every input, and
//! three rounds with independent D and Q pass the same moment and mixing tests as the
//! power-of-two path. Q costs m multiply-adds per coordinate on top of FWHT's log₂B,
//! which is why m is capped: past 15 the next power of two wins anyway. The 128 floor on
//! B keeps the working dimension a multiple of 128, which every vectorized scan kernel
//! needs, and keeps blocks large enough that the Hadamard does most of the mixing.

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

/// Smallest working dimension. See the note in `Rotation.init`.
pub const min_padded_dim: u32 = 4;

/// Smallest Hadamard block when the working dimension is not a power of two.
pub const min_block: u32 = 128;

/// Largest number of blocks mixed by the dense m×m stage.
pub const max_blocks: u32 = 15;

/// The working dimension for `dim`, and how it splits into Hadamard blocks.
pub const Shape = struct {
    /// Working dimension, `blocks * block`.
    padded: u32,
    /// Hadamard block size, a power of two.
    block: u32,
    /// Number of blocks, odd and at most `max_blocks`. 1 means the plain RHT.
    blocks: u32,
};

/// The smallest working dimension of the form `m·2^a` (m odd, m ≤ `max_blocks`,
/// 2^a ≥ `min_block`) that holds `dim`, preferring fewer blocks on a tie. Powers of
/// two, and anything at or below `min_block`, keep the plain power-of-two padding, so
/// their rotation — and every code ever produced for them — is unchanged.
pub fn shapeFor(dim: u32) Shape {
    std.debug.assert(dim >= 1 and dim <= 1 << 30);
    const pow2 = @max(min_padded_dim, std.math.ceilPowerOfTwoAssert(u32, dim));
    var best: Shape = .{ .padded = pow2, .block = pow2, .blocks = 1 };
    if (dim <= min_block or std.math.isPowerOfTwo(dim)) return best;

    var m: u32 = 3;
    while (m <= max_blocks) : (m += 2) {
        // Smallest power-of-two block, at least min_block, with m blocks covering dim.
        const need = (dim + m - 1) / m;
        const block = @max(min_block, std.math.ceilPowerOfTwoAssert(u32, need));
        const padded = m * block;
        if (padded < best.padded) best = .{ .padded = padded, .block = block, .blocks = m };
    }
    return best;
}

pub const Rotation = struct {
    /// Logical input dimension.
    dim: u32,
    /// Working dimension, `blocks * block` (see `shapeFor`).
    padded: u32,
    /// Hadamard block size: `padded` itself unless the dimension is split.
    block: u32,
    /// Blocks mixed by the dense stage; 1 for the plain power-of-two transform.
    blocks: u32,
    kind: Kind,
    rounds: u32,

    /// `.hadamard`: rounds × padded sign flips, then (when `blocks > 1`) rounds ×
    /// blocks × blocks mixing matrices, row-major. `.dense`: padded × padded.
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
        // Floored at 4 so the padded dimension is always a valid sphere dimension.
        // The coordinate density (1−t²)^((d−3)/2) needs d ≥ 3 to be bounded, and
        // below that the concentration argument the algorithm rests on is vacuous
        // anyway. Zero-padding a 1- or 2-dimensional vector into R⁴ is well defined
        // and costs three wasted codes; crashing on an assert three modules down is
        // not. Useful guarantees still need d in the hundreds.
        const shape = shapeFor(dim);
        const padded = shape.padded;
        const rounds = default_rounds;
        const mixing = rounds * shape.blocks * shape.blocks;

        const data = switch (kind) {
            .hadamard => try allocator.alloc(f32, rounds * padded + if (shape.blocks > 1) mixing else 0),
            .dense => try allocator.alloc(f32, @as(usize, padded) * padded),
        };
        errdefer allocator.free(data);

        var stream = rng.Philox.init(seed).stream(purpose);
        switch (kind) {
            .hadamard => {
                // Signs first, so a power-of-two rotation draws exactly what it always has.
                stream.fillSigns(data[0 .. rounds * padded]);
                if (shape.blocks > 1) {
                    const m = shape.blocks;
                    for (0..rounds) |r| {
                        buildDense(data[rounds * padded + r * m * m ..][0 .. m * m], m, &stream);
                    }
                }
            },
            .dense => buildDense(data, padded, &stream),
        }

        return .{
            .dim = dim,
            .padded = padded,
            .block = shape.block,
            .blocks = shape.blocks,
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
        assertDisjoint(src, dst);

        switch (self.kind) {
            .hadamard => {
                @memcpy(dst[0..src.len], src);
                @memset(dst[src.len..], 0);
                for (0..self.rounds) |round| {
                    applySigns(dst, self.signsFor(round));
                    self.blockHadamard(dst);
                    if (self.blocks > 1) mixBlocks(dst, self.mixingFor(round), self.blocks, self.block, false);
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
        assertDisjoint(src, dst);

        switch (self.kind) {
            .hadamard => {
                @memcpy(dst, src);
                // Π = H·D_{R−1} ⋯ H·D_0, so Πᵀ = D_0·H ⋯ D_{R−1}·H: reverse the
                // rounds and swap the order within each.
                var round = self.rounds;
                while (round > 0) {
                    round -= 1;
                    if (self.blocks > 1) mixBlocks(dst, self.mixingFor(round), self.blocks, self.block, true);
                    self.blockHadamard(dst);
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

    fn mixingFor(self: Rotation, round: usize) []const f32 {
        const m = self.blocks;
        return self.data[self.rounds * self.padded + round * m * m ..][0 .. m * m];
    }

    /// Normalized FWHT on each block. With one block this is the whole-vector transform.
    fn blockHadamard(self: Rotation, x: []f32) void {
        var b: usize = 0;
        while (b < self.padded) : (b += self.block) walshHadamard(x[b..][0..self.block]);
    }
};

/// `x ← (Q ⊗ I_B)·x`: block i becomes Σ_j Q[i][j]·block j, or Qᵀ with `transpose`.
///
/// Column chunks of `chunk` lanes are staged on the stack, so this is in place and
/// allocation-free like the rest of `apply`.
fn mixBlocks(x: []f32, q: []const f32, m: u32, block: u32, transpose: bool) void {
    const chunk = 16;
    const V = @Vector(chunk, f32);
    std.debug.assert(m <= max_blocks and block % chunk == 0);
    var in: [max_blocks]V = undefined;
    var c: usize = 0;
    while (c < block) : (c += chunk) {
        for (0..m) |j| in[j] = x[j * block + c ..][0..chunk].*;
        for (0..m) |i| {
            var acc: V = @splat(0);
            for (0..m) |j| {
                const w = if (transpose) q[j * m + i] else q[i * m + j];
                acc += @as(V, @splat(w)) * in[j];
            }
            x[i * block + c ..][0..chunk].* = acc;
        }
    }
}

/// Both transforms stage through `dst` and then work in place, so an aliasing
/// `src` would be partially overwritten before it is read. Caught here rather than
/// left to produce quietly wrong codes.
fn assertDisjoint(src: []const f32, dst: []const f32) void {
    const s = @intFromPtr(src.ptr);
    const d = @intFromPtr(dst.ptr);
    std.debug.assert(s + src.len * @sizeOf(f32) <= d or d + dst.len * @sizeOf(f32) <= s);
}

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

test "common embedding sizes are not padded at all" {
    // The sizes that motivated block mixing: each used to pad by a third.
    for ([_][3]u32{
        .{ 384, 384, 3 },   .{ 768, 768, 3 },   .{ 1536, 1536, 3 },
        .{ 3072, 3072, 3 }, .{ 640, 640, 5 },   .{ 1280, 1280, 5 },
        .{ 2560, 2560, 5 }, .{ 784, 896, 7 },   .{ 300, 384, 3 },
        .{ 200, 256, 1 },   .{ 1000, 1024, 1 }, .{ 3, 4, 1 },
        .{ 100, 128, 1 },   .{ 960, 1024, 1 },
    }) |case| {
        var rot = try Rotation.init(testing.allocator, case[0], .hadamard, 1, .rht_signs);
        defer rot.deinit();
        try testing.expectEqual(case[1], rot.padded);
        try testing.expectEqual(case[2], rot.blocks);
        // Every vectorized scan kernel needs a multiple of 128 above that size.
        if (rot.padded > min_block) try testing.expectEqual(@as(u32, 0), rot.padded % min_block);
    }
}

test "block rotations are orthogonal: norms, inverses and inner products" {
    var stream = rng.Philox.init(0xB10C).stream(.testing);
    for ([_]u32{ 300, 384, 768, 784 }) |dim| {
        for ([_]Kind{ .hadamard, .dense }) |kind| {
            var rot = try Rotation.init(testing.allocator, dim, kind, 0x77, .rht_signs);
            defer rot.deinit();
            const a = try testing.allocator.alloc(f32, dim);
            defer testing.allocator.free(a);
            const b = try testing.allocator.alloc(f32, dim);
            defer testing.allocator.free(b);
            const ra = try testing.allocator.alloc(f32, rot.padded);
            defer testing.allocator.free(ra);
            const rb = try testing.allocator.alloc(f32, rot.padded);
            defer testing.allocator.free(rb);
            const back = try testing.allocator.alloc(f32, rot.padded);
            defer testing.allocator.free(back);

            randomUnitVector(a, &stream);
            randomUnitVector(b, &stream);
            rot.apply(a, ra);
            rot.apply(b, rb);

            var norm: f64 = 0;
            for (ra) |v| norm += @as(f64, v) * v;
            try testing.expectApproxEqAbs(@as(f64, 1), norm, 1e-5);

            var before: f64 = 0;
            var after: f64 = 0;
            for (a, b) |x, y| before += @as(f64, x) * y;
            for (ra, rb) |x, y| after += @as(f64, x) * y;
            try testing.expectApproxEqAbs(before, after, 1e-5);

            rot.applyInverse(ra, back);
            for (a, back[0..dim]) |want, got| try testing.expectApproxEqAbs(want, got, 1e-5);
            for (back[dim..]) |v| try testing.expectApproxEqAbs(@as(f32, 0), v, 1e-5);
        }
    }
}

test "block rotations reproduce the sphere-coordinate moments" {
    // The same gate as the power-of-two path: coordinates of Πx must have variance 1/d
    // and fourth moment 3/(d(d+2)), and match the dense Haar reference.
    const dim: u32 = 384;
    const trials = 1500;
    var moments: [2]f64 = undefined;
    for ([_]Kind{ .hadamard, .dense }, 0..) |kind, ki| {
        var stream = rng.Philox.init(0x3333).stream(.testing);
        var rot = try Rotation.init(testing.allocator, dim, kind, 0x4444, .rht_signs);
        defer rot.deinit();
        try testing.expectEqual(dim, rot.padded);

        var src: [dim]f32 = undefined;
        var dst: [dim]f32 = undefined;
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
        const count: f64 = @floatFromInt(trials * dim);
        const d: f64 = @floatFromInt(dim);
        try testing.expectApproxEqRel(1.0 / d, sum2 / count, 0.02);
        try testing.expectApproxEqRel(3.0 / (d * (d + 2.0)), sum4 / count, 0.05);
        moments[ki] = sum4 / count;
    }
    try testing.expectApproxEqRel(moments[1], moments[0], 0.05);
}

test "block rotations mix a basis vector, including across blocks" {
    // The adversarial input for a Hadamard rotation, and for block mixing the worst
    // case: all energy in one block. After the rotation every block must carry its
    // share, and magnitudes must spread rather than sit at a few values.
    const dim: u32 = 768;
    var rot = try Rotation.init(testing.allocator, dim, .hadamard, 0xAAAA, .rht_signs);
    defer rot.deinit();
    try testing.expectEqual(@as(u32, 3), rot.blocks);

    var src = [_]f32{0} ** dim;
    src[5] = 1.0;
    var dst: [dim]f32 = undefined;
    rot.apply(&src, &dst);

    for (0..rot.blocks) |b| {
        var energy: f64 = 0;
        for (dst[b * rot.block ..][0..rot.block]) |v| energy += @as(f64, v) * v;
        // 1/3 each in expectation; a block left near-empty would mean no mixing.
        try testing.expect(energy > 0.15 and energy < 0.55);
    }

    var mean: f64 = 0;
    for (dst) |v| mean += @abs(v);
    mean /= dim;
    var variance: f64 = 0;
    for (dst) |v| {
        const dev = @abs(v) - mean;
        variance += dev * dev;
    }
    variance /= dim;
    try testing.expect(@sqrt(variance) / mean > 0.3);
}

test "tiny dimensions are floored to the minimum" {
    // Below this the coordinate density is not defined; padding up keeps the whole
    // pipeline working rather than tripping an assert deep in density.zig.
    for ([_]u32{ 1, 2, 3, 4 }) |dim| {
        var rot = try Rotation.init(testing.allocator, dim, .hadamard, 1, .rht_signs);
        defer rot.deinit();
        try testing.expectEqual(min_padded_dim, rot.padded);
    }
}

test "exact powers of two are not padded further" {
    for ([_]u32{ 8, 64, 128, 1024 }) |dim| {
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
