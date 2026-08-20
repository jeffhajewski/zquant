//! Scan kernel: score packed codes against a prepared query.
//!
//!     score(v) = Σ_j p_j · c[code_v[j]]
//!
//! The scalar codebook makes the per-dimension lookup table factor into a single
//! query-independent 16-entry table times a scalar (see `quant/packing.zig`), so the
//! table sits in one SIMD register for the whole scan and the work is a dot product
//! reducing along dimensions.
//!
//! ## Table lookup needs no inline assembly, but does need a barrier
//!
//! docs/DESIGN.md §4.2 assumed `tbl`/`vpshufb` would require per-architecture asm
//! behind a dispatch layer. It does not: LLVM pattern-matches the obvious elementwise
//! form — `inline for (0..16) |i| out[i] = table[idx[i]]` — into exactly one
//! instruction on every target that has one (`tbl` on aarch64, `pshufb` on SSSE3,
//! `vpshufb` on AVX2), and into scalar loads where none exists.
//!
//! **But only if the result is not immediately widened.** Inlined into this kernel,
//! LLVM folds each `extractelement` into the consumer's `sext` and never forms the
//! vector-build pattern it needs to recognize — producing 32 scalar `umov`/`bfxil`
//! pairs per chunk instead of one `tbl`. Measured cost of getting this wrong: the
//! whole kernel ran at 1.5 GB/s, only 1.2× faster than an unquantized f32 brute-force
//! scan.
//!
//! `optimizationBarrier` is an empty `asm` with a vector register constraint. It
//! emits no instruction and blocks the fold. This is a fragile,
//! compiler-version-dependent trick: `bench/scan_bench.zig` is what guards it, since
//! a regression is silent and costs ~4× rather than breaking a test.
//!
//! One semantic caveat: `pshufb` zeroes a lane whose index has bit 7 set, `tbl`
//! zeroes for index ≥ 16, and the scalar form would index out of bounds. All three
//! agree only while indices stay in 0..15. Codes are 4-bit here, so they do, and
//! `scoreInt8` asserts the bit-width rather than trusting it.
//!
//! ## Precision
//!
//! `scoreInt8` quantizes both the centroid table and the query to int8. That is the
//! opt-in fast path of §4.2, and it adds error the paper does not analyse.
//! `scoreExact` is the f32 reference and the oracle it is tested against.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const Layout = @import("../quant/packing.zig").Layout;

/// The 16 centroids, quantized to int8. Query-independent, built once per index.
pub const Table = struct {
    values: @Vector(16, i8),
    /// centroid ≈ values[k] * scale
    scale: f32,

    pub fn init(centroids: []const f32) Table {
        std.debug.assert(centroids.len <= 16);

        var magnitude: f32 = 0;
        for (centroids) |c| magnitude = @max(magnitude, @abs(c));
        // A codebook of all zeros is degenerate but reachable at bits=0.
        const scale = if (magnitude > 0) magnitude / 127.0 else 1.0;

        var values: [16]i8 = @splat(0);
        for (centroids, 0..) |c, k| {
            values[k] = @intFromFloat(@round(std.math.clamp(c / scale, -127.0, 127.0)));
        }
        return .{ .values = values, .scale = scale };
    }
};

/// A query, rotated and de-interleaved to match the code packing.
///
/// Byte `i` of a code block holds `groups = 8/bits` consecutive dimensions, so
/// extracting bit-field `k` from a 16-byte load yields the codes for dimensions
/// congruent to `k` modulo `groups`. The query is split the same way — a fixed
/// permutation applied once per query, not once per vector.
///
/// At bits=4 this is the even/odd split; at bits=2 it is four streams, at bits=1
/// eight.
pub const Query = struct {
    /// `groups` contiguous streams of `stride` values each.
    data: []i8,
    groups: usize,
    stride: usize,
    /// p_j ≈ data[(j % groups) * stride + j / groups] * scale
    scale: f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, dim: u32, bits: u6) Allocator.Error!Query {
        const groups = 8 / @as(usize, bits);
        std.debug.assert(dim % groups == 0);
        return .{
            .data = try allocator.alloc(i8, dim),
            .groups = groups,
            .stride = dim / groups,
            .scale = 1.0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Query) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }

    /// Quantize and de-interleave a rotated query. Scale is chosen per query from its
    /// own magnitude, so a small query does not lose precision to a fixed range.
    pub fn load(self: *Query, rotated: []const f32) void {
        std.debug.assert(rotated.len == self.data.len);

        var magnitude: f32 = 0;
        for (rotated) |v| magnitude = @max(magnitude, @abs(v));
        self.scale = if (magnitude > 0) magnitude / 127.0 else 1.0;
        const inverse = 1.0 / self.scale;

        for (0..self.groups) |k| {
            const stream = self.data[k * self.stride ..][0..self.stride];
            for (stream, 0..) |*dst, m| {
                const v = rotated[m * self.groups + k] * inverse;
                dst.* = @intFromFloat(@round(std.math.clamp(v, -127.0, 127.0)));
            }
        }
    }
};

/// Forces `v` to be materialized in a vector register, blocking the `sext` fold that
/// otherwise destroys the shuffle pattern (see the module comment). Emits nothing.
///
/// Architectures without a known vector-register constraint simply skip it and get
/// the scalar lookup — correct, just slower.
inline fn optimizationBarrier(v: @Vector(16, i8)) @Vector(16, i8) {
    var x = v;
    switch (builtin.cpu.arch) {
        .aarch64, .aarch64_be => asm ("" : [x] "+w" (x)),
        .x86_64, .x86 => asm ("" : [x] "+x" (x)),
        else => {},
    }
    return x;
}

/// 16-byte table lookup. One `tbl`/`pshufb`/`vpshufb`; scalar loads where absent.
inline fn lookup(table: @Vector(16, i8), idx: @Vector(16, u8)) @Vector(16, i8) {
    var out: @Vector(16, i8) = @splat(0);
    inline for (0..16) |i| out[i] = table[idx[i]];
    return optimizationBarrier(out);
}

/// Vectorized score for a comptime-known bit-width.
///
/// One 16-byte load carries `128/bits` codes: `groups` bit-fields of 16 lanes each.
/// Products are accumulated two groups at a time in i16 (`smlal`) before widening —
/// two products reach 2·127·127 = 32258, just inside i16's 32767, so pairs are the
/// most that can be batched at full range.
fn scoreBits(comptime bits: u6, table: Table, query: Query, codes: []const u8, dim: u32) f32 {
    const groups = comptime 8 / @as(usize, bits);
    const mask_value = comptime (@as(u8, 1) << @as(u3, @intCast(bits))) - 1;
    const field_mask: @Vector(16, u8) = @splat(mask_value);

    const chunks = @as(usize, dim) * bits / 128;
    var acc: @Vector(16, i32) = @splat(0);

    for (0..chunks) |ch| {
        const packed_codes: @Vector(16, u8) = codes[ch * 16 ..][0..16].*;

        comptime var pair: usize = 0;
        inline while (pair < groups / 2) : (pair += 1) {
            var chunk_acc: @Vector(16, i16) = @splat(0);
            inline for (0..2) |half| {
                const k = pair * 2 + half;
                const shift: @Vector(16, u3) = @splat(@intCast(k * bits));
                const idx = (packed_codes >> shift) & field_mask;
                const w: @Vector(16, i8) =
                    query.data[k * query.stride + ch * 16 ..][0..16].*;
                chunk_acc +%= @as(@Vector(16, i16), lookup(table.values, idx)) *%
                    @as(@Vector(16, i16), w);
            }
            acc += @as(@Vector(16, i32), chunk_acc);
        }
    }

    const total: f32 = @floatFromInt(@reduce(.Add, acc));
    return total * table.scale * query.scale;
}

/// Vectorized score. `bits` must satisfy `canVectorize`.
pub fn scoreInt8(bits: u6, table: Table, query: Query, codes: []const u8, dim: u32) f32 {
    std.debug.assert(codes.len * 8 >= @as(usize, dim) * bits);
    return switch (bits) {
        1 => scoreBits(1, table, query, codes, dim),
        2 => scoreBits(2, table, query, codes, dim),
        4 => scoreBits(4, table, query, codes, dim),
        // 3-bit codes do not divide a byte, so a 16-byte load carries a ragged
        // number of them. Deliberately unsupported rather than approximated by
        // padding into nibbles, which measurement showed is dominated by simply
        // using one more bit of MSE (see docs/notes.md).
        else => unreachable,
    };
}

/// Exact f32 reference. Correct for any bit-width and dimension.
pub fn scoreExact(
    layout: Layout,
    centroids: []const f32,
    rotated: []const f32,
    codes: []const u8,
) f32 {
    var acc: f64 = 0;
    for (rotated, 0..) |p, j| {
        acc += @as(f64, p) * centroids[layout.codeAt(codes, j)];
    }
    return @floatCast(acc);
}

/// Whether the vectorized path applies.
///
/// Bit-widths that divide a byte (1, 2, 4) keep every code inside one byte, so
/// extraction is a uniform shift-and-mask. 3, 5, 6, 7 do not and fall back to
/// `scoreExact`.
pub fn canVectorize(layout: Layout) bool {
    const divides_byte = layout.bits == 1 or layout.bits == 2 or layout.bits == 4;
    return divides_byte and (@as(usize, layout.dim) * layout.bits) % 128 == 0;
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;
const Codebook = @import("../quant/codebook.zig").Codebook;
const Density = @import("../math/density.zig").Density;

fn setup(
    dim: u32,
    bits: u6,
    seed: u64,
    out_codes: []u8,
    out_rotated: []f32,
) !Codebook {
    var cb = try Codebook.init(testing.allocator, Density.sphereCoord(dim), bits);
    errdefer cb.deinit();

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const sigma = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));

    const raw = try testing.allocator.alloc(f32, dim);
    defer testing.allocator.free(raw);
    for (raw) |*v| v.* = random.floatNorm(f32) * sigma;

    const layout = Layout.init(dim, bits);
    const codes = try testing.allocator.alloc(u8, dim);
    defer testing.allocator.free(codes);
    cb.encodeSlice(raw, codes);
    layout.pack(codes, out_codes);

    for (out_rotated) |*v| v.* = random.floatNorm(f32) * sigma;
    return cb;
}

test "int8 kernel tracks the exact reference" {
    // The int8 path is an approximation, so the test is agreement within a bounded
    // relative error, not equality. The bound comes from quantizing both the table
    // and the query to 8 bits.
    for ([_]u32{ 32, 128, 256, 1024 }) |dim| {
        const layout = Layout.init(dim, 4);
        const codes = try testing.allocator.alloc(u8, layout.stride());
        defer testing.allocator.free(codes);
        const rotated = try testing.allocator.alloc(f32, dim);
        defer testing.allocator.free(rotated);

        var cb = try setup(dim, 4, 0x5EED + dim, codes, rotated);
        defer cb.deinit();

        var query = try Query.init(testing.allocator, dim, 4);
        defer query.deinit();
        query.load(rotated);

        const table = Table.init(cb.centroids);
        const fast = scoreInt8(4, table, query, codes, dim);
        const exact = scoreExact(layout, cb.centroids, rotated, codes);

        // Scores are ~N(0, ·) with magnitude around ‖p‖·‖c‖/√d; compare against the
        // scale of the terms rather than the (possibly near-zero) total.
        var magnitude: f64 = 0;
        for (rotated) |p| magnitude += @abs(p) * 0.05;
        const tolerance: f32 = @floatCast(magnitude * 0.02 + 1e-6);
        try testing.expectApproxEqAbs(exact, fast, tolerance);
    }
}

test "int8 kernel error stays small in aggregate" {
    // A single comparison can be lucky. This measures the relative error over many
    // independent draws, which is what recall actually depends on.
    const dim: u32 = 512;
    const layout = Layout.init(dim, 4);

    var cb = try Codebook.init(testing.allocator, Density.sphereCoord(dim), 4);
    defer cb.deinit();
    const table = Table.init(cb.centroids);

    var query = try Query.init(testing.allocator, dim, 4);
    defer query.deinit();

    const raw = try testing.allocator.alloc(f32, dim);
    defer testing.allocator.free(raw);
    const codes_unpacked = try testing.allocator.alloc(u8, dim);
    defer testing.allocator.free(codes_unpacked);
    const codes = try testing.allocator.alloc(u8, layout.stride());
    defer testing.allocator.free(codes);
    const rotated = try testing.allocator.alloc(f32, dim);
    defer testing.allocator.free(rotated);

    var prng = std.Random.DefaultPrng.init(99);
    const random = prng.random();
    const sigma = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));

    var squared_error: f64 = 0;
    var squared_signal: f64 = 0;
    const trials = 400;
    for (0..trials) |_| {
        for (raw) |*v| v.* = random.floatNorm(f32) * sigma;
        cb.encodeSlice(raw, codes_unpacked);
        layout.pack(codes_unpacked, codes);
        for (rotated) |*v| v.* = random.floatNorm(f32) * sigma;
        query.load(rotated);

        const fast = scoreInt8(4, table, query, codes, dim);
        const exact = scoreExact(layout, cb.centroids, rotated, codes);
        const err = @as(f64, fast) - exact;
        squared_error += err * err;
        squared_signal += exact * exact;
    }
    // Under 1% RMS relative to the signal: far below the quantizer's own distortion.
    const relative = @sqrt(squared_error / squared_signal);
    try testing.expect(relative < 0.01);
}

test "table quantization preserves ordering and sign" {
    for (1..5) |bits| {
        var cb = try Codebook.init(testing.allocator, Density.gauss(1.0), @intCast(bits));
        defer cb.deinit();
        const table = Table.init(cb.centroids);

        // Ascending centroids must stay ascending after int8 rounding, or the
        // codebook's monotonicity is broken and encode/score disagree.
        for (1..cb.levels()) |k| {
            try testing.expect(table.values[k] >= table.values[k - 1]);
        }
        // And reconstruct to something close.
        for (cb.centroids, 0..) |c, k| {
            const back = @as(f32, @floatFromInt(table.values[k])) * table.scale;
            try testing.expectApproxEqAbs(c, back, table.scale);
        }
    }
}

test "query round trips through quantization and the parity split" {
    const dim: u32 = 64;
    var query = try Query.init(testing.allocator, dim, 4);
    defer query.deinit();

    var rotated: [dim]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(4);
    for (&rotated) |*v| v.* = prng.random().floatNorm(f32);
    query.load(&rotated);

    for (0..dim) |j| {
        const stored = query.data[(j % query.groups) * query.stride + j / query.groups];
        try testing.expectApproxEqAbs(
            rotated[j],
            @as(f32, @floatFromInt(stored)) * query.scale,
            query.scale,
        );
    }
}

test "lookup matches a scalar table index" {
    // Guards the one construct whose lowering differs per architecture.
    var prng = std.Random.DefaultPrng.init(21);
    const random = prng.random();
    for (0..64) |_| {
        var table_values: [16]i8 = undefined;
        var indices: [16]u8 = undefined;
        for (&table_values) |*v| v.* = random.intRangeAtMost(i8, -128, 127);
        for (&indices) |*v| v.* = random.uintAtMost(u8, 15);

        const got = lookup(table_values, indices);
        for (0..16) |i| try testing.expectEqual(table_values[indices[i]], got[i]);
    }
}

test "zero query and zero codes score zero" {
    const dim: u32 = 64;
    var cb = try Codebook.init(testing.allocator, Density.sphereCoord(dim), 4);
    defer cb.deinit();
    const table = Table.init(cb.centroids);

    var query = try Query.init(testing.allocator, dim, 4);
    defer query.deinit();
    const zeros = [_]f32{0} ** dim;
    query.load(&zeros);

    const layout = Layout.init(dim, 4);
    const codes = try testing.allocator.alloc(u8, layout.stride());
    defer testing.allocator.free(codes);
    @memset(codes, 0);

    try testing.expectEqual(@as(f32, 0), scoreInt8(4, table, query, codes, dim));
}

test "canVectorize gates on bit-width and dimension" {
    // Widths that divide a byte are supported...
    try testing.expect(canVectorize(Layout.init(1024, 4)));
    try testing.expect(canVectorize(Layout.init(1024, 2)));
    try testing.expect(canVectorize(Layout.init(1024, 1)));
    try testing.expect(canVectorize(Layout.init(32, 4)));
    // ...and those that do not are rejected rather than approximated.
    try testing.expect(!canVectorize(Layout.init(1024, 3)));
    try testing.expect(!canVectorize(Layout.init(1024, 5)));
    // A chunk is 16 bytes, so dim*bits must fill whole chunks.
    try testing.expect(!canVectorize(Layout.init(48, 4)));
    try testing.expect(!canVectorize(Layout.init(32, 2)));
}

test "every vectorizable bit-width tracks the exact reference" {
    // Extends the b=4 agreement test across all supported widths. A wrong shift or
    // stream mapping at b=1 or b=2 would produce plausible but consistently wrong
    // scores, exactly the failure mode the parity split had at b=4.
    const allocator = testing.allocator;
    for ([_]u6{ 1, 2, 4 }) |bits| {
        const dim: u32 = 1024;
        const layout = Layout.init(dim, bits);
        try testing.expect(canVectorize(layout));

        var cb = try Codebook.init(allocator, Density.sphereCoord(dim), bits);
        defer cb.deinit();
        const table = Table.init(cb.centroids);

        var query = try Query.init(allocator, dim, bits);
        defer query.deinit();

        const raw = try allocator.alloc(f32, dim);
        defer allocator.free(raw);
        const unpacked = try allocator.alloc(u8, dim);
        defer allocator.free(unpacked);
        const stored = try allocator.alloc(u8, layout.stride());
        defer allocator.free(stored);
        const rotated = try allocator.alloc(f32, dim);
        defer allocator.free(rotated);

        var prng = std.Random.DefaultPrng.init(0xB175 + @as(u64, bits));
        const random = prng.random();
        const sigma = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));

        var squared_error: f64 = 0;
        var squared_signal: f64 = 0;
        for (0..100) |_| {
            for (raw) |*v| v.* = random.floatNorm(f32) * sigma;
            cb.encodeSlice(raw, unpacked);
            layout.pack(unpacked, stored);
            for (rotated) |*v| v.* = random.floatNorm(f32) * sigma;
            query.load(rotated);

            const fast = scoreInt8(bits, table, query, stored, dim);
            const exact = scoreExact(layout, cb.centroids, rotated, stored);
            const err = @as(f64, fast) - exact;
            squared_error += err * err;
            squared_signal += exact * exact;
        }
        try testing.expect(@sqrt(squared_error / squared_signal) < 0.02);
    }
}

test "exact reference agrees with a naive unpacked dot product" {
    // scoreExact is the oracle for everything above, so it gets its own check.
    const dim: u32 = 96;
    const layout = Layout.init(dim, 4);
    const codes = try testing.allocator.alloc(u8, layout.stride());
    defer testing.allocator.free(codes);
    const rotated = try testing.allocator.alloc(f32, dim);
    defer testing.allocator.free(rotated);

    var cb = try setup(dim, 4, 7, codes, rotated);
    defer cb.deinit();

    const unpacked = try testing.allocator.alloc(u8, dim);
    defer testing.allocator.free(unpacked);
    layout.unpack(codes, unpacked);

    var naive: f64 = 0;
    for (rotated, unpacked) |p, code| naive += @as(f64, p) * cb.centroids[code];

    try testing.expectApproxEqAbs(
        @as(f32, @floatCast(naive)),
        scoreExact(layout, cb.centroids, rotated, codes),
        1e-6,
    );
}
