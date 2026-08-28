//! Scan kernel: score packed codes against a prepared query.
//!
//!     score(v) = Σ_j p_j · c[code_v[j]]
//!
//! The scalar codebook makes the per-dimension lookup table factor into a single
//! query-independent 16-entry table times a scalar (see `quant/packing.zig`), so the
//! table sits in one SIMD register for the whole scan and the work is a dot product
//! reducing along dimensions.
//!
//! ## Table lookup
//!
//! `simd/shuffle.zig` owns the one-instruction byte-table lookup this kernel is built
//! on. Under Zig 0.15 that was portable source LLVM pattern-matched; under 0.16 it is
//! inline assembly again. Either way the cost of getting it wrong is large — when the
//! lookup failed to vectorize, the kernel ran at 1.5 GB/s against 20.9.
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
const packing = @import("../quant/packing.zig");
const Layout = packing.Layout;
const bitmask = @import("bitmask.zig");
const shuffle = @import("shuffle.zig");
const dot = @import("dot.zig");

/// The 16 centroids, quantized to int8. Query-independent, built once per index.
pub const Table = struct {
    values: @Vector(16, i8),
    /// centroid ≈ values[k] * scale
    scale: f32,

    /// Build from a codebook of at most 16 levels.
    ///
    /// The caller must check `canVectorize` first — a wider codebook has no
    /// representation here, and this is only reachable at all because the shuffle
    /// instruction indexes 16 bytes (`max_table_bits`).
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

/// Independent accumulator registers, enough to cover `sdot` latency without
/// exhausting the register file.
const accumulators = 4;

/// Placeholder for indexes that will not take the vectorized path, so callers need
/// not make the field optional. Never read: the scalar scan does not consult it.
pub fn unusedTable() Table {
    return .{ .values = @splat(0), .scale = 1.0 };
}

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

    /// `groups` follows the layout: a sequential layout interleaves the query by
    /// `8/bits`, while a bit-plane layout stores 16 consecutive dimensions per group
    /// and needs no interleaving at all.
    /// A query in plain dimension order, for the expanded residency.
    pub fn initSequential(allocator: Allocator, dim: u32) Allocator.Error!Query {
        return .{
            .data = try allocator.alloc(i8, dim),
            .groups = 1,
            .stride = dim,
            .scale = 1.0,
            .allocator = allocator,
        };
    }

    pub fn init(allocator: Allocator, layout: Layout) Allocator.Error!Query {
        const dim = layout.dim;
        const groups: usize = switch (layout.kind()) {
            .sequential => 8 / @as(usize, layout.bits),
            .bit_plane => 1,
        };
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

/// 16-byte table lookup. See `simd/shuffle.zig` for why this is assembly.
inline fn lookup(table: @Vector(16, i8), idx: @Vector(16, u8)) @Vector(16, i8) {
    return shuffle.table16(table, idx);
}

/// Vectorized score over a sequential layout, for a comptime-known bit-width.
///
/// One 16-byte load carries `128/bits` codes: `groups` bit-fields of 16 lanes each.
/// Products accumulate two groups at a time in i16 (`smlal`) before widening — two
/// reach 2·127·127 = 32258, just inside i16's 32767, so pairs are the most that can
/// be batched at full range.
fn scoreBits(comptime bits: u6, table: Table, query: Query, codes: []const u8, dim: u32) f32 {
    const groups = comptime 8 / @as(usize, bits);
    const mask_value = comptime (@as(u8, 1) << @as(u3, @intCast(bits))) - 1;
    const field_mask: @Vector(16, u8) = @splat(mask_value);

    const chunks = @as(usize, dim) * bits / 128;

    // One accumulator per bit-field group. `sdot` has multi-cycle latency and reads
    // the accumulator it writes, so a single register serializes the loop. The slot
    // must be a *comptime* index: selecting it with a runtime expression spills the
    // accumulators to the stack and cost 42% when tried.
    var acc: [groups]dot.Acc = @splat(@splat(0));

    for (0..chunks) |ch| {
        const packed_codes: @Vector(16, u8) = codes[ch * 16 ..][0..16].*;
        inline for (0..groups) |k| {
            const shift: @Vector(16, u3) = @splat(@intCast(k * bits));
            const idx = (packed_codes >> shift) & field_mask;
            const w: @Vector(16, i8) = query.data[k * query.stride + ch * 16 ..][0..16].*;
            acc[k] = dot.accumulate(acc[k], lookup(table.values, idx), w);
        }
    }

    var folded: dot.Acc = acc[0];
    inline for (1..groups) |i| folded += acc[i];
    const sum: f32 = @floatFromInt(dot.total(folded));
    return sum * table.scale * query.scale;
}

/// Assemble 16 code indices from `bits` bit-planes.
///
/// Each plane contributes its weight where its bit is set. Roughly 3 ops per plane
/// against 2 for a whole sequential extraction, which is why bit-plane is used only
/// where shift-and-mask cannot reach — a 3-bit code straddles a byte.
inline fn planeIndex(comptime bits: u6, planes: []const u8) @Vector(16, u8) {
    var idx: @Vector(16, u8) = @splat(0);
    inline for (0..bits) |p| {
        idx |= bitmask.expandWeighted(planes[p * 2 ..][0..2].*, 1 << p);
    }
    return idx;
}

/// Vectorized score over a bit-plane layout.
fn scorePlanes(comptime bits: u6, table: Table, query: Query, codes: []const u8, dim: u32) f32 {
    const group_bytes = comptime @as(usize, bits) * (packing.plane_group / 8);
    const groups = dim / packing.plane_group;
    var acc: [accumulators]dot.Acc = @splat(@splat(0));

    // Unrolled so the accumulator index is comptime; a runtime index spills them.
    var g: usize = 0;
    while (g + accumulators <= groups) : (g += accumulators) {
        inline for (0..accumulators) |u| {
            const idx = planeIndex(bits, codes[(g + u) * group_bytes ..][0..group_bytes]);
            const w: @Vector(16, i8) = query.data[(g + u) * 16 ..][0..16].*;
            acc[u] = dot.accumulate(acc[u], lookup(table.values, idx), w);
        }
    }
    while (g < groups) : (g += 1) {
        const idx = planeIndex(bits, codes[g * group_bytes ..][0..group_bytes]);
        const w: @Vector(16, i8) = query.data[g * 16 ..][0..16].*;
        acc[0] = dot.accumulate(acc[0], lookup(table.values, idx), w);
    }

    var folded: dot.Acc = acc[0];
    inline for (1..accumulators) |i| folded += acc[i];
    const sum: f32 = @floatFromInt(dot.total(folded));
    return sum * table.scale * query.scale;
}

/// Score against codes already dequantized to int8, one byte per coordinate.
///
/// No unpacking and no table lookup: the loop is loads and `sdot`. That is the whole
/// point of the expanded residency — it spends about twice the memory of packed codes
/// to remove roughly half the instructions.
///
/// `query` must be built with `Query.initSequential`, since expanded values are stored
/// in plain dimension order rather than interleaved by bit-field.
pub fn scoreExpanded(values: []const i8, query: Query, table_scale: f32, dim: u32) f32 {
    std.debug.assert(query.groups == 1);
    std.debug.assert(values.len >= dim);
    std.debug.assert(dim % 16 == 0);

    var acc: [accumulators]dot.Acc = @splat(@splat(0));
    const chunks = dim / 16;

    var ch: usize = 0;
    while (ch + accumulators <= chunks) : (ch += accumulators) {
        inline for (0..accumulators) |u| {
            const c: @Vector(16, i8) = values[(ch + u) * 16 ..][0..16].*;
            const w: @Vector(16, i8) = query.data[(ch + u) * 16 ..][0..16].*;
            acc[u] = dot.accumulate(acc[u], c, w);
        }
    }
    while (ch < chunks) : (ch += 1) {
        const c: @Vector(16, i8) = values[ch * 16 ..][0..16].*;
        const w: @Vector(16, i8) = query.data[ch * 16 ..][0..16].*;
        acc[0] = dot.accumulate(acc[0], c, w);
    }

    var folded: dot.Acc = acc[0];
    inline for (1..accumulators) |i| folded += acc[i];
    const sum: f32 = @floatFromInt(dot.total(folded));
    return sum * table_scale * query.scale;
}

/// Score one stored vector against `Q` queries in a single pass.
///
/// The one-query kernel issues two loads per `SDOT` — the code chunk and the query
/// chunk — which pins it at one `SDOT` per cycle on the load ports, and that is the
/// ~48 G dim/s ceiling the scan sits at regardless of dimension or residency.
///
/// Batching alone did not move it (measured 1.00×) because the batch loop called the
/// one-query kernel once per query, re-walking the vector and reloading both operands
/// each time. It amortized cache lines, which were never the constraint.
///
/// Here the code chunk is loaded once and reused across all `Q` queries, so the ratio
/// falls to (1 + Q)/Q loads per `SDOT` — 1.25 at Q=4 against 2.0. The `Q` independent
/// accumulator chains also supply the instruction-level parallelism that the chunk
/// unroll provides in the single-query path, so there is no separate unroll here.
pub fn scoreExpandedMulti(
    comptime Q: usize,
    values: []const i8,
    queries: []const Query,
    table_scale: f32,
    dim: u32,
    out: []f32,
) void {
    std.debug.assert(queries.len >= Q and out.len >= Q);
    std.debug.assert(values.len >= dim);
    std.debug.assert(dim % 16 == 0);

    // Hoist the query base pointers. Reading `queries[q].data` inside the loop makes
    // the slice header itself a load — two per query per chunk instead of one, which
    // is worse than the single-query kernel it is meant to beat.
    var qp: [Q][*]const i8 = undefined;
    inline for (0..Q) |q| qp[q] = queries[q].data.ptr;
    const vp: [*]const i8 = values.ptr;

    var acc: [Q]dot.Acc = @splat(@splat(0));
    const chunks = dim / 16;

    var ch: usize = 0;
    while (ch < chunks) : (ch += 1) {
        const c: @Vector(16, i8) = vp[ch * 16 ..][0..16].*;
        inline for (0..Q) |q| {
            const w: @Vector(16, i8) = qp[q][ch * 16 ..][0..16].*;
            acc[q] = dot.accumulate(acc[q], c, w);
        }
    }

    inline for (0..Q) |q| {
        const sum: f32 = @floatFromInt(dot.total(acc[q]));
        out[q] = sum * table_scale * queries[q].scale;
    }
}

/// Widest query group the compact kernels can hold without spilling.
///
/// A sequential layout keeps one accumulator per bit-field group, so the multi-query
/// form needs `Q * groups` of them. aarch64 has 32 vector registers and the loop also
/// needs the code chunk, the unpacked values and a query chunk, so the accumulator
/// budget is 16. Exceeding it spills to the stack, which cost 42% when it happened
/// to the single-query kernel.
pub fn multiWidth(layout: Layout) usize {
    return switch (layout.kind()) {
        .sequential => @max(1, 16 / (8 / @as(usize, layout.bits))),
        .bit_plane => 4,
    };
}

/// Score one packed vector against `Q` queries in a single pass.
///
/// The unpack — shift, mask, and table lookup — does not depend on the query, so the
/// single-query kernel redoes all of it for every query in a batch. Here it happens
/// once per bit-field group and feeds `Q` SDOTs, which is a larger saving than the
/// expanded kernel gets: that one only amortizes a load, this amortizes the whole
/// unpack.
fn scoreBitsMulti(
    comptime bits: u6,
    comptime Q: usize,
    table: Table,
    queries: []const Query,
    codes: []const u8,
    dim: u32,
    out: []f32,
) void {
    const groups = comptime 8 / @as(usize, bits);
    const mask_value = comptime (@as(u8, 1) << @as(u3, @intCast(bits))) - 1;
    const field_mask: @Vector(16, u8) = @splat(mask_value);
    const chunks = @as(usize, dim) * bits / 128;

    var qp: [Q][*]const i8 = undefined;
    inline for (0..Q) |q| qp[q] = queries[q].data.ptr;
    const qstride = queries[0].stride;

    var acc: [Q][groups]dot.Acc = @splat(@splat(@splat(0)));

    for (0..chunks) |ch| {
        const packed_codes: @Vector(16, u8) = codes[ch * 16 ..][0..16].*;
        inline for (0..groups) |k| {
            const shift: @Vector(16, u3) = @splat(@intCast(k * bits));
            const idx = (packed_codes >> shift) & field_mask;
            // Hoisted out of the query loop: this is the whole point of the kernel.
            const vals = lookup(table.values, idx);
            inline for (0..Q) |q| {
                const w: @Vector(16, i8) = qp[q][k * qstride + ch * 16 ..][0..16].*;
                acc[q][k] = dot.accumulate(acc[q][k], vals, w);
            }
        }
    }

    inline for (0..Q) |q| {
        var folded: dot.Acc = acc[q][0];
        inline for (1..groups) |i| folded += acc[q][i];
        const sum: f32 = @floatFromInt(dot.total(folded));
        out[q] = sum * table.scale * queries[q].scale;
    }
}

/// Multi-query compact score. `layout` must satisfy `canVectorize`, and `queries`
/// and `out` must hold at least `multiWidth(layout)` entries.
pub fn scoreInt8Multi(
    layout: Layout,
    table: Table,
    queries: []const Query,
    codes: []const u8,
    dim: u32,
    out: []f32,
) void {
    std.debug.assert(codes.len * 8 >= @as(usize, dim) * layout.bits);
    switch (layout.kind()) {
        .sequential => switch (layout.bits) {
            1 => scoreBitsMulti(1, 2, table, queries, codes, dim, out),
            2 => scoreBitsMulti(2, 4, table, queries, codes, dim, out),
            4 => scoreBitsMulti(4, 8, table, queries, codes, dim, out),
            else => unreachable,
        },
        // Bit-plane assembles its indices from planes rather than by shift-and-mask;
        // it keeps the per-query kernel until that is worth restructuring too.
        .bit_plane => for (0..multiWidth(layout)) |q| {
            out[q] = scoreInt8(layout, table, queries[q], codes, dim);
        },
    }
}

/// Vectorized score. `bits` must satisfy `canVectorize`.
pub fn scoreInt8(layout: Layout, table: Table, query: Query, codes: []const u8, dim: u32) f32 {
    std.debug.assert(codes.len * 8 >= @as(usize, dim) * layout.bits);
    return switch (layout.kind()) {
        .sequential => switch (layout.bits) {
            1 => scoreBits(1, table, query, codes, dim),
            2 => scoreBits(2, table, query, codes, dim),
            4 => scoreBits(4, table, query, codes, dim),
            else => unreachable,
        },
        .bit_plane => switch (layout.bits) {
            3 => scorePlanes(3, table, query, codes, dim),
            else => unreachable,
        },
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

/// Largest codebook a single byte-shuffle can hold.
///
/// `tbl`/`pshufb` index a 16-byte register, so 16 levels — four bits — is a hard
/// ceiling on the vectorized path, independent of how codes are laid out. Wider
/// codebooks need multiple shuffles and a blend, which has not been shown to be
/// worth it: `prod` at b=5 (four code bits) already reaches 0.995 recall.
pub const max_table_bits: u6 = 4;

/// Whether the vectorized path applies.
///
/// Two independent conditions: the codebook must fit one shuffle table, and the
/// dimension must divide into whole units of work. Sequential needs whole 16-byte
/// chunks; bit-plane needs an even number of 16-code groups.
pub fn canVectorize(layout: Layout) bool {
    if (layout.bits > max_table_bits) return false;
    return switch (layout.kind()) {
        .sequential => (@as(usize, layout.dim) * layout.bits) % 128 == 0,
        .bit_plane => layout.dim % (2 * packing.plane_group) == 0,
    };
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

        var query = try Query.init(testing.allocator, Layout.init(dim, 4));
        defer query.deinit();
        query.load(rotated);

        const table = Table.init(cb.centroids);
        const fast = scoreInt8(Layout.init(dim, 4), table, query, codes, dim);
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

    var query = try Query.init(testing.allocator, Layout.init(dim, 4));
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

        const fast = scoreInt8(Layout.init(dim, 4), table, query, codes, dim);
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

        const values: [16]i8 = table.values;
        // Ascending centroids must stay ascending after int8 rounding, or the
        // codebook's monotonicity is broken and encode/score disagree.
        for (1..cb.levels()) |k| {
            try testing.expect(values[k] >= values[k - 1]);
        }
        // And reconstruct to something close.
        for (cb.centroids, 0..) |c, k| {
            const back = @as(f32, @floatFromInt(values[k])) * table.scale;
            try testing.expectApproxEqAbs(c, back, table.scale);
        }
    }
}

test "query round trips through quantization and the parity split" {
    const dim: u32 = 64;
    var query = try Query.init(testing.allocator, Layout.init(dim, 4));
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

        const got: [16]i8 = lookup(table_values, indices);
        for (got, indices) |g, i| try testing.expectEqual(table_values[i], g);
    }
}

test "zero query and zero codes score zero" {
    const dim: u32 = 64;
    var cb = try Codebook.init(testing.allocator, Density.sphereCoord(dim), 4);
    defer cb.deinit();
    const table = Table.init(cb.centroids);

    var query = try Query.init(testing.allocator, Layout.init(dim, 4));
    defer query.deinit();
    const zeros = [_]f32{0} ** dim;
    query.load(&zeros);

    const layout = Layout.init(dim, 4);
    const codes = try testing.allocator.alloc(u8, layout.stride());
    defer testing.allocator.free(codes);
    @memset(codes, 0);

    try testing.expectEqual(@as(f32, 0), scoreInt8(Layout.init(dim, 4), table, query, codes, dim));
}

test "canVectorize gates on bit-width and dimension" {
    // Widths that divide a byte are supported...
    try testing.expect(canVectorize(Layout.init(1024, 4)));
    try testing.expect(canVectorize(Layout.init(1024, 2)));
    try testing.expect(canVectorize(Layout.init(1024, 1)));
    try testing.expect(canVectorize(Layout.init(32, 4)));
    // ...and so are those that do not, via the bit-plane layout.
    try testing.expect(canVectorize(Layout.init(1024, 3)));
    // Beyond four bits the codebook no longer fits one 16-byte shuffle table, so
    // the fast path stops regardless of layout.
    try testing.expect(!canVectorize(Layout.init(1024, 5)));
    try testing.expect(!canVectorize(Layout.init(1024, 8)));
    // Dimension constraints differ by layout: sequential needs whole 16-byte
    // chunks, bit-plane needs an even number of 16-code groups.
    try testing.expect(!canVectorize(Layout.init(48, 4)));
    try testing.expect(!canVectorize(Layout.init(32, 2)));
    try testing.expect(!canVectorize(Layout.init(16, 3)));
}

test "every vectorizable bit-width tracks the exact reference" {
    // Extends the b=4 agreement test across all supported widths. A wrong shift or
    // stream mapping at b=1 or b=2 would produce plausible but consistently wrong
    // scores, exactly the failure mode the parity split had at b=4.
    const allocator = testing.allocator;
    // Covers both layouts: sequential at 1/2/4, bit-plane at 3. Four bits is the
    // ceiling — a 16-entry shuffle table cannot address more levels.
    for ([_]u6{ 1, 2, 3, 4 }) |bits| {
        const dim: u32 = 1024;
        const layout = Layout.init(dim, bits);
        try testing.expect(canVectorize(layout));

        var cb = try Codebook.init(allocator, Density.sphereCoord(dim), bits);
        defer cb.deinit();
        const table = Table.init(cb.centroids);

        var query = try Query.init(allocator, layout);
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

            const fast = scoreInt8(layout, table, query, stored, dim);
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

test "multi-query kernels agree exactly with the per-query kernels" {
    // Restructuring the loop must not change a single bit of the result: the multi
    // form does the same SDOTs in the same order, only with the unpack hoisted. An
    // approximate bound here would hide exactly the reordering bugs worth catching.
    for ([_]u6{ 1, 2, 4 }) |bits| {
        for ([_]u32{ 128, 256, 1024 }) |dim| {
            const layout = Layout.init(dim, bits);
            if (!canVectorize(layout)) continue;
            const width = multiWidth(layout);

            const codes = try testing.allocator.alloc(u8, layout.stride());
            defer testing.allocator.free(codes);
            const rotated = try testing.allocator.alloc(f32, dim);
            defer testing.allocator.free(rotated);
            var cb = try setup(dim, bits, 0x1234 + dim + bits, codes, rotated);
            defer cb.deinit();
            const table = Table.init(cb.centroids);

            const queries = try testing.allocator.alloc(Query, width);
            defer testing.allocator.free(queries);
            var built: usize = 0;
            defer for (0..built) |i| queries[i].deinit();
            var prng = std.Random.DefaultPrng.init(dim + bits);
            const random = prng.random();
            const scratch = try testing.allocator.alloc(f32, dim);
            defer testing.allocator.free(scratch);
            while (built < width) : (built += 1) {
                queries[built] = try Query.init(testing.allocator, layout);
                for (scratch) |*v| v.* = random.floatNorm(f32) / @sqrt(@as(f32, @floatFromInt(dim)));
                queries[built].load(scratch);
            }

            const out = try testing.allocator.alloc(f32, width);
            defer testing.allocator.free(out);
            scoreInt8Multi(layout, table, queries, codes, dim, out);

            for (0..width) |q| {
                const one = scoreInt8(layout, table, queries[q], codes, dim);
                try testing.expectEqual(one, out[q]);
            }
        }
    }
}

test "expanded multi-query kernel agrees exactly with the per-query kernel" {
    const dim: u32 = 512;
    const values = try testing.allocator.alloc(i8, dim);
    defer testing.allocator.free(values);
    var prng = std.Random.DefaultPrng.init(0xC0DE);
    const random = prng.random();
    for (values) |*v| v.* = random.intRangeAtMost(i8, -127, 127);

    const Q = 8;
    var queries: [Q]Query = undefined;
    var built: usize = 0;
    defer for (0..built) |i| queries[i].deinit();
    const scratch = try testing.allocator.alloc(f32, dim);
    defer testing.allocator.free(scratch);
    while (built < Q) : (built += 1) {
        queries[built] = try Query.initSequential(testing.allocator, dim);
        for (scratch) |*v| v.* = random.floatNorm(f32) / @sqrt(@as(f32, @floatFromInt(dim)));
        queries[built].load(scratch);
    }

    var out: [Q]f32 = undefined;
    scoreExpandedMulti(Q, values, queries[0..], 0.01, dim, out[0..]);
    for (0..Q) |q| {
        try testing.expectEqual(scoreExpanded(values, queries[q], 0.01, dim), out[q]);
    }
}
