//! Row-major bit-packed code storage.
//!
//! Each vector's codes are packed contiguously at `bits` bits per coordinate, in
//! ascending dimension order, little-endian within each byte. At b=4 that means byte
//! `i` holds dimension `2i` in its low nibble and `2i+1` in its high nibble.
//!
//! ## Why row-major, and not the FastScan blocked layout
//!
//! This reverses the choice in docs/DESIGN.md §4.2, which called for a 32-vector
//! dimension-major block. That design was carried over from product quantization,
//! where it is necessary — and here it is not.
//!
//! In PQ, each subspace has its own *query-dependent* lookup table,
//! `LUT_m[k] = ⟨q_m, centroid_{m,k}⟩`. Applying it requires every vector's code for
//! subspace `m` at once, which is exactly what a dimension-major block provides.
//!
//! TurboQuant's codebook is *scalar*, so the equivalent table factorizes:
//!
//!     LUT_j[k] = p_j · c[k]
//!
//! `c[]` is a single 16-entry table at b=4 — query-independent, shared by every
//! dimension, and small enough to sit in one SIMD register for the entire scan. Only
//! the scalar multiplier `p_j` depends on the query. FastScan's whole reason for
//! existing therefore does not apply, and the scan reduces to a per-vector dot
//! product along dimensions: precisely the shape `SDOT` (ARM) and VNNI (x86) exist
//! to accelerate.
//!
//! Measured cost at d=1024, b=4, per vector: 32 × 16-byte loads, each yielding 32
//! codes for 6 ops (and, shift, 2×tbl, 2×sdot) — 192 ops total, ~0.19 ops/dimension,
//! landing near 37 GB/s. The dimension-major form needs an int8→f32 widen per
//! dimension and costs roughly 3× that, which leaves the scan compute-bound rather
//! than memory-bound.
//!
//! Row-major is also simply less machinery: no transpose at insert, and sequential
//! access per vector.
//!
//! ## Nibble order and the query
//!
//! Because byte `i` holds dimensions `2i` and `2i+1`, masking the low nibbles of a
//! 16-byte load yields codes for the *even* dimensions and shifting down the high
//! nibbles yields the *odd* ones. The query must be de-interleaved the same way to
//! line up. That is a fixed permutation applied once per query, not per vector.

const std = @import("std");

/// Vectors' code blocks are padded to this multiple so every vector starts aligned
/// and the kernel's 16-byte chunk loop never straddles a vector boundary.
pub const alignment: usize = 16;

pub const Layout = struct {
    /// Padded dimension — one code per padded coordinate.
    dim: u32,
    bits: u6,

    pub fn init(dim: u32, bits: u6) Layout {
        std.debug.assert(bits >= 1 and bits <= 8);
        std.debug.assert(dim >= 1);
        return .{ .dim = dim, .bits = bits };
    }

    /// Bytes of code actually used by one vector, before alignment padding.
    pub fn codeBytes(self: Layout) usize {
        return (@as(usize, self.dim) * self.bits + 7) / 8;
    }

    /// Stride between consecutive vectors, including alignment padding.
    pub fn stride(self: Layout) usize {
        return std.mem.alignForward(usize, self.codeBytes(), alignment);
    }

    pub fn totalBytes(self: Layout, count: usize) usize {
        return count * self.stride();
    }

    pub fn pack(self: Layout, codes: []const u8, dst: []u8) void {
        std.debug.assert(codes.len == self.dim);
        std.debug.assert(dst.len >= self.codeBytes());
        @memset(dst[0..self.stride()], 0);
        for (codes, 0..) |code, j| {
            std.debug.assert(code < (@as(u16, 1) << @as(u4, @intCast(self.bits))));
            writeBits(dst, j * self.bits, self.bits, code);
        }
    }

    pub fn unpack(self: Layout, src: []const u8, codes: []u8) void {
        std.debug.assert(codes.len == self.dim);
        std.debug.assert(src.len >= self.codeBytes());
        for (codes, 0..) |*code, j| {
            code.* = readBits(src, j * self.bits, self.bits);
        }
    }

    pub fn codeAt(self: Layout, src: []const u8, dim_index: usize) u8 {
        return readBits(src, dim_index * self.bits, self.bits);
    }

    /// One vector's code block within packed storage.
    pub fn vectorSlice(self: Layout, storage: []const u8, index: usize) []const u8 {
        return storage[index * self.stride() ..][0..self.stride()];
    }

    pub fn vectorSliceMut(self: Layout, storage: []u8, index: usize) []u8 {
        return storage[index * self.stride() ..][0..self.stride()];
    }

    pub fn packAll(self: Layout, codes: []const u8, count: usize, dst: []u8) void {
        std.debug.assert(codes.len == count * @as(usize, self.dim));
        std.debug.assert(dst.len == self.totalBytes(count));
        for (0..count) |i| {
            self.pack(codes[i * @as(usize, self.dim) ..][0..self.dim], self.vectorSliceMut(dst, i));
        }
    }

    pub fn unpackAll(self: Layout, src: []const u8, count: usize, codes: []u8) void {
        std.debug.assert(codes.len == count * @as(usize, self.dim));
        std.debug.assert(src.len == self.totalBytes(count));
        for (0..count) |i| {
            self.unpack(self.vectorSlice(src, i), codes[i * @as(usize, self.dim) ..][0..self.dim]);
        }
    }
};

/// Write `bits` bits of `value` at `offset`. A code spans at most two bytes because
/// `bits <= 8`.
fn writeBits(dst: []u8, offset: usize, bits: u6, value: u8) void {
    const byte = offset >> 3;
    const shift: u3 = @intCast(offset & 7);
    dst[byte] |= @truncate(@as(u16, value) << @as(u4, shift));
    const consumed = 8 - @as(usize, shift);
    if (bits > consumed) {
        dst[byte + 1] |= @truncate(@as(u16, value) >> @intCast(consumed));
    }
}

fn readBits(src: []const u8, offset: usize, bits: u6) u8 {
    const byte = offset >> 3;
    const shift: u3 = @intCast(offset & 7);
    const mask: u16 = (@as(u16, 1) << @as(u4, @intCast(bits))) - 1;
    var value: u16 = @as(u16, src[byte]) >> @as(u4, shift);
    const consumed = 8 - @as(usize, shift);
    if (bits > consumed) {
        value |= @as(u16, src[byte + 1]) << @intCast(consumed);
    }
    return @truncate(value & mask);
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;

test "sizing and alignment" {
    for ([_]struct { dim: u32, bits: u6, code: usize, stride: usize }{
        .{ .dim = 1024, .bits = 4, .code = 512, .stride = 512 },
        .{ .dim = 1024, .bits = 2, .code = 256, .stride = 256 },
        .{ .dim = 256, .bits = 4, .code = 128, .stride = 128 },
        .{ .dim = 100, .bits = 4, .code = 50, .stride = 64 }, // padded up
        .{ .dim = 64, .bits = 1, .code = 8, .stride = 16 },
        .{ .dim = 128, .bits = 8, .code = 128, .stride = 128 },
    }) |case| {
        const layout = Layout.init(case.dim, case.bits);
        try testing.expectEqual(case.code, layout.codeBytes());
        try testing.expectEqual(case.stride, layout.stride());
        try testing.expect(layout.stride() % alignment == 0);
    }
}

test "b=4 packs dimension 2i in the low nibble, 2i+1 in the high" {
    // Pins the nibble order the scan kernel's query de-interleaving depends on.
    const layout = Layout.init(4, 4);
    var buf = [_]u8{0} ** 16;
    layout.pack(&[_]u8{ 0x3, 0xA, 0x5, 0xC }, &buf);
    try testing.expectEqual(@as(u8, 0xA3), buf[0]);
    try testing.expectEqual(@as(u8, 0xC5), buf[1]);
}

test "round trip at every supported bit-width" {
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const random = prng.random();

    for ([_]u6{ 1, 2, 3, 4, 5, 8 }) |bits| {
        for ([_]u32{ 1, 7, 64, 100, 256 }) |dim| {
            const layout = Layout.init(dim, bits);
            const max: u8 = @intCast((@as(u16, 1) << @as(u4, @intCast(bits))) - 1);

            const count: usize = 37;
            const codes = try testing.allocator.alloc(u8, count * dim);
            defer testing.allocator.free(codes);
            for (codes) |*c| c.* = random.uintAtMost(u8, max);

            const storage = try testing.allocator.alloc(u8, layout.totalBytes(count));
            defer testing.allocator.free(storage);
            const back = try testing.allocator.alloc(u8, count * dim);
            defer testing.allocator.free(back);

            layout.packAll(codes, count, storage);
            layout.unpackAll(storage, count, back);
            try testing.expectEqualSlices(u8, codes, back);
        }
    }
}

test "three- and five-bit codes straddle byte boundaries correctly" {
    // The only configurations exercising the two-byte path in the bit helpers.
    for ([_]u6{ 3, 5 }) |bits| {
        const layout = Layout.init(64, bits);
        var prng = std.Random.DefaultPrng.init(11);
        const random = prng.random();
        const max: u8 = @intCast((@as(u16, 1) << @as(u4, @intCast(bits))) - 1);

        var codes: [64]u8 = undefined;
        for (&codes) |*c| c.* = random.uintAtMost(u8, max);

        const storage = try testing.allocator.alloc(u8, layout.totalBytes(1));
        defer testing.allocator.free(storage);
        var back: [64]u8 = undefined;

        layout.packAll(&codes, 1, storage);
        layout.unpackAll(storage, 1, &back);
        try testing.expectEqualSlices(u8, &codes, &back);
    }
}

test "codeAt agrees with a full unpack" {
    const layout = Layout.init(33, 4);
    var prng = std.Random.DefaultPrng.init(3);
    const random = prng.random();

    const count = 9;
    const codes = try testing.allocator.alloc(u8, count * 33);
    defer testing.allocator.free(codes);
    for (codes) |*c| c.* = random.uintAtMost(u8, 15);

    const storage = try testing.allocator.alloc(u8, layout.totalBytes(count));
    defer testing.allocator.free(storage);
    layout.packAll(codes, count, storage);

    for (0..count) |i| {
        const slice = layout.vectorSlice(storage, i);
        for (0..33) |j| try testing.expectEqual(codes[i * 33 + j], layout.codeAt(slice, j));
    }
}

test "vectors do not bleed into each other" {
    // Alignment padding means a vector's slice extends past its codes; writing one
    // vector must not disturb its neighbours.
    const layout = Layout.init(100, 4); // 50 bytes of codes, 64-byte stride
    const count = 4;
    const storage = try testing.allocator.alloc(u8, layout.totalBytes(count));
    defer testing.allocator.free(storage);

    const all_max = [_]u8{15} ** 100;
    const all_zero = [_]u8{0} ** 100;
    for (0..count) |i| {
        layout.pack(if (i == 1) &all_max else &all_zero, layout.vectorSliceMut(storage, i));
    }

    var back: [100]u8 = undefined;
    for (0..count) |i| {
        layout.unpack(layout.vectorSlice(storage, i), &back);
        const want: u8 = if (i == 1) 15 else 0;
        for (back) |c| try testing.expectEqual(want, c);
    }
    // Padding bytes stay zero.
    for (0..count) |i| {
        for (layout.vectorSlice(storage, i)[layout.codeBytes()..]) |byte| {
            try testing.expectEqual(@as(u8, 0), byte);
        }
    }
}

test "storage matches the advertised bits per coordinate" {
    // The compression claim, checked arithmetically: at b=4, d=1024 that is 512
    // bytes per vector, 8x smaller than f32, with no padding overhead.
    const layout = Layout.init(1024, 4);
    try testing.expectEqual(@as(usize, 1024 * 4 / 8), layout.stride());
    try testing.expectEqual(@as(usize, 512), layout.stride());

    const f32_bytes = 1024 * @sizeOf(f32);
    try testing.expectEqual(@as(usize, 8), f32_bytes / layout.stride());
}

test "bit field helpers handle every offset and width" {
    for (1..9) |bits_usize| {
        const bits: u6 = @intCast(bits_usize);
        const max: u8 = @intCast((@as(u16, 1) << @as(u4, @intCast(bits))) - 1);
        for (0..16) |offset| {
            for ([_]u8{ 0, 1, max }) |value| {
                var buf = [_]u8{0} ** 4;
                writeBits(&buf, offset, bits, value);
                try testing.expectEqual(value, readBits(&buf, offset, bits));
            }
        }
    }
}
