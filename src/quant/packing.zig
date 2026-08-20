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

/// How codes are arranged within a vector's block.
pub const Kind = enum {
    /// Codes laid end to end, little-endian. Extraction is one shift and one mask,
    /// because a code never straddles a byte.
    sequential,
    /// Codes split across `bits` bit-planes, in groups of `plane_group`. Used where
    /// a code *does* straddle a byte, so shift-and-mask cannot reach it: the planes
    /// are expanded with `simd/bitmask.zig` and recombined by weight.
    bit_plane,
};

/// Codes per bit-plane group. 16 to match the SIMD lane count, so one group is
/// exactly one vector register's worth of codes.
pub const plane_group: usize = 16;

/// Sequential where a code fits inside a byte, bit-plane otherwise.
///
/// The split is by whether `bits` divides 8, not by which is faster in general:
/// sequential is markedly cheaper (≈2 ops per 16 codes against ≈17) and is used
/// wherever it is available at all.
pub fn kindFor(bits: u6) Kind {
    return if (8 % @as(usize, bits) == 0) .sequential else .bit_plane;
}

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

    pub fn kind(self: Layout) Kind {
        return kindFor(self.bits);
    }

    fn groups(self: Layout) usize {
        return (@as(usize, self.dim) + plane_group - 1) / plane_group;
    }

    /// Bytes of code actually used by one vector, before alignment padding.
    ///
    /// Both layouts cost `bits` per coordinate; bit-plane rounds up to whole groups
    /// of 16 rather than whole bytes, so it wastes nothing at any dimension that is
    /// a multiple of 16.
    pub fn codeBytes(self: Layout) usize {
        return switch (self.kind()) {
            .sequential => (@as(usize, self.dim) * self.bits + 7) / 8,
            .bit_plane => self.groups() * self.bits * (plane_group / 8),
        };
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

        const limit = @as(u16, 1) << @as(u4, @intCast(self.bits));
        switch (self.kind()) {
            .sequential => for (codes, 0..) |code, j| {
                std.debug.assert(code < limit);
                writeBits(dst, j * self.bits, self.bits, code);
            },
            .bit_plane => for (codes, 0..) |code, j| {
                std.debug.assert(code < limit);
                const group = j / plane_group;
                const slot = j % plane_group;
                const base = group * self.bits * (plane_group / 8);
                for (0..self.bits) |plane| {
                    if ((code >> @intCast(plane)) & 1 == 1) {
                        dst[base + plane * 2 + (slot >> 3)] |= @as(u8, 1) << @intCast(slot & 7);
                    }
                }
            },
        }
    }

    pub fn unpack(self: Layout, src: []const u8, codes: []u8) void {
        std.debug.assert(codes.len == self.dim);
        std.debug.assert(src.len >= self.codeBytes());
        for (codes, 0..) |*code, j| code.* = self.codeAt(src, j);
    }

    pub fn codeAt(self: Layout, src: []const u8, dim_index: usize) u8 {
        return switch (self.kind()) {
            .sequential => readBits(src, dim_index * self.bits, self.bits),
            .bit_plane => blk: {
                const group = dim_index / plane_group;
                const slot = dim_index % plane_group;
                const base = group * self.bits * (plane_group / 8);
                var code: u8 = 0;
                for (0..self.bits) |plane| {
                    const bit = (src[base + plane * 2 + (slot >> 3)] >> @intCast(slot & 7)) & 1;
                    code |= bit << @intCast(plane);
                }
                break :blk code;
            },
        };
    }

    /// Byte offset of a bit-plane group's planes within a vector's block.
    pub fn planeBase(self: Layout, group: usize) usize {
        std.debug.assert(self.kind() == .bit_plane);
        return group * self.bits * (plane_group / 8);
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

test "kind is chosen by whether a code fits in a byte" {
    for ([_]u6{ 1, 2, 4, 8 }) |bits| try testing.expectEqual(Kind.sequential, kindFor(bits));
    for ([_]u6{ 3, 5, 6, 7 }) |bits| try testing.expectEqual(Kind.bit_plane, kindFor(bits));
}

test "bit-plane layout costs exactly bits per coordinate" {
    // The point of grouping by 16 rather than rounding to bytes: no waste at any
    // dimension that is a multiple of 16, which every padded dimension is.
    for ([_]u6{ 3, 5, 6, 7 }) |bits| {
        const layout = Layout.init(1024, bits);
        try testing.expectEqual(Kind.bit_plane, layout.kind());
        try testing.expectEqual(@as(usize, 1024 * @as(usize, bits) / 8), layout.codeBytes());
    }
}

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

test "bit-plane round trip and plane structure" {
    var prng = std.Random.DefaultPrng.init(0xB17);
    const random = prng.random();

    for ([_]u6{ 3, 5, 6, 7 }) |bits| {
        for ([_]u32{ 16, 64, 768, 1024 }) |dim| {
            const layout = Layout.init(dim, bits);
            const max: u8 = @intCast((@as(u16, 1) << @as(u4, @intCast(bits))) - 1);

            const codes = try testing.allocator.alloc(u8, dim);
            defer testing.allocator.free(codes);
            const back = try testing.allocator.alloc(u8, dim);
            defer testing.allocator.free(back);
            const stored = try testing.allocator.alloc(u8, layout.stride());
            defer testing.allocator.free(stored);

            for (codes) |*c| c.* = random.uintAtMost(u8, max);
            layout.pack(codes, stored);
            layout.unpack(stored, back);
            try testing.expectEqualSlices(u8, codes, back);
        }
    }
}

test "bit-plane places each code bit in its own plane" {
    // Pins the layout the SIMD unpacker reads: plane p, byte slot>>3, bit slot&7.
    const layout = Layout.init(16, 3);
    var stored = [_]u8{0} ** 16;

    var codes = [_]u8{0} ** 16;
    codes[0] = 0b101; // planes 0 and 2 set for slot 0
    codes[9] = 0b010; // plane 1 set for slot 9 (byte 1, bit 1)
    layout.pack(&codes, &stored);

    try testing.expectEqual(@as(u8, 0b0000_0001), stored[0]); // plane 0, byte 0
    try testing.expectEqual(@as(u8, 0), stored[1]);
    try testing.expectEqual(@as(u8, 0), stored[2]); // plane 1, byte 0
    try testing.expectEqual(@as(u8, 0b0000_0010), stored[3]); // plane 1, byte 1 -> slot 9
    try testing.expectEqual(@as(u8, 0b0000_0001), stored[4]); // plane 2, byte 0
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
