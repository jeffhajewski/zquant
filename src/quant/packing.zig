//! Blocked, dimension-major code storage.
//!
//! The scan kernel (docs/DESIGN.md §4.2) walks dimensions while holding one
//! accumulator per vector in registers. That wants all 32 vectors' codes for a single
//! dimension to be contiguous, which is the transpose of how codes are produced.
//! This module owns that transpose and the sub-byte packing under it.
//!
//! Layout, for a block of `block_vectors` vectors:
//!
//!     block = dim slots, laid out consecutively
//!     slot j = codes for dimension j, one per vector, packed at `bits` bits each
//!
//! A slot is `block_vectors * bits / 8` bytes: 16 for b=4, 8 for b=2, 12 for b=3.
//! Codes are packed sequentially little-endian within a slot, so at b=4 byte `i`
//! holds vector `2i` in its low nibble and `2i+1` in its high nibble.
//!
//! That sequential rule is uniform across bit-widths, which is why it was chosen over
//! the FastScan convention of putting vectors 0–15 in low nibbles and 16–31 in high.
//! Both are equally good for a `tbl`/`vpshufb` kernel; they differ only in which
//! block-local slot each lane lands in. **Block-local vector order is an
//! implementation detail** — the kernel is free to produce accumulators in whatever
//! order falls out of its shuffles, so long as it maps back through the same rule.

const std = @import("std");

/// Vectors per block.
///
/// 32 because that is what one 128-bit shuffle covers at 4 bits per code: 16 bytes of
/// nibbles. Smaller wastes the register file; larger spills accumulators.
pub const block_vectors: usize = 32;

pub const Layout = struct {
    /// Padded dimension — one code per padded coordinate.
    dim: u32,
    bits: u6,

    pub fn init(dim: u32, bits: u6) Layout {
        std.debug.assert(bits >= 1 and bits <= 8);
        std.debug.assert(dim >= 1);
        // Keeps every slot a whole number of bytes, so slots never straddle.
        std.debug.assert((block_vectors * @as(usize, bits)) % 8 == 0);
        return .{ .dim = dim, .bits = bits };
    }

    /// Bytes holding all 32 vectors' codes for one dimension.
    pub fn slotBytes(self: Layout) usize {
        return block_vectors * @as(usize, self.bits) / 8;
    }

    pub fn blockBytes(self: Layout) usize {
        return @as(usize, self.dim) * self.slotBytes();
    }

    pub fn blockCount(count: usize) usize {
        return (count + block_vectors - 1) / block_vectors;
    }

    /// Total storage for `count` vectors, including the zero-padded tail of the last
    /// block. Padding is real storage, not a rounding convention: the kernel reads
    /// whole blocks and the index masks results afterwards.
    pub fn totalBytes(self: Layout, count: usize) usize {
        return blockCount(count) * self.blockBytes();
    }

    pub fn codesPerVector(self: Layout) usize {
        return self.dim;
    }

    /// Write one vector's codes into its place in a block.
    ///
    /// `codes` is the natural row-major output of the quantizer: one byte per
    /// coordinate. `slot` is the vector's index within the block, 0..32.
    pub fn scatter(self: Layout, block: []u8, slot: usize, codes: []const u8) void {
        std.debug.assert(slot < block_vectors);
        std.debug.assert(codes.len == self.dim);
        std.debug.assert(block.len >= self.blockBytes());

        const slot_bytes = self.slotBytes();
        for (codes, 0..) |code, j| {
            std.debug.assert(code < (@as(u16, 1) << @as(u4, @intCast(self.bits))));
            writeBits(block[j * slot_bytes ..][0..slot_bytes], slot * self.bits, self.bits, code);
        }
    }

    /// Read one vector's codes back out of a block.
    pub fn gather(self: Layout, block: []const u8, slot: usize, codes: []u8) void {
        std.debug.assert(slot < block_vectors);
        std.debug.assert(codes.len == self.dim);
        std.debug.assert(block.len >= self.blockBytes());

        const slot_bytes = self.slotBytes();
        for (codes, 0..) |*code, j| {
            code.* = readBits(block[j * slot_bytes ..][0..slot_bytes], slot * self.bits, self.bits);
        }
    }

    pub fn codeAt(self: Layout, block: []const u8, slot: usize, dim_index: usize) u8 {
        const slot_bytes = self.slotBytes();
        return readBits(block[dim_index * slot_bytes ..][0..slot_bytes], slot * self.bits, self.bits);
    }

    /// The bytes holding every vector's code for one dimension. This is what the scan
    /// kernel loads per iteration.
    pub fn slice(self: Layout, block: []const u8, dim_index: usize) []const u8 {
        const slot_bytes = self.slotBytes();
        return block[dim_index * slot_bytes ..][0..slot_bytes];
    }

    /// Pack `count` vectors (row-major, `dim` bytes each) into blocked storage.
    ///
    /// The tail of a partial block is zero, which decodes to code 0. That is a valid
    /// code, so it produces a real but meaningless score; callers must bound results
    /// by `count` rather than relying on padding to be inert.
    pub fn packAll(self: Layout, codes: []const u8, count: usize, dst: []u8) void {
        std.debug.assert(codes.len == count * @as(usize, self.dim));
        std.debug.assert(dst.len == self.totalBytes(count));
        @memset(dst, 0);

        const block_bytes = self.blockBytes();
        for (0..count) |i| {
            const block = i / block_vectors;
            const slot = i % block_vectors;
            self.scatter(
                dst[block * block_bytes ..][0..block_bytes],
                slot,
                codes[i * @as(usize, self.dim) ..][0..self.dim],
            );
        }
    }

    pub fn unpackAll(self: Layout, src: []const u8, count: usize, codes: []u8) void {
        std.debug.assert(codes.len == count * @as(usize, self.dim));
        std.debug.assert(src.len == self.totalBytes(count));

        const block_bytes = self.blockBytes();
        for (0..count) |i| {
            const block = i / block_vectors;
            const slot = i % block_vectors;
            self.gather(
                src[block * block_bytes ..][0..block_bytes],
                slot,
                codes[i * @as(usize, self.dim) ..][0..self.dim],
            );
        }
    }
};

/// Write `bits` bits of `value` at `offset` within `dst`.
///
/// A code spans at most two bytes because `bits <= 8`.
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

test "slot and block sizing" {
    for ([_]struct { bits: u6, slot: usize }{
        .{ .bits = 1, .slot = 4 },
        .{ .bits = 2, .slot = 8 },
        .{ .bits = 3, .slot = 12 },
        .{ .bits = 4, .slot = 16 },
        .{ .bits = 8, .slot = 32 },
    }) |case| {
        const layout = Layout.init(64, case.bits);
        try testing.expectEqual(case.slot, layout.slotBytes());
        try testing.expectEqual(case.slot * 64, layout.blockBytes());
    }
}

test "b=4 packs two vectors per byte, low nibble first" {
    // Pins the documented sequential rule, which the scan kernel's lane mapping
    // depends on. If this changes, the kernel changes with it.
    const layout = Layout.init(1, 4);
    var block = [_]u8{0} ** 16;

    layout.scatter(&block, 0, &[_]u8{0x3});
    layout.scatter(&block, 1, &[_]u8{0xA});
    layout.scatter(&block, 2, &[_]u8{0x5});

    try testing.expectEqual(@as(u8, 0xA3), block[0]);
    try testing.expectEqual(@as(u8, 0x05), block[1]);
}

test "round trip at every supported bit-width" {
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const random = prng.random();

    for ([_]u6{ 1, 2, 4, 8 }) |bits| {
        for ([_]u32{ 1, 7, 64, 128 }) |dim| {
            const layout = Layout.init(dim, bits);
            const max: u8 = @intCast((@as(u16, 1) << @as(u4, @intCast(bits))) - 1);

            const count: usize = 70; // deliberately not a multiple of 32
            const codes = try testing.allocator.alloc(u8, count * dim);
            defer testing.allocator.free(codes);
            for (codes) |*c| c.* = random.uintAtMost(u8, max);

            const packed_bytes = try testing.allocator.alloc(u8, layout.totalBytes(count));
            defer testing.allocator.free(packed_bytes);
            const back = try testing.allocator.alloc(u8, count * dim);
            defer testing.allocator.free(back);

            layout.packAll(codes, count, packed_bytes);
            layout.unpackAll(packed_bytes, count, back);
            try testing.expectEqualSlices(u8, codes, back);
        }
    }
}

test "three-bit codes straddle byte boundaries correctly" {
    // b=3 is the case where a code crosses a byte: offsets 6, 14, 22 ... all split.
    // Included even though b=3 is not a primary target, because it is the only
    // configuration that exercises the two-byte path in readBits/writeBits.
    const layout = Layout.init(3, 3);
    var prng = std.Random.DefaultPrng.init(11);
    const random = prng.random();

    const count = block_vectors;
    const codes = try testing.allocator.alloc(u8, count * 3);
    defer testing.allocator.free(codes);
    for (codes) |*c| c.* = random.uintAtMost(u8, 7);

    const packed_bytes = try testing.allocator.alloc(u8, layout.totalBytes(count));
    defer testing.allocator.free(packed_bytes);
    const back = try testing.allocator.alloc(u8, count * 3);
    defer testing.allocator.free(back);

    layout.packAll(codes, count, packed_bytes);
    layout.unpackAll(packed_bytes, count, back);
    try testing.expectEqualSlices(u8, codes, back);
}

test "codeAt agrees with a full gather" {
    const layout = Layout.init(33, 4);
    var prng = std.Random.DefaultPrng.init(3);
    const random = prng.random();

    const count = 40;
    const codes = try testing.allocator.alloc(u8, count * 33);
    defer testing.allocator.free(codes);
    for (codes) |*c| c.* = random.uintAtMost(u8, 15);

    const packed_bytes = try testing.allocator.alloc(u8, layout.totalBytes(count));
    defer testing.allocator.free(packed_bytes);
    layout.packAll(codes, count, packed_bytes);

    const block_bytes = layout.blockBytes();
    for (0..count) |i| {
        const block = packed_bytes[(i / block_vectors) * block_bytes ..][0..block_bytes];
        for (0..33) |j| {
            try testing.expectEqual(
                codes[i * 33 + j],
                layout.codeAt(block, i % block_vectors, j),
            );
        }
    }
}

test "a dimension slice holds exactly that dimension's codes" {
    // What the scan kernel loads per iteration: one contiguous run covering all 32
    // vectors for a single dimension, and nothing else.
    const layout = Layout.init(8, 4);
    const block = try testing.allocator.alloc(u8, layout.blockBytes());
    defer testing.allocator.free(block);
    @memset(block, 0);

    // Give dimension 5 a distinctive pattern and leave the rest zero.
    var codes = [_]u8{0} ** 8;
    for (0..block_vectors) |v| {
        codes[5] = @intCast(v % 16);
        layout.scatter(block, v, &codes);
    }

    const slot = layout.slice(block, 5);
    try testing.expectEqual(@as(usize, 16), slot.len);
    for (0..block_vectors) |v| {
        const lo_or_hi: u8 = if (v % 2 == 0) slot[v / 2] & 0x0F else slot[v / 2] >> 4;
        try testing.expectEqual(@as(u8, @intCast(v % 16)), lo_or_hi);
    }
    // Every other dimension is untouched.
    for (0..8) |j| {
        if (j == 5) continue;
        for (layout.slice(block, j)) |byte| try testing.expectEqual(@as(u8, 0), byte);
    }
}

test "partial blocks are zero-padded" {
    const layout = Layout.init(4, 4);
    const count = 5;
    const codes = [_]u8{15} ** (count * 4);

    const packed_bytes = try testing.allocator.alloc(u8, layout.totalBytes(count));
    defer testing.allocator.free(packed_bytes);
    layout.packAll(&codes, count, packed_bytes);

    // One block regardless, and slots 5..31 read as code 0.
    try testing.expectEqual(@as(usize, 1), Layout.blockCount(count));
    for (count..block_vectors) |slot| {
        for (0..4) |j| {
            try testing.expectEqual(@as(u8, 0), layout.codeAt(packed_bytes, slot, j));
        }
    }
}

test "block count boundaries" {
    try testing.expectEqual(@as(usize, 0), Layout.blockCount(0));
    try testing.expectEqual(@as(usize, 1), Layout.blockCount(1));
    try testing.expectEqual(@as(usize, 1), Layout.blockCount(32));
    try testing.expectEqual(@as(usize, 2), Layout.blockCount(33));
    try testing.expectEqual(@as(usize, 4), Layout.blockCount(128));
}

test "storage matches the advertised bits per coordinate" {
    // The compression claim, checked arithmetically: b bits per padded coordinate
    // and nothing else. At b=4, d=1024 that is 512 bytes per vector, 8x smaller
    // than f32.
    const layout = Layout.init(1024, 4);
    const per_vector = layout.totalBytes(block_vectors) / block_vectors;
    try testing.expectEqual(@as(usize, 512), per_vector);
    try testing.expectEqual(@as(usize, 1024 * 4 / 8), per_vector);
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
