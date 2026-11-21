//! Where an index keeps its vectors' codes.
//!
//! Most codes are stored one vector after another, `stride` bytes each, the order the
//! per-vector kernels read. One-bit codes are stored a column at a time instead, for
//! the block kernels of `simd/columns.zig`: in blocks of 16 vectors, with byte k of each
//! of the 16 together, so byte k of vector v is at
//!
//!     (v / 16) · 16 · stride  +  16 · k  +  v % 16
//!
//! The last block is allocated whole and its unused lanes hold zeros. Either way a
//! vector's bytes are the same as `packing.Layout.pack` gives; only where they sit
//! differs, so files hold them one vector after another whatever the arrangement.

const std = @import("std");
const Allocator = std.mem.Allocator;
const columns = @import("../simd/columns.zig");
const scan = @import("../simd/scan.zig");

pub const block = columns.block;

pub const Codes = struct {
    bytes: std.ArrayList(u8) = .empty,
    /// Bytes per vector.
    stride: usize,
    /// Blocks of 16 vectors, a byte column at a time, rather than one after another.
    by_column: bool,
    len: usize = 0,

    pub fn init(stride: usize, by_column: bool) Codes {
        return .{ .stride = stride, .by_column = by_column };
    }

    pub fn deinit(self: *Codes, allocator: Allocator) void {
        self.bytes.deinit(allocator);
        self.* = undefined;
    }

    /// Bytes `n` vectors take.
    pub fn bytesFor(self: Codes, n: usize) usize {
        return if (self.by_column) (n + block - 1) / block * block * self.stride else n * self.stride;
    }

    pub fn ensureUnusedCapacity(self: *Codes, allocator: Allocator, n: usize) Allocator.Error!void {
        try self.bytes.ensureTotalCapacity(allocator, self.bytesFor(self.len + n));
    }

    pub fn ensureTotalCapacityPrecise(self: *Codes, allocator: Allocator, n: usize) Allocator.Error!void {
        try self.bytes.ensureTotalCapacityPrecise(allocator, self.bytesFor(n));
    }

    /// Add `n` vectors of zero bytes. Capacity must already be reserved.
    pub fn growAssumeCapacity(self: *Codes, n: usize) void {
        const want = self.bytesFor(self.len + n);
        @memset(self.bytes.addManyAsSliceAssumeCapacity(want - self.bytes.items.len), 0);
        self.len += n;
    }

    /// Append one vector's bytes. Capacity must already be reserved.
    pub fn appendAssumeCapacity(self: *Codes, code: []const u8) void {
        self.growAssumeCapacity(1);
        self.set(self.len - 1, code);
    }

    /// Append vectors stored one after another, as files hold them.
    pub fn appendRowsAssumeCapacity(self: *Codes, rows: []const u8) void {
        std.debug.assert(rows.len % self.stride == 0);
        if (!self.by_column) {
            self.bytes.appendSliceAssumeCapacity(rows);
            self.len += rows.len / self.stride;
            return;
        }
        var at: usize = 0;
        while (at < rows.len) : (at += self.stride) self.appendAssumeCapacity(rows[at..][0..self.stride]);
    }

    /// Set vector `i`'s bytes. Writes only vector `i`'s bytes, so threads may set
    /// different vectors at once.
    pub fn set(self: *const Codes, i: usize, code: []const u8) void {
        std.debug.assert(i < self.len and code.len == self.stride);
        if (!self.by_column) {
            @memcpy(self.bytes.items[i * self.stride ..][0..self.stride], code);
            return;
        }
        const base = self.blockStart(i) + i % block;
        for (code, 0..) |b, k| self.bytes.items[base + k * block] = b;
    }

    /// Pack one-bit codes, one per byte of `unpacked`, as vector `i`'s bytes: what
    /// `packing.Layout.pack` gives at one bit, written straight to their columns.
    pub fn setOneBit(self: *const Codes, i: usize, unpacked: []const u8) void {
        std.debug.assert(self.by_column and i < self.len and unpacked.len <= self.stride * 8);
        const base = self.blockStart(i) + i % block;
        for (0..self.stride) |k| {
            var b: u8 = 0;
            for (0..8) |t| {
                const j = k * 8 + t;
                if (j < unpacked.len) b |= @as(u8, @intCast(unpacked[j])) << @intCast(t);
            }
            self.bytes.items[base + k * block] = b;
        }
    }

    /// ⟨p, ũ⟩ for vector `i` from a one-bit query's tables: the sum the block kernels
    /// take for it, through the same `columns.Affine`, so the same bits.
    pub fn oneBitTerm(self: *const Codes, i: usize, query: scan.Query, table: scan.Table) f32 {
        std.debug.assert(self.by_column and i < self.len);
        const s = columns.sumAt(query.lut, self.ptr() + self.blockStart(i), i % block, self.stride);
        return query.affine(table).apply(s);
    }

    /// Vector `i`'s bytes: in place when stored one after another, otherwise gathered
    /// into `buf`, which must hold `stride` bytes.
    pub fn get(self: *const Codes, i: usize, buf: []u8) []const u8 {
        std.debug.assert(i < self.len);
        if (!self.by_column) return self.bytes.items[i * self.stride ..][0..self.stride];
        const base = self.blockStart(i) + i % block;
        for (buf[0..self.stride], 0..) |*b, k| b.* = self.bytes.items[base + k * block];
        return buf[0..self.stride];
    }

    /// Every vector's bytes one after another, `len · stride` of them, as files hold them.
    pub fn writeRows(self: *const Codes, out: []u8) void {
        std.debug.assert(out.len == self.len * self.stride);
        if (!self.by_column) return @memcpy(out, self.bytes.items[0..out.len]);
        for (0..self.len) |i| _ = self.get(i, out[i * self.stride ..][0..self.stride]);
    }

    /// Byte offset of the block holding vector `i`.
    pub fn blockStart(self: Codes, i: usize) usize {
        return i / block * block * self.stride;
    }

    pub fn ptr(self: *const Codes) [*]const u8 {
        return self.bytes.items.ptr;
    }

    /// For the per-vector kernels: vector `i`'s bytes in place.
    pub fn row(self: *const Codes, i: usize) []const u8 {
        std.debug.assert(!self.by_column);
        return self.bytes.items[i * self.stride ..][0..self.stride];
    }

    /// `row`, to write. Only vector `i`'s bytes, as `set`.
    pub fn rowMut(self: *const Codes, i: usize) []u8 {
        std.debug.assert(!self.by_column);
        return self.bytes.items[i * self.stride ..][0..self.stride];
    }
};

const testing = std.testing;

test "both arrangements hold the same bytes, in and out" {
    var prng = std.Random.DefaultPrng.init(0xC0DE);
    const random = prng.random();
    const stride = 48;
    for ([_]usize{ 0, 1, 15, 16, 17, 40 }) |n| {
        const rows = try testing.allocator.alloc(u8, n * stride);
        defer testing.allocator.free(rows);
        random.bytes(rows);
        for ([_]bool{ false, true }) |by_column| {
            var c = Codes.init(stride, by_column);
            defer c.deinit(testing.allocator);
            // Half appended in one go, the rest one at a time.
            try c.ensureUnusedCapacity(testing.allocator, n);
            c.appendRowsAssumeCapacity(rows[0 .. n / 2 * stride]);
            for (n / 2..n) |i| c.appendAssumeCapacity(rows[i * stride ..][0..stride]);
            try testing.expectEqual(n, c.len);
            try testing.expectEqual(c.bytesFor(n), c.bytes.items.len);

            const back = try testing.allocator.alloc(u8, n * stride);
            defer testing.allocator.free(back);
            c.writeRows(back);
            try testing.expectEqualSlices(u8, rows, back);
            var buf: [stride]u8 = undefined;
            for (0..n) |i| try testing.expectEqualSlices(u8, rows[i * stride ..][0..stride], c.get(i, &buf));

            if (by_column and n > 0) {
                // The documented position, and zeros in the last block's unused lanes.
                const v = n - 1;
                try testing.expectEqual(rows[v * stride + 5], c.bytes.items[c.blockStart(v) + 5 * block + v % block]);
                for (n..c.bytesFor(n) / stride) |lane_v| {
                    for (0..stride) |k| try testing.expectEqual(@as(u8, 0), c.bytes.items[c.blockStart(lane_v) + k * block + lane_v % block]);
                }
            }
        }
    }
}

test "set after grow, as parallel encoding does" {
    var c = Codes.init(16, true);
    defer c.deinit(testing.allocator);
    try c.ensureUnusedCapacity(testing.allocator, 20);
    c.growAssumeCapacity(20);
    var code: [16]u8 = undefined;
    for (0..20) |i| {
        @memset(&code, @intCast(i));
        c.set(i, &code);
    }
    var buf: [16]u8 = undefined;
    for (0..20) |i| for (c.get(i, &buf)) |b| try testing.expectEqual(@as(u8, @intCast(i)), b);
}

test "setOneBit packs as the layout does" {
    const packing = @import("../quant/packing.zig");
    var prng = std.Random.DefaultPrng.init(0xB17);
    const random = prng.random();
    const layout = packing.Layout.init(256, 1);
    var c = Codes.init(layout.stride(), true);
    defer c.deinit(testing.allocator);
    try c.ensureUnusedCapacity(testing.allocator, 3);
    c.growAssumeCapacity(3);
    var unpacked: [256]u8 = undefined;
    for (&unpacked) |*u| u.* = random.uintLessThan(u8, 2);
    var want: [32]u8 = undefined;
    layout.pack(&unpacked, &want);
    c.setOneBit(2, &unpacked);
    var buf: [32]u8 = undefined;
    try testing.expectEqualSlices(u8, &want, c.get(2, &buf));
}
