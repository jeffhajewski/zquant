//! Saving and loading a `FlatIndex`.
//!
//! An index is cheap to describe and expensive to rebuild: everything but the codes is
//! a deterministic function of (dim, bits, seed, …), but the codes cost a pass over the
//! original vectors, which the caller may no longer have. So the format stores the
//! construction parameters, the calibration, and the codes verbatim, and loading
//! rebuilds the rest from the parameters.
//!
//! ## Layout
//!
//! Little-endian throughout, no alignment padding:
//!
//!     off  size  field
//!       0     4  magic "ZQNT"
//!       4     4  format_version     this layout
//!       8     4  encoding_version   which codes a given (dim, bits, seed) produces
//!      12     4  dim
//!      16     1  bits               2..6
//!      17     1  metric             0 inner product, 1 cosine, 2 l2
//!      18     1  rotation           0 hadamard, 1 dense
//!      19     1  correction         0 scalar, 1 qjl sketch
//!      20     1  residency          0 compact, 1 expanded
//!      21     1  use_sketch
//!      22     1  exact_scan
//!      23     1  calibrated
//!      24     8  seed
//!      32     8  count
//!      40     4  padded             working dimension
//!      44     4  code_stride        bytes of codes per vector
//!      48     4  sketch_stride      bytes of sketch per vector
//!      52     4  reserved, 0
//!      56        shifts             padded × f32
//!                scales             padded × f32
//!                codes              count × code_stride
//!                sketches           count × sketch_stride
//!                scalars            count × (f16 norm, f16 gamma)
//!     end-4   4  CRC-32 (IEEE) of every byte before it
//!
//! The working dimension and strides are derivable from the parameters; they are
//! stored anyway and checked on load, so a reader whose layout disagrees refuses the
//! file instead of misreading it.
//!
//! ## The two versions
//!
//! `format_version` changes when this layout does. `encoding_version` changes when the
//! *codes* do — when the rotation, codebook, or packing would encode the same vector
//! differently for the same parameters. Codes from another encoding are well-formed
//! bytes that decode to the wrong vectors, so a mismatch is refused rather than
//! loaded. tests/golden.zig pins the encodings and names this constant next to its
//! hashes, so the two cannot be updated apart.

const std = @import("std");
const Allocator = std.mem.Allocator;
const flat = @import("flat.zig");
const FlatIndex = flat.FlatIndex;
const rotation = @import("../math/rotation.zig");

pub const magic = "ZQNT";
pub const format_version: u32 = 1;

/// Bump whenever the golden hashes in tests/golden.zig change.
///
/// 1: block-mixed rotation for dimensions that are not a power of two (768 = 3·256),
///    rather than padding them to the next power of two.
pub const encoding_version: u32 = 1;

/// Largest dimension a file may declare. Matches ZQ_MAX_DIM in the C ABI.
pub const max_dim: u32 = 1 << 16;

/// Largest working dimension accepted with the dense reference rotation, which holds
/// padded² floats: 64 MiB here. Without a cap, a few bytes of header could ask for
/// 16 GiB.
pub const max_dense_padded: u32 = 4096;

const header_len = 56;
const crc_len = 4;
const scalar_len = 4;

pub const Error = error{
    /// Not a zquant index at all.
    NotAnIndex,
    /// A zquant index from a different format or encoding version.
    IncompatibleVersion,
    /// The right kind of file, but damaged: wrong length, bad checksum, or a field
    /// out of range.
    Corrupt,
    /// The destination buffer is smaller than `serializedSize`.
    BufferTooSmall,
};

/// Exact size of `write`'s output.
pub fn serializedSize(index: *const FlatIndex) usize {
    const padded: usize = index.quantizer.padded();
    const per_vector = index.codeStride() + index.sketchStride() + scalar_len;
    return header_len + 8 * padded + index.count() * per_vector + crc_len;
}

/// Serialize `index` into `out`, which must hold `serializedSize(index)` bytes.
/// Returns the bytes written.
pub fn write(index: *const FlatIndex, out: []u8) Error!usize {
    const total = serializedSize(index);
    if (out.len < total) return error.BufferTooSmall;
    const buf = out[0..total];
    const p = index.params;
    const padded = index.quantizer.padded();

    @memcpy(buf[0..4], magic);
    put(u32, buf, 4, format_version);
    put(u32, buf, 8, encoding_version);
    put(u32, buf, 12, p.dim);
    buf[16] = p.bits;
    buf[17] = switch (p.metric) {
        .inner_product => 0,
        .cosine => 1,
        .l2 => 2,
    };
    buf[18] = switch (p.rotation) {
        .hadamard => 0,
        .dense => 1,
    };
    buf[19] = switch (p.correction) {
        .scalar => 0,
        .qjl_sketch => 1,
    };
    buf[20] = switch (p.residency) {
        .compact => 0,
        .expanded => 1,
    };
    buf[21] = @intFromBool(p.use_sketch);
    buf[22] = @intFromBool(p.exact_scan);
    buf[23] = @intFromBool(index.calibrated);
    put(u64, buf, 24, p.seed);
    put(u64, buf, 32, index.count());
    put(u32, buf, 40, padded);
    put(u32, buf, 44, @intCast(index.codeStride()));
    put(u32, buf, 48, @intCast(index.sketchStride()));
    put(u32, buf, 52, 0);

    var at: usize = header_len;
    for (index.shifts) |v| {
        put(u32, buf, at, @bitCast(v));
        at += 4;
    }
    for (index.scales) |v| {
        put(u32, buf, at, @bitCast(v));
        at += 4;
    }
    @memcpy(buf[at..][0..index.codes.items.len], index.codes.items);
    at += index.codes.items.len;
    @memcpy(buf[at..][0..index.sketches.items.len], index.sketches.items);
    at += index.sketches.items.len;
    for (index.scalars.items) |s| {
        put(u16, buf, at, @bitCast(s.norm));
        put(u16, buf, at + 2, @bitCast(s.gamma));
        at += scalar_len;
    }
    put(u32, buf, at, std.hash.Crc32.hash(buf[0..at]));
    at += crc_len;
    std.debug.assert(at == total);
    return total;
}

/// Rebuild an index from `bytes`. Every field is checked, so arbitrary bytes produce
/// an error, never a crash or a half-initialized index.
pub fn read(allocator: Allocator, bytes: []const u8) (Error || Allocator.Error)!FlatIndex {
    if (bytes.len < 8 or !std.mem.eql(u8, bytes[0..4], magic)) return error.NotAnIndex;
    if (get(u32, bytes, 4) != format_version) return error.IncompatibleVersion;
    if (bytes.len < header_len + crc_len) return error.Corrupt;
    if (get(u32, bytes, 8) != encoding_version) return error.IncompatibleVersion;

    // Checked before anything is parsed, so a damaged length field cannot send the
    // reader past what is there.
    const body = bytes[0 .. bytes.len - crc_len];
    if (std.hash.Crc32.hash(body) != get(u32, bytes, bytes.len - crc_len)) return error.Corrupt;

    const dim = get(u32, bytes, 12);
    const bits = bytes[16];
    if (dim == 0 or dim > max_dim) return error.Corrupt;
    if (bits < 2 or bits > 6) return error.Corrupt;
    for (bytes[21..24]) |b| if (b > 1) return error.Corrupt;
    if (get(u32, bytes, 52) != 0) return error.Corrupt;

    const params: flat.Params = .{
        .dim = dim,
        .bits = @intCast(bits),
        .metric = switch (bytes[17]) {
            0 => .inner_product,
            1 => .cosine,
            2 => .l2,
            else => return error.Corrupt,
        },
        .rotation = switch (bytes[18]) {
            0 => .hadamard,
            1 => .dense,
            else => return error.Corrupt,
        },
        .correction = switch (bytes[19]) {
            0 => .scalar,
            1 => .qjl_sketch,
            else => return error.Corrupt,
        },
        .residency = switch (bytes[20]) {
            0 => .compact,
            1 => .expanded,
            else => return error.Corrupt,
        },
        .use_sketch = bytes[21] == 1,
        .exact_scan = bytes[22] == 1,
        .seed = get(u64, bytes, 24),
    };
    const calibrated = bytes[23] == 1;

    // Everything about the layout is checked against the file before anything is
    // allocated, so the size of what a header asks for is bounded by the size of the
    // file presenting it.
    const padded = get(u32, bytes, 40);
    const code_stride: usize = get(u32, bytes, 44);
    const sketch_stride: usize = get(u32, bytes, 48);
    if (padded != rotation.shapeFor(dim).padded) return error.Corrupt;
    if (params.rotation == .dense and padded > max_dense_padded) return error.Corrupt;
    const count = std.math.cast(usize, get(u64, bytes, 32)) orelse return error.Corrupt;
    // Ids are u32 throughout search.
    if (count > std.math.maxInt(u32)) return error.Corrupt;
    const per_vector = code_stride + sketch_stride + scalar_len;
    const vectors_len = std.math.mul(usize, count, per_vector) catch return error.Corrupt;
    const fixed = header_len + 8 * @as(usize, padded) + crc_len;
    const expected = std.math.add(usize, fixed, vectors_len) catch return error.Corrupt;
    if (bytes.len != expected) return error.Corrupt;

    var index = FlatIndex.init(allocator, params) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.Corrupt,
    };
    errdefer index.deinit();
    if (index.quantizer.padded() != padded or
        index.codeStride() != code_stride or
        index.sketchStride() != sketch_stride) return error.Corrupt;

    var at: usize = header_len;
    for (index.shifts) |*v| {
        v.* = @bitCast(get(u32, bytes, at));
        at += 4;
    }
    for (index.scales) |*v| {
        v.* = @bitCast(get(u32, bytes, at));
        at += 4;
    }
    for (index.shifts, index.scales) |sh, sc| {
        if (!std.math.isFinite(sh) or !std.math.isFinite(sc)) return error.Corrupt;
    }
    if (!calibrated) {
        for (index.shifts, index.scales) |sh, sc| if (sh != 0 or sc != 1) return error.Corrupt;
    }
    index.calibrated = calibrated;

    try index.reserve(count);
    index.codes.appendSliceAssumeCapacity(bytes[at..][0 .. count * code_stride]);
    at += count * code_stride;
    index.sketches.appendSliceAssumeCapacity(bytes[at..][0 .. count * sketch_stride]);
    at += count * sketch_stride;
    for (0..count) |_| {
        index.scalars.appendAssumeCapacity(.{
            .norm = @bitCast(get(u16, bytes, at)),
            .gamma = @bitCast(get(u16, bytes, at + 2)),
        });
        at += scalar_len;
    }
    std.debug.assert(at == bytes.len - crc_len);
    return index;
}

fn put(comptime T: type, buf: []u8, at: usize, v: T) void {
    std.mem.writeInt(T, buf[at..][0..@sizeOf(T)], v, .little);
}

fn get(comptime T: type, buf: []const u8, at: usize) T {
    return std.mem.readInt(T, buf[at..][0..@sizeOf(T)], .little);
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;

fn randomUnit(v: []f32, random: std.Random) void {
    var sq: f32 = 0;
    for (v) |*x| {
        x.* = random.floatNorm(f32) + 0.2;
        sq += x.* * x.*;
    }
    const inv = 1 / @sqrt(sq);
    for (v) |*x| x.* *= inv;
}

fn roundTrip(index: *const FlatIndex) !FlatIndex {
    const buf = try testing.allocator.alloc(u8, serializedSize(index));
    defer testing.allocator.free(buf);
    try testing.expectEqual(buf.len, try write(index, buf));
    return read(testing.allocator, buf);
}

test "a loaded index is the saved one: same bytes, same search results" {
    var prng = std.Random.DefaultPrng.init(0x5A7E);
    const random = prng.random();
    const dim = 200; // not a power of two, so the block rotation is covered
    const n = 300;
    var data: [n * dim]f32 = undefined;
    for (0..n) |i| randomUnit(data[i * dim ..][0..dim], random);

    const Case = struct { params: flat.Params, calibrate: bool };
    for ([_]Case{
        .{ .params = .{ .dim = dim, .bits = 5, .seed = 1 }, .calibrate = false },
        .{ .params = .{ .dim = dim, .bits = 4, .seed = 2, .metric = .cosine }, .calibrate = true },
        .{ .params = .{ .dim = dim, .bits = 4, .seed = 3, .metric = .l2, .residency = .expanded }, .calibrate = false },
        .{ .params = .{ .dim = dim, .bits = 3, .seed = 0xFFFF_FFFF_FFFF_FFF1, .correction = .qjl_sketch }, .calibrate = false },
    }) |case| {
        var index = try FlatIndex.init(testing.allocator, case.params);
        defer index.deinit();
        if (case.calibrate) try index.calibrate(&data);
        try index.addBatch(&data);

        var loaded = try roundTrip(&index);
        defer loaded.deinit();

        try testing.expectEqual(index.count(), loaded.count());
        try testing.expectEqual(index.calibrated, loaded.calibrated);
        try testing.expectEqualSlices(u8, index.codes.items, loaded.codes.items);
        try testing.expectEqualSlices(u8, index.sketches.items, loaded.sketches.items);
        try testing.expectEqualSlices(f32, index.scales, loaded.scales);
        try testing.expectEqualSlices(f32, index.shifts, loaded.shifts);

        var a = try FlatIndex.Searcher.init(testing.allocator, index, 10);
        defer a.deinit();
        var b = try FlatIndex.Searcher.init(testing.allocator, loaded, 10);
        defer b.deinit();
        for (0..5) |q| {
            const want = index.search(data[q * dim ..][0..dim], &a);
            const got = loaded.search(data[q * dim ..][0..dim], &b);
            try testing.expectEqualSlices(flat.Entry, want, got);
        }

        // And it keeps working as an index: adding to the loaded one matches adding
        // to the original.
        _ = try index.add(data[0..dim]);
        _ = try loaded.add(data[0..dim]);
        try testing.expectEqualSlices(u8, index.codes.items, loaded.codes.items);
    }
}

test "an empty index round-trips" {
    var index = try FlatIndex.init(testing.allocator, .{ .dim = 64, .bits = 4, .seed = 9 });
    defer index.deinit();
    var loaded = try roundTrip(&index);
    defer loaded.deinit();
    try testing.expectEqual(@as(usize, 0), loaded.count());
    try testing.expectEqual(index.params.seed, loaded.params.seed);
}

test "damaged files are refused, never misread" {
    var prng = std.Random.DefaultPrng.init(0xBAD);
    const random = prng.random();
    var index = try FlatIndex.init(testing.allocator, .{ .dim = 64, .bits = 4, .seed = 9 });
    defer index.deinit();
    var v: [64]f32 = undefined;
    for (0..10) |_| {
        randomUnit(&v, random);
        _ = try index.add(&v);
    }
    const good = try testing.allocator.alloc(u8, serializedSize(&index));
    defer testing.allocator.free(good);
    _ = try write(&index, good);
    const bad = try testing.allocator.alloc(u8, good.len);
    defer testing.allocator.free(bad);

    try testing.expectError(error.NotAnIndex, read(testing.allocator, "not an index"));
    try testing.expectError(error.NotAnIndex, read(testing.allocator, ""));
    try testing.expectError(error.BufferTooSmall, write(&index, bad[0 .. bad.len - 1]));

    // Every single-bit flip anywhere in the file is caught.
    for (0..good.len) |i| {
        @memcpy(bad, good);
        bad[i] ^= 0x10;
        if (read(testing.allocator, bad)) |loaded| {
            var l = loaded;
            l.deinit();
            return error.TestUnexpectedResult;
        } else |_| {}
    }

    // Truncated anywhere.
    for ([_]usize{ 7, 56, 100, good.len - 1 }) |len| {
        try testing.expect(std.meta.isError(read(testing.allocator, good[0..len])));
    }

    // A future version is refused as such, not as corruption.
    @memcpy(bad, good);
    put(u32, bad, 8, encoding_version + 1);
    put(u32, bad, bad.len - 4, std.hash.Crc32.hash(bad[0 .. bad.len - 4]));
    try testing.expectError(error.IncompatibleVersion, read(testing.allocator, bad));

    // A consistent checksum over an inconsistent header still fails the field checks:
    // a count that does not match the length.
    @memcpy(bad, good);
    put(u64, bad, 32, 11);
    put(u32, bad, bad.len - 4, std.hash.Crc32.hash(bad[0 .. bad.len - 4]));
    try testing.expectError(error.Corrupt, read(testing.allocator, bad));
    @memcpy(bad, good);
    put(u64, bad, 32, std.math.maxInt(u64));
    put(u32, bad, bad.len - 4, std.hash.Crc32.hash(bad[0 .. bad.len - 4]));
    try testing.expectError(error.Corrupt, read(testing.allocator, bad));
    @memcpy(bad, good);
    bad[16] = 1; // bits
    put(u32, bad, bad.len - 4, std.hash.Crc32.hash(bad[0 .. bad.len - 4]));
    try testing.expectError(error.Corrupt, read(testing.allocator, bad));
}
