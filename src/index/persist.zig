//! Saving and loading a `FlatIndex` or an `IvfIndex`.
//!
//! An index is cheap to describe and expensive to rebuild: everything but the codes is
//! a deterministic function of (dim, bits, seed, …), but the codes cost a pass over the
//! original vectors, which the caller may no longer have. So the format stores the
//! construction parameters, the calibration or the list centroids, and the codes
//! verbatim, and loading rebuilds the rest from the parameters.
//!
//! ## Layout
//!
//! Little-endian throughout, no alignment padding. The header is the same for both
//! kinds of index:
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
//!      52     4  lists              0 for a flat index
//!
//! A flat index follows with:
//!
//!      56        centre             padded × f32, zero unless calibrated
//!                codes              count × code_stride
//!                sketches           count × sketch_stride
//!                scalars            count × (f16 norm, f16 gamma)
//!                biases             count × f16, only if calibrated
//!     end-4   4  CRC-32 (IEEE) of every byte before it
//!
//! A partitioned one, whose header holds a default flat index's values for the settings
//! it has none of (`correction` 0, `use_sketch` 1, `exact_scan` 0, `calibrated` 0,
//! `sketch_stride` 0), follows with:
//!
//!      56        centroids          lists × padded × f32
//!                sizes              lists × u32, vectors in each list
//!                codes              count × code_stride, list 0's first
//!                scalars            count × (f16 norm, f16 gamma), in the same order
//!                biases             count × f16
//!                ids                count × u32
//!     end-4   4  CRC-32 (IEEE) of every byte before it
//!
//! Each list's ids rise, and together they are 0..count−1 once each. The reader checks
//! both, since search and `scoreIds` rely on them.
//!
//! The `lists` field was reserved and zero before partitioned indexes could be saved, so
//! flat files are unchanged by it. Each reader must handle both kinds; `kind` says which
//! a file holds.
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
const ivf = @import("ivf.zig");
const IvfIndex = ivf.IvfIndex;
const rotation = @import("../math/rotation.zig");

pub const magic = "ZQNT";

/// 2: calibration is a centre and a per-vector bias rather than a per-coordinate shift
///    and scale, and the per-vector scalar under `.scalar` is the factor that makes the
///    estimate exact along the vector rather than a least-squares fit. Version 1 files
///    are refused: their scalars mean something else.
pub const format_version: u32 = 2;

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
const bias_len = 2;

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
    /// An intact index of the other kind: a partitioned one given to `read`, or a flat
    /// one given to `readPartitioned`. `kind` says which a file holds.
    WrongKind,
};

pub const Kind = enum { flat, partitioned };

/// Which kind of index `bytes` holds, from its header alone. The rest is checked only
/// when it is read.
pub fn kind(bytes: []const u8) Error!Kind {
    if (bytes.len < 8 or !std.mem.eql(u8, bytes[0..4], magic)) return error.NotAnIndex;
    if (get(u32, bytes, 4) != format_version) return error.IncompatibleVersion;
    if (bytes.len < header_len + crc_len) return error.Corrupt;
    return if (get(u32, bytes, 52) == 0) .flat else .partitioned;
}

/// Exact size of `write`'s output.
pub fn serializedSize(index: *const FlatIndex) usize {
    const padded: usize = index.quantizer.padded();
    const per_vector = index.codeStride() + index.sketchStride() + scalar_len + index.biasStride();
    return header_len + 4 * padded + index.count() * per_vector + crc_len;
}

/// Serialize `index` into `out`, which must hold `serializedSize(index)` bytes.
/// Returns the bytes written.
pub fn write(index: *const FlatIndex, out: []u8) Error!usize {
    const total = serializedSize(index);
    if (out.len < total) return error.BufferTooSmall;
    const buf = out[0..total];
    writeHeader(buf, .{
        .params = index.params,
        .calibrated = index.calibrated,
        .count = index.count(),
        .padded = index.quantizer.padded(),
        .code_stride = index.codeStride(),
        .sketch_stride = index.sketchStride(),
        .lists = 0,
    });

    var at: usize = header_len;
    for (index.centre) |v| {
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
    for (index.biases.items) |b| {
        put(u16, buf, at, @bitCast(b));
        at += bias_len;
    }
    put(u32, buf, at, std.hash.Crc32.hash(buf[0..at]));
    at += crc_len;
    std.debug.assert(at == total);
    return total;
}

const Header = struct {
    params: flat.Params,
    calibrated: bool,
    count: usize,
    padded: u32,
    code_stride: usize,
    sketch_stride: usize,
    lists: u32,
};

fn writeHeader(buf: []u8, h: Header) void {
    const p = h.params;
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
    buf[23] = @intFromBool(h.calibrated);
    put(u64, buf, 24, p.seed);
    put(u64, buf, 32, h.count);
    put(u32, buf, 40, h.padded);
    put(u32, buf, 44, @intCast(h.code_stride));
    put(u32, buf, 48, @intCast(h.sketch_stride));
    put(u32, buf, 52, h.lists);
}

/// Check the magic, versions and checksum, then parse and range-check the header. The
/// layout fields are returned as stored, for the caller to check against the index it
/// builds.
fn readHeader(bytes: []const u8) Error!Header {
    _ = try kind(bytes);
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
    const padded = get(u32, bytes, 40);
    if (padded != rotation.shapeFor(dim).padded) return error.Corrupt;
    if (params.rotation == .dense and padded > max_dense_padded) return error.Corrupt;
    const count = std.math.cast(usize, get(u64, bytes, 32)) orelse return error.Corrupt;
    // Ids are u32 throughout search.
    if (count > std.math.maxInt(u32)) return error.Corrupt;
    return .{
        .params = params,
        .calibrated = bytes[23] == 1,
        .count = count,
        .padded = padded,
        .code_stride = get(u32, bytes, 44),
        .sketch_stride = get(u32, bytes, 48),
        .lists = get(u32, bytes, 52),
    };
}

/// Rebuild an index from `bytes`. Every field is checked, so arbitrary bytes produce
/// an error, never a crash or a half-initialized index.
pub fn read(allocator: Allocator, bytes: []const u8) (Error || Allocator.Error)!FlatIndex {
    const h = try readHeader(bytes);
    if (h.lists != 0) return error.WrongKind;
    const params = h.params;
    const calibrated = h.calibrated;
    // `calibrate` refuses the sketch correction, so no writer produces both.
    if (calibrated and params.correction != .scalar) return error.Corrupt;

    // Everything about the layout is checked against the file before anything is
    // allocated, so the size of what a header asks for is bounded by the size of the
    // file presenting it.
    const padded = h.padded;
    const code_stride = h.code_stride;
    const sketch_stride = h.sketch_stride;
    const count = h.count;
    const per_vector = code_stride + sketch_stride + scalar_len + @as(usize, if (calibrated) bias_len else 0);
    const vectors_len = std.math.mul(usize, count, per_vector) catch return error.Corrupt;
    const fixed = header_len + 4 * @as(usize, padded) + crc_len;
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
    for (index.centre) |*v| {
        v.* = @bitCast(get(u32, bytes, at));
        at += 4;
    }
    for (index.centre) |c| {
        if (!std.math.isFinite(c)) return error.Corrupt;
        if (!calibrated and c != 0) return error.Corrupt;
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
    if (calibrated) for (0..count) |_| {
        index.biases.appendAssumeCapacity(@bitCast(get(u16, bytes, at)));
        at += bias_len;
    };
    std.debug.assert(at == bytes.len - crc_len);
    return index;
}

const size_len = 4;
const id_len = 4;

/// The flat parameters a partitioned index's header carries: its own, and a default flat
/// index's for the settings it lacks.
fn flatParams(p: ivf.Params) flat.Params {
    return .{
        .dim = p.dim,
        .bits = p.bits,
        .metric = p.metric,
        .seed = p.seed,
        .rotation = p.rotation,
        .residency = p.residency,
    };
}

/// Exact size of `writePartitioned`'s output.
pub fn serializedSizePartitioned(index: *const IvfIndex) usize {
    const padded: usize = index.quantizer.padded();
    const per_list = 4 * padded + size_len;
    const per_vector = index.codeStride() + scalar_len + bias_len + id_len;
    return header_len + index.listCount() * per_list + index.count() * per_vector + crc_len;
}

/// Serialize a trained partitioned index into `out`, which must hold
/// `serializedSizePartitioned(index)` bytes. Returns the bytes written.
pub fn writePartitioned(index: *const IvfIndex, out: []u8) (Error || error{NotTrained})!usize {
    if (!index.trained) return error.NotTrained;
    const total = serializedSizePartitioned(index);
    if (out.len < total) return error.BufferTooSmall;
    const buf = out[0..total];
    writeHeader(buf, .{
        .params = flatParams(index.params),
        .calibrated = false,
        .count = index.count(),
        .padded = index.quantizer.padded(),
        .code_stride = index.codeStride(),
        .sketch_stride = 0,
        .lists = @intCast(index.listCount()),
    });

    var at: usize = header_len;
    for (index.centroids) |v| {
        put(u32, buf, at, @bitCast(v));
        at += 4;
    }
    for (index.lists) |l| {
        put(u32, buf, at, @intCast(l.len()));
        at += size_len;
    }
    for (index.lists) |l| {
        @memcpy(buf[at..][0..l.codes.items.len], l.codes.items);
        at += l.codes.items.len;
    }
    for (index.lists) |l| for (l.scalars.items) |s| {
        put(u16, buf, at, @bitCast(s.norm));
        put(u16, buf, at + 2, @bitCast(s.gamma));
        at += scalar_len;
    };
    for (index.lists) |l| for (l.biases.items) |b| {
        put(u16, buf, at, @bitCast(b));
        at += bias_len;
    };
    for (index.lists) |l| for (l.ids.items) |id| {
        put(u32, buf, at, id);
        at += id_len;
    };
    put(u32, buf, at, std.hash.Crc32.hash(buf[0..at]));
    at += crc_len;
    std.debug.assert(at == total);
    return total;
}

/// Rebuild a partitioned index from `bytes`, checked as thoroughly as `read` checks a
/// flat one.
pub fn readPartitioned(allocator: Allocator, bytes: []const u8) (Error || Allocator.Error)!IvfIndex {
    const h = try readHeader(bytes);
    if (h.lists == 0) return error.WrongKind;
    if (h.lists > ivf.max_lists) return error.Corrupt;
    const fp = h.params;
    const params: ivf.Params = .{
        .dim = fp.dim,
        .bits = fp.bits,
        .lists = h.lists,
        .metric = fp.metric,
        .seed = fp.seed,
        .rotation = fp.rotation,
        .residency = fp.residency,
    };
    if (!std.meta.eql(fp, flatParams(params)) or h.calibrated or h.sketch_stride != 0) return error.Corrupt;

    // As for a flat index, the whole layout is checked against the file's length before
    // anything is allocated.
    const lists: usize = h.lists;
    const padded: usize = h.padded;
    const count = h.count;
    const stride = h.code_stride;
    const per_vector = stride + scalar_len + bias_len + id_len;
    const lists_len = std.math.mul(usize, lists, 4 * padded + size_len) catch return error.Corrupt;
    const vectors_len = std.math.mul(usize, count, per_vector) catch return error.Corrupt;
    const expected = std.math.add(usize, header_len + crc_len, lists_len) catch return error.Corrupt;
    if (bytes.len != std.math.add(usize, expected, vectors_len) catch return error.Corrupt) return error.Corrupt;
    const sizes_at = header_len + lists * 4 * padded;
    const sizes = bytes[sizes_at..][0 .. lists * size_len];
    var sum: u64 = 0;
    for (0..lists) |l| sum += listLen(sizes, l);
    if (sum != count) return error.Corrupt;

    var index = IvfIndex.init(allocator, params) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.Corrupt,
    };
    errdefer index.deinit();
    if (index.quantizer.padded() != padded or index.codeStride() != stride) return error.Corrupt;

    var at: usize = header_len;
    for (index.centroids) |*v| {
        v.* = @bitCast(get(u32, bytes, at));
        if (!std.math.isFinite(v.*)) return error.Corrupt;
        at += 4;
    }
    index.useCentroids();

    for (index.lists, 0..) |*l, i| {
        const len = listLen(sizes, i);
        try l.codes.ensureTotalCapacityPrecise(allocator, len * stride);
        try l.scalars.ensureTotalCapacityPrecise(allocator, len);
        try l.biases.ensureTotalCapacityPrecise(allocator, len);
        try l.ids.ensureTotalCapacityPrecise(allocator, len);
    }
    try index.list_of.resize(allocator, count);
    var seen = try std.DynamicBitSetUnmanaged.initEmpty(allocator, count);
    defer seen.deinit(allocator);

    at = sizes_at + lists * size_len;
    for (index.lists, 0..) |*l, i| {
        const len = listLen(sizes, i);
        l.codes.appendSliceAssumeCapacity(bytes[at..][0 .. len * stride]);
        at += len * stride;
    }
    for (index.lists, 0..) |*l, i| for (0..listLen(sizes, i)) |_| {
        l.scalars.appendAssumeCapacity(.{
            .norm = @bitCast(get(u16, bytes, at)),
            .gamma = @bitCast(get(u16, bytes, at + 2)),
        });
        at += scalar_len;
    };
    for (index.lists) |*l| for (0..l.scalars.items.len) |_| {
        l.biases.appendAssumeCapacity(@bitCast(get(u16, bytes, at)));
        at += bias_len;
    };
    for (index.lists, 0..) |*l, i| for (0..l.scalars.items.len) |j| {
        const id = get(u32, bytes, at);
        at += id_len;
        // Ids rise within a list, and each of 0..count−1 appears once in all.
        if (id >= count or seen.isSet(id)) return error.Corrupt;
        if (j > 0 and id <= l.ids.items[j - 1]) return error.Corrupt;
        seen.set(id);
        l.ids.appendAssumeCapacity(id);
        index.list_of.items[id] = @intCast(i);
    };
    index.total = count;
    std.debug.assert(at == bytes.len - crc_len);
    return index;
}

fn listLen(sizes: []const u8, l: usize) usize {
    return get(u32, sizes, l * size_len);
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
        try testing.expectEqualSlices(f32, index.centre, loaded.centre);
        try testing.expectEqualSlices(f16, index.biases.items, loaded.biases.items);

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

// -- partitioned ---------------------------------------------------------------

fn clustered(a: Allocator, rows: usize, dim: usize, random: std.Random) ![]f32 {
    const centres = try a.alloc(f32, 8 * dim);
    defer a.free(centres);
    for (centres) |*c| c.* = random.floatNorm(f32);
    const out = try a.alloc(f32, rows * dim);
    for (0..rows) |r| {
        const c = random.uintLessThan(usize, 8);
        for (out[r * dim ..][0..dim], centres[c * dim ..][0..dim]) |*x, cv| x.* = cv + 0.5 * random.floatNorm(f32);
    }
    return out;
}

fn saved(a: Allocator, index: *const IvfIndex) ![]u8 {
    const buf = try a.alloc(u8, serializedSizePartitioned(index));
    errdefer a.free(buf);
    try testing.expectEqual(buf.len, try writePartitioned(index, buf));
    return buf;
}

test "a loaded partitioned index is the saved one" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x1F5);
    const dim = 200;
    const n = 1200;
    const data = try clustered(a, n, dim, prng.random());
    defer a.free(data);

    const Case = struct { params: ivf.Params, first: usize };
    for ([_]Case{
        .{ .params = .{ .dim = dim, .bits = 2, .lists = 12, .seed = 1 }, .first = n },
        .{ .params = .{ .dim = dim, .bits = 4, .lists = 7, .seed = 2, .metric = .cosine, .residency = .expanded }, .first = 500 },
        .{ .params = .{ .dim = dim, .bits = 6, .lists = 5, .seed = 3, .metric = .l2, .rotation = .dense }, .first = 0 },
    }) |c| {
        var index = try IvfIndex.init(a, c.params);
        defer index.deinit();
        try index.train(data, 2);
        try index.addBatch(data[0 .. c.first * dim]);

        const bytes = try saved(a, &index);
        defer a.free(bytes);
        try testing.expectEqual(Kind.partitioned, try kind(bytes));
        var loaded = try readPartitioned(a, bytes);
        defer loaded.deinit();
        try testing.expectEqual(index.count(), loaded.count());
        try testing.expectEqualSlices(f32, index.centroids, loaded.centroids);
        try testing.expectEqualSlices(f32, index.centroid_sq, loaded.centroid_sq);
        try testing.expectEqualSlices(u16, index.list_of.items, loaded.list_of.items);

        // Saving the loaded index gives the same bytes.
        const again = try saved(a, &loaded);
        defer a.free(again);
        try testing.expectEqualSlices(u8, bytes, again);

        // It keeps working as an index: adding the rest to both gives the same bytes,
        // and both search alike.
        try index.addBatch(data[c.first * dim ..]);
        try loaded.addBatchParallel(data[c.first * dim ..], 3);
        const grown = try saved(a, &index);
        defer a.free(grown);
        const grown_loaded = try saved(a, &loaded);
        defer a.free(grown_loaded);
        try testing.expectEqualSlices(u8, grown, grown_loaded);

        var sa = try IvfIndex.Searcher.init(a, index, 5, 10, 3);
        defer sa.deinit();
        var sb = try IvfIndex.Searcher.init(a, loaded, 5, 10, 3);
        defer sb.deinit();
        const queries = data[0 .. 5 * dim];
        try testing.expectEqualSlices(flat.Entry, index.searchBatch(queries, &sa), loaded.searchBatch(queries, &sb));
    }
}

test "partitioned files are refused when damaged, and each kind only by its reader" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xBAD2);
    const dim = 32;
    const data = try clustered(a, 60, dim, prng.random());
    defer a.free(data);
    var index = try IvfIndex.init(a, .{ .dim = dim, .bits = 3, .lists = 4, .seed = 5 });
    defer index.deinit();

    var buf: [16]u8 = undefined;
    try testing.expectError(error.NotTrained, writePartitioned(&index, &buf));
    try index.train(data, 1);
    try index.addBatch(data);
    const good = try saved(a, &index);
    defer a.free(good);
    try testing.expectError(error.BufferTooSmall, writePartitioned(&index, good[0 .. good.len - 1]));

    // Each reader refuses the other kind, as such.
    try testing.expectError(error.WrongKind, read(a, good));
    var fl = try FlatIndex.init(a, .{ .dim = dim, .bits = 3 });
    defer fl.deinit();
    const flat_bytes = try a.alloc(u8, serializedSize(&fl));
    defer a.free(flat_bytes);
    _ = try write(&fl, flat_bytes);
    try testing.expectEqual(Kind.flat, try kind(flat_bytes));
    try testing.expectError(error.WrongKind, readPartitioned(a, flat_bytes));

    const bad = try a.alloc(u8, good.len);
    defer a.free(bad);
    for (0..good.len) |i| {
        @memcpy(bad, good);
        bad[i] ^= 0x10;
        if (readPartitioned(a, bad)) |loaded| {
            var l = loaded;
            l.deinit();
            return error.TestUnexpectedResult;
        } else |_| {}
    }
    for ([_]usize{ 7, 56, 100, good.len - 1 }) |len| {
        try testing.expect(std.meta.isError(readPartitioned(a, good[0..len])));
    }

    // A consistent checksum over inconsistent contents still fails the field checks.
    const sizes_at = header_len + 4 * 4 * @as(usize, index.quantizer.padded());
    const ids_at = good.len - crc_len - 4 * index.count();
    const Edit = struct { at: usize, value: u32 };
    for ([_]Edit{
        .{ .at = 52, .value = ivf.max_lists + 1 }, // too many lists
        .{ .at = sizes_at, .value = get(u32, good, sizes_at) + 1 }, // sizes not summing to count
        .{ .at = ids_at, .value = get(u32, good, ids_at + 4) }, // a repeated id
        .{ .at = ids_at, .value = @intCast(index.count()) }, // an id past the end
        .{ .at = 19, .value = get(u32, good, 19) | 1 }, // the sketch correction
    }) |e| {
        @memcpy(bad, good);
        put(u32, bad, e.at, e.value);
        put(u32, bad, bad.len - 4, std.hash.Crc32.hash(bad[0 .. bad.len - 4]));
        try testing.expectError(error.Corrupt, readPartitioned(a, bad));
    }
    // Ids out of order within a list: swap list 0's first two.
    const first = get(u32, good, ids_at);
    const second = get(u32, good, ids_at + 4);
    try testing.expect(get(u32, good, sizes_at) >= 2);
    @memcpy(bad, good);
    put(u32, bad, ids_at, second);
    put(u32, bad, ids_at + 4, first);
    put(u32, bad, bad.len - 4, std.hash.Crc32.hash(bad[0 .. bad.len - 4]));
    try testing.expectError(error.Corrupt, readPartitioned(a, bad));
}

test "an empty partitioned index round-trips" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xE);
    const data = try clustered(a, 40, 16, prng.random());
    defer a.free(data);
    var index = try IvfIndex.init(a, .{ .dim = 16, .bits = 4, .lists = 3, .seed = 8 });
    defer index.deinit();
    try index.train(data, 1);
    const bytes = try saved(a, &index);
    defer a.free(bytes);
    var loaded = try readPartitioned(a, bytes);
    defer loaded.deinit();
    try testing.expectEqual(@as(usize, 0), loaded.count());
    try testing.expect(loaded.trained);
    try loaded.addBatch(data);
    try testing.expectEqual(@as(usize, 40), loaded.count());
}
