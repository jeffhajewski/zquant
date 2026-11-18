//! The search loop's scan kernels, as two calls with C-compatible signatures.
//!
//! The kernels choose their instructions at compile time, so a library built for
//! x86_64_v2 (as releases are) uses 128-bit SSSE3 kernels even on a CPU with AVX2,
//! which runs the 256-bit ones about 1.7x faster. `dispatch.zig` fixes that by linking
//! this file in again, compiled for a higher CPU level (see `kernel_entry.zig`), and
//! choosing between the copies at run time. These two calls are the seam: each scores
//! a whole tile, so one indirect call covers dozens of vector-query pairs.
//!
//! Without dispatch, `FlatIndex` calls the same functions directly.

const std = @import("std");
const scan = @import("scan.zig");
const dot = @import("dot.zig");
const shuffle = @import("shuffle.zig");
const packing = @import("../quant/packing.zig");

/// The batch tile: vectors x queries scored together. bench/tile_bench.zig measured
/// seven shapes on x86_64 (Zen 3 and Zen 5) and arm64 (Apple M5, Neoverse N2), and no
/// other shape was reliably faster on any of them; docs/notes.md has the numbers.
pub const batch_tile = 4;
pub const batch_group = 4;
/// Vectors per tile for a single query. Q=1 leaves registers free, so it is wider.
pub const single_tile = 8;

/// One search's scan inputs. `table` and `queries` are `*const scan.Table` and
/// `[*]const scan.Query`. They are Zig structs, which have no guaranteed layout across
/// separate compilations, so a level whose `abi` differs from this build's is never
/// used (`dispatch.zig`).
pub const Job = extern struct {
    table: *const anyopaque,
    queries: *const anyopaque,
    /// Queries in the batch.
    n: usize,
    codes: [*]const u8,
    /// Bytes per stored vector.
    stride: usize,
    /// Working (padded) dimension.
    padded: u32,
    /// Code bits of the compact layout; unused when `expanded`.
    bits: u8,
    expanded: bool,
    /// The expanded residency's dequantization scale.
    table_scale: f32,
    /// Scores go to `out[v * out_stride + q]`.
    out: [*]f32,
    out_stride: usize,
};

/// Scores `batch_tile` stored vectors starting at `v0` against all `job.n` queries.
/// The layout must be expanded or tileable (`scan.canTile`).
pub fn batchTile(job: *const Job, v0: usize) void {
    const queries = queriesOf(job);
    const os = job.out_stride;
    const base = job.codes + v0 * job.stride;
    var i: usize = 0;
    if (job.expanded) {
        const values: [*]const i8 = @ptrCast(base);
        while (i + batch_group <= job.n) : (i += batch_group) {
            scan.scoreExpandedTiled(batch_tile, batch_group, values, job.stride, queries[i..], job.table_scale, job.padded, outAt(job, i, batch_tile, batch_group), os);
        }
        while (i < job.n) : (i += 1) {
            for (0..batch_tile) |u| {
                job.out[u * os + i] = scan.scoreExpanded(values[u * job.stride ..][0..job.stride], queries[i], job.table_scale, job.padded);
            }
        }
    } else {
        const table: *const scan.Table = @ptrCast(@alignCast(job.table));
        const layout = packing.Layout.init(job.padded, @intCast(job.bits));
        while (i + batch_group <= job.n) : (i += batch_group) {
            scan.scoreInt8Tiled(batch_tile, batch_group, layout, table.*, queries[i..], base, job.stride, job.padded, outAt(job, i, batch_tile, batch_group), os);
        }
        while (i < job.n) : (i += 1) {
            for (0..batch_tile) |u| {
                job.out[u * os + i] = scan.scoreInt8(layout, table.*, queries[i], base[u * job.stride ..][0..job.stride], job.padded);
            }
        }
    }
}

/// Scores `tiles * single_tile` stored vectors starting at `v0` against query 0, into
/// `out[0 ..]` (`out_stride` is not used).
pub fn singleTiles(job: *const Job, v0: usize, tiles: usize) void {
    const queries = queriesOf(job)[0..1];
    for (0..tiles) |t| {
        const first = v0 + t * single_tile;
        const out = job.out[t * single_tile ..][0..single_tile];
        if (job.expanded) {
            scan.scoreExpandedTiled(single_tile, 1, @ptrCast(job.codes + first * job.stride), job.stride, queries, job.table_scale, job.padded, out, 1);
        } else {
            const table: *const scan.Table = @ptrCast(@alignCast(job.table));
            const layout = packing.Layout.init(job.padded, @intCast(job.bits));
            scan.scoreInt8Tiled(single_tile, 1, layout, table.*, queries, job.codes + first * job.stride, job.stride, job.padded, out, 1);
        }
    }
}

fn queriesOf(job: *const Job) []const scan.Query {
    const many: [*]const scan.Query = @ptrCast(@alignCast(job.queries));
    return many[0..job.n];
}

/// The part of `out` a V x Q tile at query `i` writes.
fn outAt(job: *const Job, i: usize, comptime V: usize, comptime Q: usize) []f32 {
    return job.out[i..][0 .. (V - 1) * job.out_stride + Q];
}

/// A set of kernels, as one compilation exports it.
pub const Kernels = extern struct {
    /// `abi` of the compilation that built these; see `Job`.
    abi: u64,
    /// What these kernels compiled to, in `zq_build_kernels`'s format.
    description: [*:0]const u8,
    batch_tile: *const fn (job: *const Job, v0: usize) callconv(.c) void,
    single_tiles: *const fn (job: *const Job, v0: usize, tiles: usize) callconv(.c) void,
};

fn batchTileC(job: *const Job, v0: usize) callconv(.c) void {
    batchTile(job, v0);
}

fn singleTilesC(job: *const Job, v0: usize, tiles: usize) callconv(.c) void {
    singleTiles(job, v0, tiles);
}

/// This compilation's kernels.
pub const kernels: Kernels = .{
    .abi = abi,
    .description = description,
    .batch_tile = &batchTileC,
    .single_tiles = &singleTilesC,
};

/// What this compilation's kernels are. The prefix is the marker CI greps release
/// binaries for; `wide=` names the 256-bit kernels where they are compiled in.
pub const description = std.fmt.comptimePrint("zquant-kernels shuffle={s} dot={s}{s}", .{
    if (shuffle.accelerated) "native" else "scalar",
    if (dot.accelerated) "native" else "scalar",
    if (dot.has_vnni) " wide=avx2+vnni" else if (dot.has_avx2) " wide=avx2" else "",
});

/// A fingerprint of the layouts `Job` passes opaquely. Two compilations agree on it
/// exactly when they agree on those layouts.
pub const abi: u64 = blk: {
    var h: u64 = 0xcbf29ce484222325;
    for ([_]type{ scan.Query, scan.Table }) |T| {
        h = mix(h, @sizeOf(T));
        h = mix(h, @alignOf(T));
        for (@typeInfo(T).@"struct".fields) |f| {
            h = mix(h, @offsetOf(T, f.name));
            h = mix(h, @sizeOf(f.type));
        }
    }
    break :blk h;
};

fn mix(h: u64, x: u64) u64 {
    return (h ^ x) *% 0x100000001b3;
}
