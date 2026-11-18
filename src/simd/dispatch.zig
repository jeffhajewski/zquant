//! Run-time choice among scan kernels compiled for several CPU levels.
//!
//! The library's own kernels are compiled for the CPU it was built for, x86_64_v2 in a
//! release, and that build cannot use AVX2 even where the CPU has it. build.zig also
//! compiles `tiles.zig` for each higher level (`kernel_entry.zig`) and lists them here,
//! best first, with the features each needs. The first whose features this CPU has is
//! used; with none, the library's own. A build that already targets the highest level,
//! such as a native one, has no levels and calls the kernels directly.
//!
//! `ZQUANT_KERNELS=<level>` restricts the choice to that level, and any other value
//! (`none`, say) to the library's own kernels: for testing and for comparing the two.
//! It never selects kernels the CPU cannot run.

const std = @import("std");
const builtin = @import("builtin");
const tiles = @import("tiles.zig");
const detect = @import("../cpu_detect.zig");
const options = @import("dispatch_options");

const Level = struct {
    name: []const u8,
    /// Space-separated features the level was compiled with.
    required: []const u8,
    kernels: *const tiles.Kernels,
};

/// From build.zig's `levels`: `name:feature feature;name:feature ...`, best first.
const levels: []const Level = blk: {
    var list: []const Level = &.{};
    var it = std.mem.tokenizeScalar(u8, options.levels, ';');
    while (it.next()) |entry| {
        const colon = std.mem.indexOfScalar(u8, entry, ':').?;
        const name = entry[0..colon];
        list = list ++ [_]Level{.{
            .name = name,
            .required = entry[colon + 1 ..],
            .kernels = @extern(*const tiles.Kernels, .{ .name = "zq_kernels_" ++ name }),
        }};
    }
    break :blk list;
};

/// 0: not chosen yet. 1: the library's own. k + 2: `levels[k]`.
var choice = std.atomic.Value(usize).init(0);

/// The kernels to use, or null for the library's own. Chosen once; a race between two
/// first calls is harmless, since both choose the same.
pub fn active() ?*const tiles.Kernels {
    if (comptime levels.len == 0) return null;
    var c = choice.load(.acquire);
    if (c == 0) {
        c = choose();
        choice.store(c, .release);
    }
    return if (c == 1) null else levels[c - 2].kernels;
}

fn choose() usize {
    const forced: ?[]const u8 = if (builtin.link_libc)
        if (std.c.getenv("ZQUANT_KERNELS")) |v| std.mem.span(v) else null
    else
        null;
    for (levels, 0..) |level, k| {
        if (forced) |name| if (!std.mem.eql(u8, name, level.name)) continue;
        // A level built by a compilation that lays out the shared structs differently
        // would read garbage; build.zig makes that impossible, and this makes sure.
        if (level.kernels.abi != tiles.abi) continue;
        if (detect.hasAll(level.required)) return k + 2;
    }
    return 1;
}

/// What the kernels in use compiled to, for `zq_build_kernels`.
pub fn description() [*:0]const u8 {
    return if (active()) |k| k.description else tiles.description;
}

pub inline fn batchTile(kernels: ?*const tiles.Kernels, job: *const tiles.Job, v0: usize) void {
    if (kernels) |k| k.batch_tile(job, v0) else tiles.batchTile(job, v0);
}

pub inline fn singleTiles(kernels: ?*const tiles.Kernels, job: *const tiles.Job, v0: usize, count: usize) void {
    if (kernels) |k| k.single_tiles(job, v0, count) else tiles.singleTiles(job, v0, count);
}

test "every level this CPU can run scores exactly as the library's own kernels" {
    // The levels are separate compilations reached through `Job`, so this checks the
    // seam as well as the kernels: a mistake in either shows up as a difference, and
    // the sums are integers, so any difference is one.
    const testing = std.testing;
    const scan = @import("scan.zig");
    const packing = @import("../quant/packing.zig");

    var runnable: usize = 0;
    var prng = std.Random.DefaultPrng.init(0xd15);
    const rand = prng.random();
    const n = 7; // a group of 4 and a tail of 3
    const vectors = 2 * tiles.single_tile;

    for (levels) |level| {
        if (level.kernels.abi != tiles.abi or !detect.hasAll(level.required)) continue;
        runnable += 1;

        for ([_]?u6{ 1, 2, 3, 4, null }) |maybe_bits| {
            const dim: u32 = 384;
            const expanded = maybe_bits == null;
            const layout = packing.Layout.init(dim, maybe_bits orelse 4);
            if (!expanded and !scan.canTile(layout)) continue;
            const stride = if (expanded) dim else layout.stride();
            const codes = try testing.allocator.alloc(u8, vectors * stride);
            defer testing.allocator.free(codes);
            rand.bytes(codes);

            var centroids: [16]f32 = undefined;
            for (&centroids) |*c| c.* = rand.floatNorm(f32);
            const table = scan.Table.init(centroids[0 .. @as(usize, 1) << layout.bits]);
            const rotated = try testing.allocator.alloc(f32, dim);
            defer testing.allocator.free(rotated);
            var queries: [n]scan.Query = undefined;
            for (&queries) |*q| {
                q.* = if (expanded)
                    try scan.Query.initSequential(testing.allocator, dim)
                else
                    try scan.Query.init(testing.allocator, layout);
                for (rotated) |*x| x.* = rand.floatNorm(f32);
                q.load(rotated);
            }
            defer for (&queries) |*q| q.deinit();

            var own: [vectors * n]f32 = undefined;
            var theirs: [vectors * n]f32 = undefined;
            var job: tiles.Job = .{
                .table = &table,
                .queries = &queries,
                .n = n,
                .codes = codes.ptr,
                .stride = stride,
                .padded = dim,
                .bits = layout.bits,
                .expanded = expanded,
                .table_scale = 0.01,
                .out = &own,
                .out_stride = n,
            };
            tiles.batchTile(&job, tiles.batch_tile);
            job.out = &theirs;
            level.kernels.batch_tile(&job, tiles.batch_tile);
            try testing.expectEqualSlices(f32, own[0 .. tiles.batch_tile * n], theirs[0 .. tiles.batch_tile * n]);

            job.n = 1;
            job.out = &own;
            tiles.singleTiles(&job, 0, 2);
            job.out = &theirs;
            level.kernels.single_tiles(&job, 0, 2);
            try testing.expectEqualSlices(f32, own[0..vectors], theirs[0..vectors]);
        }
    }
    if (runnable == 0) return error.SkipZigTest;
}
