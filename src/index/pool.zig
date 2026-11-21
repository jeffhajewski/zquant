//! Work shared out a chunk at a time, so fast cores aren't left waiting on slow ones.
//!
//! Splitting work evenly ahead of time runs at the slowest thread's pace. On a machine
//! with performance and efficiency cores that's the efficiency cores' pace: building a
//! million-vector index on an M5 with 8 threads, k-means assigned 80k rows a second per
//! thread where one thread alone managed 165k. Here each worker takes the next chunk
//! from a shared counter until none are left. Callers make each chunk's result
//! independent of which worker ran it, so nothing depends on the schedule or on the
//! number of threads.

const std = @import("std");

/// At most this many workers, the calling thread among them.
pub const max_workers = 64;

/// How many workers `forChunks` uses for `n` items in chunks of `chunk`: per-worker
/// scratch needs this many.
pub fn workers(threads: usize, n: usize, chunk: usize) usize {
    return @max(1, @min(threads, max_workers, std.math.divCeil(usize, n, chunk) catch unreachable));
}

/// Call `run(ctx, worker, start, len)` for consecutive chunks of `chunk` items covering
/// `0..n`, on `workers(threads, n, chunk)` threads including this one; `worker` is below
/// that. A thread that fails to spawn just leaves its chunks to the others.
pub fn forChunks(
    threads: usize,
    n: usize,
    chunk: usize,
    ctx: anytype,
    comptime run: fn (@TypeOf(ctx), worker: usize, start: usize, len: usize) void,
) void {
    std.debug.assert(chunk > 0);
    const count = workers(threads, n, chunk);
    var next = std.atomic.Value(usize).init(0);
    const Worker = struct {
        fn go(c: @TypeOf(ctx), w: usize, counter: *std.atomic.Value(usize), total: usize, size: usize) void {
            while (true) {
                const start = counter.fetchAdd(size, .monotonic);
                if (start >= total) return;
                run(c, w, start, @min(size, total - start));
            }
        }
    };
    var handles: [max_workers]?std.Thread = @splat(null);
    for (1..count) |w| handles[w] = std.Thread.spawn(.{}, Worker.go, .{ ctx, w, &next, n, chunk }) catch null;
    Worker.go(ctx, 0, &next, n, chunk);
    for (handles[1..count]) |h| if (h) |t| t.join();
}

test "every item is visited once, whatever the thread count" {
    const n = 1000;
    for ([_]usize{ 1, 3, 8, 100 }) |threads| {
        for ([_]usize{ 1, 7, 64, 2000 }) |chunk| {
            var seen: [n]std.atomic.Value(u32) = undefined;
            for (&seen) |*s| s.* = .init(0);
            const Ctx = struct {
                seen: *[n]std.atomic.Value(u32),
                limit: usize,
                fn run(c: @This(), w: usize, start: usize, len: usize) void {
                    std.debug.assert(w < c.limit);
                    for (start..start + len) |i| _ = c.seen[i].fetchAdd(1, .monotonic);
                }
            };
            forChunks(threads, n, chunk, Ctx{ .seen = &seen, .limit = workers(threads, n, chunk) }, Ctx.run);
            for (&seen) |*s| try std.testing.expectEqual(@as(u32, 1), s.load(.monotonic));
        }
    }
    // Nothing to do is fine too.
    forChunks(4, 0, 8, @as(u8, 0), struct {
        fn run(_: u8, _: usize, _: usize, _: usize) void {
            unreachable;
        }
    }.run);
}
