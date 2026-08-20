//! Bounded top-k collector.
//!
//! A min-heap of size k over scores. The root is the weakest kept candidate, so
//! `offer` rejects with a single compare in the overwhelmingly common case — for
//! k ≪ n the heap is barely touched after the first few thousand candidates.
//!
//! docs/DESIGN.md §4.4 described a SIMD gate: compare 32 block scores against a
//! broadcast threshold and `movemask` to find survivors. That shape belonged to the
//! dimension-major layout, which produced 32 scores at once. Row-major storage
//! (see `quant/packing.zig`) produces one score at a time, so the gate is a scalar
//! compare — which was always the cheap part; the win was never the compare itself
//! but avoiding the heap.

const std = @import("std");

pub const Entry = struct {
    id: u32,
    score: f32,
};

pub const TopK = struct {
    /// Min-heap on score. Capacity is k.
    heap: []Entry,
    len: usize,

    /// `buffer.len` is k. Caller owns it, so a search allocates nothing per query
    /// beyond its result slice.
    pub fn init(buffer: []Entry) TopK {
        return .{ .heap = buffer, .len = 0 };
    }

    pub fn capacity(self: TopK) usize {
        return self.heap.len;
    }

    /// Score a candidate must beat. Negative infinity until k candidates are held,
    /// so nothing is rejected before the heap fills.
    pub fn threshold(self: TopK) f32 {
        return if (self.len < self.heap.len) -std.math.inf(f32) else self.heap[0].score;
    }

    pub fn offer(self: *TopK, score: f32, id: u32) void {
        if (self.len < self.heap.len) {
            self.heap[self.len] = .{ .id = id, .score = score };
            self.len += 1;
            self.siftUp(self.len - 1);
            return;
        }
        // The hot path: one compare, no memory traffic.
        //
        // Strictly greater, so ties keep the earlier id. That makes results stable
        // under insertion order, which matters for reproducible benchmarks on
        // corpora with duplicate vectors.
        if (score <= self.heap[0].score) return;
        self.heap[0] = .{ .id = id, .score = score };
        self.siftDown(0);
    }

    /// Sorts the heap in place, descending, and returns it. The collector is spent
    /// afterwards.
    pub fn drain(self: *TopK) []Entry {
        const out = self.heap[0..self.len];
        std.mem.sort(Entry, out, {}, struct {
            fn desc(_: void, a: Entry, b: Entry) bool {
                if (a.score != b.score) return a.score > b.score;
                return a.id < b.id;
            }
        }.desc);
        self.len = 0;
        return out;
    }

    fn siftUp(self: *TopK, start: usize) void {
        var i = start;
        while (i > 0) {
            const parent = (i - 1) / 2;
            if (self.heap[parent].score <= self.heap[i].score) break;
            std.mem.swap(Entry, &self.heap[parent], &self.heap[i]);
            i = parent;
        }
    }

    fn siftDown(self: *TopK, start: usize) void {
        var i = start;
        while (true) {
            const left = 2 * i + 1;
            if (left >= self.len) break;
            const right = left + 1;
            var smallest = left;
            if (right < self.len and self.heap[right].score < self.heap[left].score) {
                smallest = right;
            }
            if (self.heap[i].score <= self.heap[smallest].score) break;
            std.mem.swap(Entry, &self.heap[i], &self.heap[smallest]);
            i = smallest;
        }
    }
};

// -- tests -------------------------------------------------------------------

const testing = std.testing;

/// Brute-force reference: sort everything, take the first k.
fn referenceTopK(
    allocator: std.mem.Allocator,
    scores: []const f32,
    k: usize,
) ![]Entry {
    const all = try allocator.alloc(Entry, scores.len);
    defer allocator.free(all);
    for (all, scores, 0..) |*e, s, i| e.* = .{ .id = @intCast(i), .score = s };
    std.mem.sort(Entry, all, {}, struct {
        fn desc(_: void, a: Entry, b: Entry) bool {
            if (a.score != b.score) return a.score > b.score;
            return a.id < b.id;
        }
    }.desc);
    const out = try allocator.alloc(Entry, @min(k, scores.len));
    @memcpy(out, all[0..out.len]);
    return out;
}

test "matches a full sort on random scores" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const random = prng.random();

    for ([_]usize{ 1, 2, 5, 10, 64, 100 }) |k| {
        for ([_]usize{ 0, 1, 7, 63, 1000 }) |n| {
            const scores = try allocator.alloc(f32, n);
            defer allocator.free(scores);
            for (scores) |*s| s.* = random.floatNorm(f32);

            const buffer = try allocator.alloc(Entry, k);
            defer allocator.free(buffer);
            var collector = TopK.init(buffer);
            for (scores, 0..) |s, i| collector.offer(s, @intCast(i));

            const got = collector.drain();
            const want = try referenceTopK(allocator, scores, k);
            defer allocator.free(want);

            try testing.expectEqual(want.len, got.len);
            for (want, got) |w, g| {
                try testing.expectEqual(w.id, g.id);
                try testing.expectEqual(w.score, g.score);
            }
        }
    }
}

test "results are sorted descending" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(3);

    const buffer = try allocator.alloc(Entry, 20);
    defer allocator.free(buffer);
    var collector = TopK.init(buffer);
    for (0..500) |i| collector.offer(prng.random().floatNorm(f32), @intCast(i));

    const got = collector.drain();
    for (1..got.len) |i| try testing.expect(got[i - 1].score >= got[i].score);
}

test "threshold tracks the weakest kept candidate" {
    const allocator = testing.allocator;
    const buffer = try allocator.alloc(Entry, 3);
    defer allocator.free(buffer);
    var collector = TopK.init(buffer);

    // Nothing is rejected before the heap fills.
    try testing.expectEqual(-std.math.inf(f32), collector.threshold());
    collector.offer(5.0, 0);
    collector.offer(1.0, 1);
    try testing.expectEqual(-std.math.inf(f32), collector.threshold());

    collector.offer(3.0, 2);
    try testing.expectEqual(@as(f32, 1.0), collector.threshold());

    // A better candidate raises it; a worse one does not.
    collector.offer(4.0, 3);
    try testing.expectEqual(@as(f32, 3.0), collector.threshold());
    collector.offer(0.5, 4);
    try testing.expectEqual(@as(f32, 3.0), collector.threshold());
}

test "ties keep the earlier id" {
    // Makes results stable under insertion order, which corpora with duplicate
    // vectors otherwise make nondeterministic.
    const allocator = testing.allocator;
    const buffer = try allocator.alloc(Entry, 2);
    defer allocator.free(buffer);
    var collector = TopK.init(buffer);

    for (0..10) |i| collector.offer(1.0, @intCast(i));
    const got = collector.drain();
    try testing.expectEqual(@as(u32, 0), got[0].id);
    try testing.expectEqual(@as(u32, 1), got[1].id);
}

test "fewer candidates than k" {
    const allocator = testing.allocator;
    const buffer = try allocator.alloc(Entry, 10);
    defer allocator.free(buffer);
    var collector = TopK.init(buffer);

    collector.offer(1.0, 0);
    collector.offer(2.0, 1);
    const got = collector.drain();
    try testing.expectEqual(@as(usize, 2), got.len);
    try testing.expectEqual(@as(u32, 1), got[0].id);
}

test "handles infinities and equal scores throughout" {
    const allocator = testing.allocator;
    const buffer = try allocator.alloc(Entry, 4);
    defer allocator.free(buffer);
    var collector = TopK.init(buffer);

    collector.offer(-std.math.inf(f32), 0);
    collector.offer(std.math.inf(f32), 1);
    collector.offer(0.0, 2);
    collector.offer(std.math.inf(f32), 3);
    collector.offer(-1.0, 4);

    const got = collector.drain();
    try testing.expectEqual(@as(usize, 4), got.len);
    try testing.expectEqual(@as(u32, 1), got[0].id);
    try testing.expectEqual(@as(u32, 3), got[1].id);
    try testing.expectEqual(@as(u32, 2), got[2].id);
    try testing.expectEqual(@as(u32, 4), got[3].id);
}

test "reusable across queries" {
    const allocator = testing.allocator;
    const buffer = try allocator.alloc(Entry, 3);
    defer allocator.free(buffer);
    var collector = TopK.init(buffer);

    for (0..2) |round| {
        for (0..100) |i| collector.offer(@floatFromInt(i), @intCast(i));
        const got = collector.drain();
        try testing.expectEqual(@as(usize, 3), got.len);
        try testing.expectEqual(@as(u32, 99), got[0].id);
        try testing.expectEqual(@as(usize, 0), collector.len);
        _ = round;
    }
}
