//! Exhaustive-scan index over quantized codes.
//!
//! Owns a `Prod` quantizer and the packed corpus, and answers top-k queries by
//! scoring every vector. No partitioning — that is `index/ivf.zig` later. The point
//! of this layer is that the quantizer, packing, and both kernels compose into
//! something callable, and that the full estimator is used rather than half of it.
//!
//! ## The score
//!
//!     ⟨q, x⟩ ≈ ‖x‖ · [ ⟨p, ỹ⟩ + γ·sketch_scale·⟨S'p, qjl⟩ ]
//!                     └ simd/scan ┘         └ simd/sketch ┘
//!
//! Both terms, always. The first alone is the MSE-only estimate that `prod` exists to
//! correct, biased by 2/π at one MSE bit.
//!
//! ## Bit-widths
//!
//! `prod` at total width `b` spends `b−1` bits on codes, and every width from 2 to 5
//! vectorizes: those dividing a byte use sequential packing, 3-bit codes use
//! bit-planes (`quant/packing.zig`). Beyond that the codebook outgrows the 16-entry
//! shuffle table and the exact f32 scan takes over — no loss in practice, since b=5
//! already reaches 0.995 recall.

const std = @import("std");
const Allocator = std.mem.Allocator;

const prod_mod = @import("../quant/prod.zig");
const packing = @import("../quant/packing.zig");
const scan = @import("../simd/scan.zig");
const sketch_kernel = @import("../simd/sketch.zig");
const topk = @import("topk.zig");
const RotationKind = @import("../math/rotation.zig").Kind;

pub const Entry = topk.Entry;

pub const Metric = enum {
    /// Raw ⟨q, x⟩. Larger is better.
    inner_product,
    /// ⟨q, x⟩ / (‖q‖·‖x‖). Larger is better.
    cosine,
    /// Negated squared Euclidean distance, so larger is still better and one
    /// comparison rule serves every metric.
    l2,
};

pub const Params = struct {
    dim: u32,
    /// Total bits per coordinate: `bits − 1` for codes plus one sketch bit.
    bits: u6,
    metric: Metric = .inner_product,
    seed: u64 = 0,
    rotation: RotationKind = .hadamard,
    /// Force the exact f32 scan even where the int8 kernel applies. The int8 path
    /// adds bounded query-side error (docs/DESIGN.md §4.2) and is the default only
    /// because that error is measured; this is the escape hatch.
    exact_scan: bool = false,
};

/// Half-precision per-vector scalars. See the field comment on `FlatIndex.scalars`.
pub const StoredScalars = struct {
    norm: f16,
    gamma: f16,

    fn from(s: prod_mod.Scalars) StoredScalars {
        return .{ .norm = @floatCast(s.norm), .gamma = @floatCast(s.gamma) };
    }
};

pub const FlatIndex = struct {
    quantizer: prod_mod.Prod,
    layout: packing.Layout,
    metric: Metric,
    exact_scan: bool,

    /// Packed codes, `layout.stride()` bytes per vector.
    codes: std.ArrayList(u8),
    /// QJL sign bitmaps, `sketchLen()` bytes per vector.
    sketches: std.ArrayList(u8),
    /// ‖x‖ and ‖y − ỹ‖ per vector, at half precision.
    ///
    /// f16 rather than f32 because these are per *vector*, not per coordinate, so
    /// they do not shrink with the bit budget: 8 bytes of f32 is under 2% of a
    /// d=1024 vector but 11% of a d=128 one, which is exactly the KV-cache regime.
    /// f16 carries ~3 decimal digits, far more than needed for a scale factor whose
    /// own inputs are quantized to 4 bits.
    scalars: std.ArrayList(StoredScalars),

    workspace: prod_mod.Workspace,
    /// Reusable unpacked-code buffer for `add`. Held here rather than allocated per
    /// call: `add` runs once per vector, and a transient allocation per vector is
    /// both wasteful and needless pressure on the allocator during a bulk load.
    encode_scratch: []u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, params: Params) !FlatIndex {
        var quantizer = try prod_mod.Prod.init(allocator, .{
            .dim = params.dim,
            .bits = params.bits,
            .seed = params.seed,
            .rotation = params.rotation,
        });
        errdefer quantizer.deinit();

        var workspace = try prod_mod.Workspace.init(allocator, quantizer);
        errdefer workspace.deinit();

        const encode_scratch = try allocator.alloc(u8, quantizer.codeLen());
        errdefer allocator.free(encode_scratch);

        return .{
            .quantizer = quantizer,
            .layout = packing.Layout.init(quantizer.padded(), quantizer.mse.bits),
            .metric = params.metric,
            .exact_scan = params.exact_scan,
            .codes = .empty,
            .sketches = .empty,
            .scalars = .empty,
            .workspace = workspace,
            .encode_scratch = encode_scratch,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FlatIndex) void {
        self.allocator.free(self.encode_scratch);
        self.codes.deinit(self.allocator);
        self.sketches.deinit(self.allocator);
        self.scalars.deinit(self.allocator);
        self.workspace.deinit();
        self.quantizer.deinit();
        self.* = undefined;
    }

    pub fn count(self: FlatIndex) usize {
        return self.scalars.items.len;
    }

    pub fn dim(self: FlatIndex) u32 {
        return self.quantizer.dim();
    }

    /// Whether queries take the vectorized path. False when the MSE stage's
    /// bit-width does not divide a byte — notably `bits = 4`.
    pub fn vectorized(self: FlatIndex) bool {
        return !self.exact_scan and
            scan.canVectorize(self.layout) and
            sketch_kernel.canVectorize(self.quantizer.padded());
    }

    /// Total bytes of index storage per vector, scalars included.
    ///
    /// Includes them deliberately: an earlier version reported only codes and
    /// sketch, which overstated the compression ratio by 11% at d=128.
    pub fn bytesPerVector(self: FlatIndex) usize {
        return self.layout.stride() + self.quantizer.sketchLen() + @sizeOf(StoredScalars);
    }

    /// Bytes of codes and sketch only — the part that scales with the bit budget.
    pub fn codeBytesPerVector(self: FlatIndex) usize {
        return self.layout.stride() + self.quantizer.sketchLen();
    }

    pub fn reserve(self: *FlatIndex, additional: usize) !void {
        try self.codes.ensureUnusedCapacity(self.allocator, additional * self.layout.stride());
        try self.sketches.ensureUnusedCapacity(self.allocator, additional * self.quantizer.sketchLen());
        try self.scalars.ensureUnusedCapacity(self.allocator, additional);
    }

    /// Encode and append one vector. Returns its id.
    ///
    /// Ids are sequential and stable; nothing here removes or reorders. Encoding is
    /// data-oblivious, so a vector's codes never depend on what was added before it.
    pub fn add(self: *FlatIndex, vector: []const f32) !u32 {
        std.debug.assert(vector.len == self.dim());

        const stride = self.layout.stride();
        const sketch_len = self.quantizer.sketchLen();

        // Encode into scratch, then pack. The quantizer emits one byte per
        // coordinate; storage wants them bit-packed.
        const scratch = self.encode_scratch;

        try self.codes.appendNTimes(self.allocator, 0, stride);
        errdefer self.codes.shrinkRetainingCapacity(self.codes.items.len - stride);
        try self.sketches.appendNTimes(self.allocator, 0, sketch_len);
        errdefer self.sketches.shrinkRetainingCapacity(self.sketches.items.len - sketch_len);

        const codes_slot = self.codes.items[self.codes.items.len - stride ..];
        const sketch_slot = self.sketches.items[self.sketches.items.len - sketch_len ..];

        const scalars = self.quantizer.encode(vector, scratch, sketch_slot, &self.workspace);
        self.layout.pack(scratch, codes_slot);

        try self.scalars.append(self.allocator, StoredScalars.from(scalars));
        return @intCast(self.scalars.items.len - 1);
    }

    /// Append `n` vectors from a row-major buffer.
    pub fn addBatch(self: *FlatIndex, vectors: []const f32) !void {
        const d = self.dim();
        std.debug.assert(vectors.len % d == 0);
        const n = vectors.len / d;
        try self.reserve(n);
        for (0..n) |i| _ = try self.add(vectors[i * d ..][0..d]);
    }

    /// Per-query state. Held by the caller so repeated searches allocate nothing.
    pub const Searcher = struct {
        query_state: prod_mod.QueryState,
        scan_query: scan.Query,
        sketch_query: sketch_kernel.Query,
        table: scan.Table,
        heap: []Entry,
        workspace: prod_mod.Workspace,
        allocator: Allocator,

        pub fn init(allocator: Allocator, index: FlatIndex, k: usize) !Searcher {
            var query_state = try prod_mod.QueryState.init(allocator, index.quantizer);
            errdefer query_state.deinit();
            var scan_query = try scan.Query.init(allocator, index.layout);
            errdefer scan_query.deinit();
            var sketch_query = try sketch_kernel.Query.init(allocator, index.quantizer.padded());
            errdefer sketch_query.deinit();
            var workspace = try prod_mod.Workspace.init(allocator, index.quantizer);
            errdefer workspace.deinit();
            const heap = try allocator.alloc(Entry, k);

            return .{
                .query_state = query_state,
                .scan_query = scan_query,
                .sketch_query = sketch_query,
                // Only build the shuffle table when it will actually be used. A
                // codebook wider than 16 levels does not fit one, and building it
                // anyway overran a 16-entry array — silently, in ReleaseFast, where
                // the bounds assert is compiled out.
                .table = if (index.vectorized())
                    scan.Table.init(index.quantizer.mse.codebook.centroids)
                else
                    scan.unusedTable(),
                .heap = heap,
                .workspace = workspace,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Searcher) void {
            self.allocator.free(self.heap);
            self.workspace.deinit();
            self.sketch_query.deinit();
            self.scan_query.deinit();
            self.query_state.deinit();
            self.* = undefined;
        }
    };

    /// Top-k search. Results are written into `searcher`'s heap and returned as a
    /// slice into it, valid until the next search.
    pub fn search(self: *FlatIndex, query: []const f32, searcher: *Searcher) []Entry {
        std.debug.assert(query.len == self.dim());
        std.debug.assert(searcher.heap.len > 0);

        // One rotation and one sketch for the whole corpus — the point of staying in
        // the rotated basis (docs/DESIGN.md §1.3).
        self.quantizer.prepareQuery(query, &searcher.query_state, &searcher.workspace);
        const use_simd = self.vectorized();
        if (use_simd) {
            searcher.scan_query.load(searcher.query_state.rotated);
            searcher.sketch_query.load(searcher.query_state.sketched);
        }

        const query_norm = if (self.metric == .cosine or self.metric == .l2)
            euclideanNorm(query)
        else
            0;

        var collector = topk.TopK.init(searcher.heap);
        const stride = self.layout.stride();
        const sketch_len = self.quantizer.sketchLen();
        const padded = self.quantizer.padded();
        const centroids = self.quantizer.mse.codebook.centroids;

        for (self.scalars.items, 0..) |scalars, i| {
            const code_slot = self.codes.items[i * stride ..][0..stride];
            const sketch_slot = self.sketches.items[i * sketch_len ..][0..sketch_len];

            const mse_term = if (use_simd)
                scan.scoreInt8(self.layout, searcher.table, searcher.scan_query, code_slot, padded)
            else
                scan.scoreExact(self.layout, centroids, searcher.query_state.rotated, code_slot);

            const sketch_term = if (use_simd)
                sketch_kernel.signDot(searcher.sketch_query, sketch_slot, padded)
            else
                sketch_kernel.signDotExact(searcher.query_state.sketched, sketch_slot);

            const norm: f32 = scalars.norm;
            const gamma: f32 = scalars.gamma;
            const estimate = norm * (mse_term + gamma * self.quantizer.sketch_scale * sketch_term);

            collector.offer(self.score(estimate, norm, query_norm), @intCast(i));
        }

        return collector.drain();
    }

    /// Map an inner-product estimate onto the metric, always larger-is-better.
    fn score(self: FlatIndex, estimate: f32, norm: f32, query_norm: f32) f32 {
        return switch (self.metric) {
            .inner_product => estimate,
            // A zero vector has no direction; give it the worst possible score
            // rather than a NaN that would corrupt the heap ordering.
            .cosine => if (norm > 0 and query_norm > 0)
                estimate / (norm * query_norm)
            else
                -std.math.inf(f32),
            // ‖q−x‖² = ‖q‖² + ‖x‖² − 2⟨q,x⟩, negated so larger is better. The ‖q‖²
            // term is constant per query but kept so scores are meaningful values
            // rather than only comparable ones.
            .l2 => -(query_norm * query_norm + norm * norm - 2.0 * estimate),
        };
    }
};

fn euclideanNorm(x: []const f32) f32 {
    var sum: f64 = 0;
    for (x) |v| sum += @as(f64, v) * v;
    return @floatCast(@sqrt(sum));
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;

fn randomUnit(buf: []f32, random: std.Random) void {
    var norm: f64 = 0;
    for (buf) |*v| {
        const g = random.floatNorm(f32);
        v.* = g;
        norm += @as(f64, g) * g;
    }
    const inv: f32 = @floatCast(1.0 / @sqrt(norm));
    for (buf) |*v| v.* *= inv;
}

test "search finds exact matches from the corpus" {
    // The sharpest end-to-end check available: query with a vector that is in the
    // corpus. Its own entry should come back first, since it maximizes the true
    // inner product by a wide margin.
    const allocator = testing.allocator;
    const dim: u32 = 256;
    const n = 500;

    var index = try FlatIndex.init(allocator, .{ .dim = dim, .bits = 5, .seed = 0x5EED });
    defer index.deinit();
    try testing.expect(index.vectorized());

    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(1);
    for (0..n) |i| randomUnit(corpus[i * dim ..][0..dim], prng.random());
    try index.addBatch(corpus);
    try testing.expectEqual(@as(usize, n), index.count());

    var searcher = try Searcher10.init(allocator, index);
    defer searcher.deinit();

    var found: usize = 0;
    for (0..50) |i| {
        const results = index.search(corpus[i * dim ..][0..dim], &searcher.inner);
        if (results[0].id == @as(u32, @intCast(i))) found += 1;
    }
    // Self-retrieval should be near-perfect even at 4 code bits.
    try testing.expect(found >= 48);
}

const Searcher10 = struct {
    inner: FlatIndex.Searcher,
    fn init(allocator: Allocator, index: FlatIndex) !Searcher10 {
        return .{ .inner = try FlatIndex.Searcher.init(allocator, index, 10) };
    }
    fn deinit(self: *Searcher10) void {
        self.inner.deinit();
    }
};

test "vectorized and exact scans agree on ranking" {
    // The int8 path is an approximation; it must not reorder results relative to the
    // exact path often enough to matter. Run both over the same corpus and compare.
    const allocator = testing.allocator;
    const dim: u32 = 512;
    const n = 800;

    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(7);
    for (0..n) |i| randomUnit(corpus[i * dim ..][0..dim], prng.random());

    var fast = try FlatIndex.init(allocator, .{ .dim = dim, .bits = 5, .seed = 3 });
    defer fast.deinit();
    var exact = try FlatIndex.init(allocator, .{ .dim = dim, .bits = 5, .seed = 3, .exact_scan = true });
    defer exact.deinit();
    try fast.addBatch(corpus);
    try exact.addBatch(corpus);
    try testing.expect(fast.vectorized());
    try testing.expect(!exact.vectorized());

    var fast_searcher = try FlatIndex.Searcher.init(allocator, fast, 10);
    defer fast_searcher.deinit();
    var exact_searcher = try FlatIndex.Searcher.init(allocator, exact, 10);
    defer exact_searcher.deinit();

    const query = try allocator.alloc(f32, dim);
    defer allocator.free(query);

    var overlap: usize = 0;
    var total: usize = 0;
    for (0..40) |_| {
        randomUnit(query, prng.random());
        const a = fast.search(query, &fast_searcher);
        var ids: [10]u32 = undefined;
        for (a, 0..) |e, j| ids[j] = e.id;
        const b = exact.search(query, &exact_searcher);
        for (b) |e| {
            total += 1;
            if (std.mem.indexOfScalar(u32, &ids, e.id) != null) overlap += 1;
        }
    }
    const agreement = @as(f64, @floatFromInt(overlap)) / @as(f64, @floatFromInt(total));
    try testing.expect(agreement > 0.95);
}

test "every bit-width up to 5 vectorizes" {
    // bits=4 produces 3-bit codes, which straddle bytes; it reaches the fast path
    // through the bit-plane layout rather than by wasting a bit on nibble padding.
    // bits=5 is the last that fits: its 4-bit codes fill the 16-entry shuffle table.
    const allocator = testing.allocator;
    for (2..6) |bits| {
        var index = try FlatIndex.init(allocator, .{ .dim = 1024, .bits = @intCast(bits) });
        defer index.deinit();
        try testing.expectEqual(@as(u6, @intCast(bits - 1)), index.quantizer.mse.bits);
        try testing.expect(index.vectorized());
        // And the code budget is still exactly `bits` per coordinate.
        try testing.expectEqual(@as(usize, 1024 * bits / 8), index.codeBytesPerVector());
    }
}

test "every metric ranks correctly" {
    const allocator = testing.allocator;
    const dim: u32 = 128;
    const n = 300;

    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(11);
    const random = prng.random();
    for (0..n) |i| {
        randomUnit(corpus[i * dim ..][0..dim], random);
        // Vary the norms so inner product, cosine, and L2 genuinely differ.
        const factor = 0.5 + random.float(f32) * 2.0;
        for (corpus[i * dim ..][0..dim]) |*v| v.* *= factor;
    }

    for ([_]Metric{ .inner_product, .cosine, .l2 }) |metric| {
        var index = try FlatIndex.init(allocator, .{
            .dim = dim,
            .bits = 5,
            .metric = metric,
            .seed = 5,
        });
        defer index.deinit();
        try index.addBatch(corpus);

        var searcher = try FlatIndex.Searcher.init(allocator, index, 5);
        defer searcher.deinit();

        const query = try allocator.alloc(f32, dim);
        defer allocator.free(query);

        var hits: usize = 0;
        const trials = 30;
        for (0..trials) |_| {
            randomUnit(query, random);

            // Exact best under this metric.
            var best: u32 = 0;
            var best_score: f64 = -std.math.inf(f64);
            for (0..n) |i| {
                const row = corpus[i * dim ..][0..dim];
                var d: f64 = 0;
                var rn: f64 = 0;
                var qn: f64 = 0;
                for (row, query) |x, qv| {
                    d += @as(f64, x) * qv;
                    rn += @as(f64, x) * x;
                    qn += @as(f64, qv) * qv;
                }
                const s: f64 = switch (metric) {
                    .inner_product => d,
                    .cosine => d / (@sqrt(rn) * @sqrt(qn)),
                    .l2 => -(qn + rn - 2.0 * d),
                };
                if (s > best_score) {
                    best_score = s;
                    best = @intCast(i);
                }
            }

            const results = index.search(query, &searcher);
            for (results) |e| {
                if (e.id == best) {
                    hits += 1;
                    break;
                }
            }
        }
        // The true best must land in the top 5 nearly always.
        try testing.expect(hits >= trials - 3);
    }
}

test "results are sorted and ids are in range" {
    const allocator = testing.allocator;
    const dim: u32 = 64;
    var index = try FlatIndex.init(allocator, .{ .dim = dim, .bits = 3, .seed = 2 });
    defer index.deinit();

    var prng = std.Random.DefaultPrng.init(13);
    const vector = try allocator.alloc(f32, dim);
    defer allocator.free(vector);
    for (0..100) |_| {
        randomUnit(vector, prng.random());
        _ = try index.add(vector);
    }

    var searcher = try FlatIndex.Searcher.init(allocator, index, 7);
    defer searcher.deinit();
    randomUnit(vector, prng.random());
    const results = index.search(vector, &searcher);

    try testing.expectEqual(@as(usize, 7), results.len);
    for (1..results.len) |i| try testing.expect(results[i - 1].score >= results[i].score);
    for (results) |e| try testing.expect(e.id < 100);
}

test "empty index and k larger than the corpus" {
    const allocator = testing.allocator;
    const dim: u32 = 64;
    var index = try FlatIndex.init(allocator, .{ .dim = dim, .bits = 3 });
    defer index.deinit();

    var searcher = try FlatIndex.Searcher.init(allocator, index, 10);
    defer searcher.deinit();

    const query = [_]f32{0.1} ** dim;
    try testing.expectEqual(@as(usize, 0), index.search(&query, &searcher).len);

    const vector = [_]f32{0.5} ** dim;
    _ = try index.add(&vector);
    _ = try index.add(&vector);
    try testing.expectEqual(@as(usize, 2), index.search(&query, &searcher).len);
}

test "searcher is reusable and allocation-free per query" {
    const allocator = testing.allocator;
    const dim: u32 = 128;
    var index = try FlatIndex.init(allocator, .{ .dim = dim, .bits = 5, .seed = 1 });
    defer index.deinit();

    var prng = std.Random.DefaultPrng.init(17);
    const vector = try allocator.alloc(f32, dim);
    defer allocator.free(vector);
    for (0..200) |_| {
        randomUnit(vector, prng.random());
        _ = try index.add(vector);
    }

    var searcher = try FlatIndex.Searcher.init(allocator, index, 5);
    defer searcher.deinit();

    // Repeated searches must not leak; testing.allocator would catch it.
    for (0..50) |_| {
        randomUnit(vector, prng.random());
        const results = index.search(vector, &searcher);
        try testing.expectEqual(@as(usize, 5), results.len);
    }
}

test "search works at every bit-width, vectorized or not" {
    // Constructing a Searcher above the shuffle-table limit used to overrun a
    // 16-entry array: the vectorization test stopped at b=5 and the wider-bit test
    // only checked storage sizing, so nothing ever built a Searcher at b=6. Debug
    // caught it as an assert; ReleaseFast corrupted memory instead.
    const allocator = testing.allocator;
    const dim: u32 = 128;
    const n = 200;

    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(0x5A5E);
    const random = prng.random();
    for (0..n) |i| randomUnit(corpus[i * dim ..][0..dim], random);

    for (2..8) |bits| {
        var index = try FlatIndex.init(allocator, .{
            .dim = dim,
            .bits = @intCast(bits),
            .seed = 9,
        });
        defer index.deinit();
        try index.addBatch(corpus);

        var searcher = try FlatIndex.Searcher.init(allocator, index, 10);
        defer searcher.deinit();

        // Self-retrieval must work whichever scan path is taken.
        var hits: usize = 0;
        for (0..20) |i| {
            const results = index.search(corpus[i * dim ..][0..dim], &searcher);
            try testing.expectEqual(@as(usize, 10), results.len);
            for (results) |e| {
                if (e.id == @as(u32, @intCast(i))) {
                    hits += 1;
                    break;
                }
            }
        }
        try testing.expect(hits >= 19);
    }
}

test "storage matches the advertised bit budget" {
    const allocator = testing.allocator;
    for ([_]u6{ 2, 3, 4, 5, 6, 7 }) |bits| {
        var index = try FlatIndex.init(allocator, .{ .dim = 1024, .bits = bits });
        defer index.deinit();
        // b bits per coordinate exactly: (b−1) code bits plus one sketch bit, in
        // both the sequential and bit-plane layouts.
        try testing.expectEqual(@as(usize, 1024 * @as(usize, bits) / 8), index.codeBytesPerVector());
        // Plus two half-precision scalars, which do not scale with dimension.
        try testing.expectEqual(index.codeBytesPerVector() + 4, index.bytesPerVector());
    }
}

test "half-precision scalars do not measurably hurt ranking" {
    // f16 norms and gammas carry ~3 decimal digits against inputs already quantized
    // to a handful of bits. Verified rather than assumed: compare retrieval against
    // the full-precision estimator on the same corpus.
    const allocator = testing.allocator;
    const dim: u32 = 256;
    const n = 600;

    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(0x5CA1);
    const random = prng.random();
    for (0..n) |i| {
        randomUnit(corpus[i * dim ..][0..dim], random);
        // Vary norms so the stored scale actually matters.
        const factor = 0.01 + random.float(f32) * 100.0;
        for (corpus[i * dim ..][0..dim]) |*v| v.* *= factor;
    }

    var index = try FlatIndex.init(allocator, .{ .dim = dim, .bits = 5, .seed = 4 });
    defer index.deinit();
    try index.addBatch(corpus);

    var searcher = try FlatIndex.Searcher.init(allocator, index, 10);
    defer searcher.deinit();

    const query = try allocator.alloc(f32, dim);
    defer allocator.free(query);

    var hits: usize = 0;
    const trials = 40;
    for (0..trials) |_| {
        randomUnit(query, random);
        var best: u32 = 0;
        var best_score: f64 = -std.math.inf(f64);
        for (0..n) |i| {
            var d: f64 = 0;
            for (corpus[i * dim ..][0..dim], query) |x, qv| d += @as(f64, x) * qv;
            if (d > best_score) {
                best_score = d;
                best = @intCast(i);
            }
        }
        for (index.search(query, &searcher)) |e| {
            if (e.id == best) {
                hits += 1;
                break;
            }
        }
    }
    try testing.expect(hits >= trials - 4);
}
