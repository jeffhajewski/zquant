//! Partitioned index: k-means lists, residuals against each list's centroid, and a search
//! that scans only the lists nearest the query.
//!
//! ## The score
//!
//! A calibrated `FlatIndex`'s, with the vector's list centroid in place of the one corpus
//! centre (`flat.encodeResidual`, `flat.scalarScore`):
//!
//!     ⟨q, x⟩ ≈ ‖x‖ · [ f·⟨p, ũ⟩  +  ‖q‖·b  +  ⟨p, c_list⟩ ]
//!
//! A centroid near the vector leaves a shorter residual than the corpus centre does, and
//! the same codes describe a shorter residual more precisely. So partitioning raises recall
//! even with every list scanned, before it saves any work. With one list this is a
//! calibrated flat index; a test checks the two agree.
//!
//! ## The search
//!
//! Rank the centroids against the query, then scan the `probes` nearest lists. Centroids
//! are ranked by Euclidean distance to the query's direction, which is how the lists were
//! formed: k-means over unit directions.
//!
//! A batch of queries is scanned list by list rather than query by query. Each list is
//! read once for all the queries in the batch that probe it, through the kernels a flat
//! scan uses: the tiled multi-query kernel when four or more queries share the list, and
//! the one-query tiles otherwise. With many lists and few probes most lists see one query
//! or none, so batching saves far less here than over a flat corpus. That is inherent to
//! partitioning, and is what the smaller scan pays for.
//!
//! ## Storage
//!
//! Per vector: the codes, the f16 norm, factor and bias of a calibrated flat index, and a
//! u32 id, since lists hold vectors out of insertion order. Per index: the centroids,
//! `lists × padded` floats.

const std = @import("std");
const Allocator = std.mem.Allocator;

const flat = @import("flat.zig");
const prod_mod = @import("../quant/prod.zig");
const packing = @import("../quant/packing.zig");
const scan = @import("../simd/scan.zig");
const tiles = @import("../simd/tiles.zig");
const dispatch = @import("../simd/dispatch.zig");
const topk = @import("topk.zig");
const rng = @import("../math/rng.zig");
const RotationKind = @import("../math/rotation.zig").Kind;

pub const Entry = topk.Entry;
pub const Metric = flat.Metric;
pub const Residency = flat.Residency;
pub const StoredScalars = flat.StoredScalars;

/// The id in a result slot that no vector filled, because the probed lists held fewer
/// than k vectors between them. Its score is −inf, so it sorts last.
pub const no_id: u32 = std.math.maxInt(u32);

/// Most lists an index may have.
pub const max_lists: u32 = 1 << 16;

/// Lloyd iterations. IVF needs lists that are compact, not a converged optimum, and
/// iterations past about ten move recall by a fraction of a point.
const kmeans_iterations = 10;

/// Training rows per list, at most. Below about 40 a centroid is mostly noise; above a
/// few hundred, training costs time and buys nothing.
pub const max_train_per_list = 256;

pub const Params = struct {
    dim: u32,
    /// Total bits per coordinate, as for `FlatIndex`: the codebook uses `bits − 1`.
    bits: u6,
    /// Number of k-means lists.
    lists: u32,
    metric: Metric = .inner_product,
    seed: u64 = 0,
    rotation: RotationKind = .hadamard,
    residency: Residency = .compact,
};

pub const IvfIndex = struct {
    params: Params,
    quantizer: prod_mod.Prod,
    layout: packing.Layout,
    /// int8 centroid table, for `.expanded` codes.
    table: scan.Table,
    /// List centroids in the rotated basis, `lists × padded`. Meaningful once trained.
    centroids: []f32,
    /// ‖c‖² per list, for ranking centroids by distance.
    centroid_sq: []f32,
    lists: []List,
    trained: bool,
    total: usize,
    allocator: Allocator,

    pub const List = struct {
        codes: std.ArrayList(u8) = .empty,
        scalars: std.ArrayList(StoredScalars) = .empty,
        biases: std.ArrayList(f16) = .empty,
        ids: std.ArrayList(u32) = .empty,

        pub fn len(self: List) usize {
            return self.ids.items.len;
        }

        fn deinit(self: *List, allocator: Allocator) void {
            self.codes.deinit(allocator);
            self.scalars.deinit(allocator);
            self.biases.deinit(allocator);
            self.ids.deinit(allocator);
        }
    };

    pub fn init(allocator: Allocator, params: Params) !IvfIndex {
        if (params.residency == .expanded and params.bits > 5) return error.BitWidthTooWideForExpanded;
        // Expanded codes are int8 values, readable only by the int8 kernels, which need
        // whole 16-dimension chunks. Only a dimension of 8 or less pads to fewer.
        if (params.residency == .expanded and params.dim <= 8) return error.DimensionTooSmallForExpanded;
        if (params.lists == 0 or params.lists > max_lists) return error.BadListCount;

        var quantizer = try prod_mod.Prod.init(allocator, .{
            .dim = params.dim,
            .bits = params.bits,
            .seed = params.seed,
            .rotation = params.rotation,
        });
        errdefer quantizer.deinit();
        const width: usize = quantizer.padded();

        const centroids = try allocator.alloc(f32, params.lists * width);
        errdefer allocator.free(centroids);
        @memset(centroids, 0);
        const centroid_sq = try allocator.alloc(f32, params.lists);
        errdefer allocator.free(centroid_sq);
        @memset(centroid_sq, 0);
        const lists = try allocator.alloc(List, params.lists);
        for (lists) |*l| l.* = .{};

        return .{
            .params = params,
            .quantizer = quantizer,
            .layout = packing.Layout.init(quantizer.padded(), quantizer.mse.bits),
            .table = if (quantizer.mse.codebook.levels() <= 16)
                scan.Table.init(quantizer.mse.codebook.centroids)
            else
                scan.unusedTable(),
            .centroids = centroids,
            .centroid_sq = centroid_sq,
            .lists = lists,
            .trained = false,
            .total = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *IvfIndex) void {
        for (self.lists) |*l| l.deinit(self.allocator);
        self.allocator.free(self.lists);
        self.allocator.free(self.centroid_sq);
        self.allocator.free(self.centroids);
        self.quantizer.deinit();
        self.* = undefined;
    }

    pub fn count(self: IvfIndex) usize {
        return self.total;
    }

    pub fn dim(self: IvfIndex) u32 {
        return self.quantizer.dim();
    }

    pub fn listCount(self: IvfIndex) usize {
        return self.lists.len;
    }

    fn padded(self: IvfIndex) usize {
        return self.quantizer.padded();
    }

    fn centroid(self: IvfIndex, l: usize) []const f32 {
        return self.centroids[l * self.padded() ..][0..self.padded()];
    }

    /// Bytes of codes per vector; the same rule as `FlatIndex.codeStride`.
    pub fn codeStride(self: IvfIndex) usize {
        return switch (self.params.residency) {
            .compact => self.layout.stride(),
            .expanded => std.mem.alignForward(usize, self.padded(), 16),
        };
    }

    /// Bytes stored per vector: codes, three f16 scalars and a u32 id. The centroids are
    /// per index; `centroidBytes` reports them.
    pub fn bytesPerVector(self: IvfIndex) usize {
        return self.codeStride() + @sizeOf(StoredScalars) + @sizeOf(f16) + @sizeOf(u32);
    }

    pub fn centroidBytes(self: IvfIndex) usize {
        return self.centroids.len * @sizeOf(f32);
    }

    /// Whether the int8 kernels apply; the same rule as `FlatIndex.vectorized`.
    pub fn vectorized(self: IvfIndex) bool {
        if (self.params.residency == .expanded) return self.padded() % 16 == 0;
        return scan.canVectorize(self.layout);
    }

    // -- training -----------------------------------------------------------------------

    /// Fit the lists by k-means over the sample's rotated directions. Must come before any
    /// add, and may be repeated while the index is empty. Rows of zero have no direction
    /// and are skipped; at least `lists` others are needed. At most
    /// `max_train_per_list × lists` rows are used, chosen by the seed.
    ///
    /// The result depends on the sample and the seed, not on `threads`.
    pub fn train(self: *IvfIndex, sample: []const f32, threads: usize) !void {
        if (self.total > 0) return error.IndexNotEmpty;
        const d = self.dim();
        std.debug.assert(sample.len % d == 0);
        if (!flat.allFinite(sample)) return error.NonFiniteInput;
        const rows_in = sample.len / d;
        const k = self.lists.len;
        const p = self.padded();
        const a = self.allocator;
        if (rows_in < k) return error.EmptySample;
        if (rows_in > std.math.maxInt(u32)) return error.SampleTooLarge;

        // A seeded shuffle, so a capped sample is a random subset rather than the first
        // rows, which are often sorted by something.
        const order = try a.alloc(u32, rows_in);
        defer a.free(order);
        var stream = rng.Philox.init(self.params.seed).stream(.ivf_training);
        stream.fillPermutation(order);

        const cap = @min(rows_in, max_train_per_list * k);
        const data = try a.alloc(f32, cap * p);
        defer a.free(data);
        var ws = try prod_mod.Workspace.init(a, self.quantizer);
        defer ws.deinit();
        var rows: usize = 0;
        for (order) |r| {
            if (rows == cap) break;
            const row = sample[@as(usize, r) * d ..][0..d];
            if (self.quantizer.mse.encodeRotated(row, data[rows * p ..][0..p], ws.staging) == 0) continue;
            rows += 1;
        }
        if (rows < k) return error.EmptySample;

        try kmeans(a, data[0 .. rows * p], p, k, threads, self.centroids);
        for (0..k) |l| self.centroid_sq[l] = dotExact(self.centroid(l), self.centroid(l));
        self.trained = true;
    }

    // -- adding -------------------------------------------------------------------------

    /// Encode and append a row-major batch. Ids continue from `count()` in row order.
    pub fn addBatch(self: *IvfIndex, vectors: []const f32) !void {
        return self.addBatchParallel(vectors, 1);
    }

    /// Below this many vectors per thread, spawning costs more than it saves.
    const min_rows_per_thread = 256;
    /// Vectors encoded per pass, bounding the staging memory a large add needs.
    const add_chunk = 1 << 16;

    /// `addBatch` on up to `threads` threads. The index comes out byte-identical whatever
    /// the thread count, and the batch goes in whole or not at all.
    pub fn addBatchParallel(self: *IvfIndex, vectors: []const f32, threads: usize) !void {
        if (!self.trained) return error.NotTrained;
        const d = self.dim();
        std.debug.assert(vectors.len % d == 0);
        const n = vectors.len / d;
        if (n == 0) return;
        if (!flat.allFinite(vectors)) return error.NonFiniteInput;
        if (self.total + n >= no_id) return error.TooManyVectors;

        const stride = self.codeStride();
        const a = self.allocator;
        const chunk = @min(n, add_chunk);
        // Every vector's list, kept from the first pass to the second: 4 bytes a vector,
        // against the cost of ranking every centroid for it again.
        const assigned = try a.alloc(u32, n);
        defer a.free(assigned);
        const staged: Staging = .{
            .lists = &.{},
            .scalars = try a.alloc(StoredScalars, chunk),
            .biases = try a.alloc(f16, chunk),
            .codes = try a.alloc(u8, chunk * stride),
        };
        defer staged.free(a);
        const workers = @max(1, @min(threads, chunk / min_rows_per_thread));
        const encoders = try a.alloc(Encoder, workers);
        defer a.free(encoders);
        var made: usize = 0;
        defer for (encoders[0..made]) |*e| e.deinit(a);
        while (made < workers) : (made += 1) encoders[made] = try Encoder.init(a, self);

        // Assign every vector first and reserve for all of them, so that once appending
        // starts nothing can fail. Encoding then runs a chunk at a time, since staging a
        // large batch's codes whole would not fit.
        const per_list = try a.alloc(usize, self.lists.len);
        defer a.free(per_list);
        @memset(per_list, 0);
        var start: usize = 0;
        while (start < n) : (start += chunk) {
            const take = @min(chunk, n - start);
            var out = staged.slice(0, take, stride);
            out.lists = assigned[start..][0..take];
            self.runEncode(vectors[start * d ..][0 .. take * d], out, encoders, .assign);
        }
        for (assigned) |l| per_list[l] += 1;
        for (self.lists, per_list) |*l, extra| {
            if (extra == 0) continue;
            try l.codes.ensureUnusedCapacity(a, extra * stride);
            try l.scalars.ensureUnusedCapacity(a, extra);
            try l.biases.ensureUnusedCapacity(a, extra);
            try l.ids.ensureUnusedCapacity(a, extra);
        }

        start = 0;
        while (start < n) : (start += chunk) {
            const take = @min(chunk, n - start);
            var out = staged.slice(0, take, stride);
            out.lists = assigned[start..][0..take];
            self.runEncode(vectors[start * d ..][0 .. take * d], out, encoders, .encode);
            for (0..take) |v| {
                const l = &self.lists[assigned[start + v]];
                l.codes.appendSliceAssumeCapacity(staged.codes[v * stride ..][0..stride]);
                l.scalars.appendAssumeCapacity(staged.scalars[v]);
                l.biases.appendAssumeCapacity(staged.biases[v]);
                l.ids.appendAssumeCapacity(@intCast(self.total + start + v));
            }
        }
        self.total += n;
    }

    const Staging = struct {
        lists: []u32,
        scalars: []StoredScalars,
        biases: []f16,
        codes: []u8,

        fn slice(s: Staging, from: usize, take: usize, stride: usize) Staging {
            return .{
                .lists = if (s.lists.len == 0) s.lists else s.lists[from..][0..take],
                .scalars = s.scalars[from..][0..take],
                .biases = s.biases[from..][0..take],
                .codes = s.codes[from * stride ..][0 .. take * stride],
            };
        }

        fn free(s: Staging, a: Allocator) void {
            a.free(s.scalars);
            a.free(s.biases);
            a.free(s.codes);
        }
    };

    /// `.assign` chooses each row's list into `out.lists`; `.encode` encodes each row
    /// against the list already chosen there.
    const Mode = enum { assign, encode };

    /// Assign or encode every row of `vectors` into `out`, spread over `encoders.len`
    /// threads. Each row lands in its own slot.
    fn runEncode(self: *const IvfIndex, vectors: []const f32, out: Staging, encoders: []Encoder, mode: Mode) void {
        const d = self.dim();
        const n = vectors.len / d;
        const stride = self.codeStride();
        const workers = @max(1, @min(encoders.len, n / min_rows_per_thread));

        const Work = struct {
            index: *const IvfIndex,
            vectors: []const f32,
            out: Staging,
            enc: *Encoder,
            mode: Mode,

            fn run(w: @This()) void {
                w.index.encodeRows(w.vectors, w.out, w.enc, w.mode);
            }
        };
        var handles: [64]?std.Thread = @splat(null);
        const spawnable = @min(workers, handles.len);
        const base = n / spawnable;
        const rem = n % spawnable;
        var start: usize = 0;
        var own: Work = undefined;
        for (0..spawnable) |w| {
            const take = base + @intFromBool(w < rem);
            const work: Work = .{
                .index = self,
                .vectors = vectors[start * d ..][0 .. take * d],
                .out = out.slice(start, take, stride),
                .enc = &encoders[w],
                .mode = mode,
            };
            start += take;
            if (w == 0) {
                own = work;
                continue;
            }
            handles[w] = std.Thread.spawn(.{}, Work.run, .{work}) catch blk: {
                work.run();
                break :blk null;
            };
        }
        own.run();
        for (handles[0..spawnable]) |h| if (h) |t| t.join();
    }

    /// Per-thread encoding scratch.
    const Encoder = struct {
        ws: prod_mod.Workspace,
        codes: []u8,
        /// Up to `block` rotated directions, and their dot products with every centroid.
        rotated: []f32,
        dots: []f32,

        const block = 4;

        fn init(a: Allocator, index: *const IvfIndex) !Encoder {
            var ws = try prod_mod.Workspace.init(a, index.quantizer);
            errdefer ws.deinit();
            const codes = try a.alloc(u8, index.padded());
            errdefer a.free(codes);
            const rotated = try a.alloc(f32, block * index.padded());
            errdefer a.free(rotated);
            const dots = try a.alloc(f32, block * index.lists.len);
            return .{ .ws = ws, .codes = codes, .rotated = rotated, .dots = dots };
        }

        fn deinit(self: *Encoder, a: Allocator) void {
            a.free(self.dots);
            a.free(self.rotated);
            a.free(self.codes);
            self.ws.deinit();
        }
    };

    fn encodeRows(self: *const IvfIndex, vectors: []const f32, out: Staging, enc: *Encoder, mode: Mode) void {
        const d = self.dim();
        const p = self.padded();
        const k = self.lists.len;
        const stride = self.codeStride();
        const n = vectors.len / d;
        var norms: [Encoder.block]f32 = undefined;
        var r0: usize = 0;
        while (r0 < n) : (r0 += Encoder.block) {
            const take = @min(Encoder.block, n - r0);
            for (0..take) |u| {
                const row = vectors[(r0 + u) * d ..][0..d];
                norms[u] = self.quantizer.mse.encodeRotated(row, enc.rotated[u * p ..][0..p], enc.ws.staging);
            }
            if (mode == .assign) {
                centroidDots(enc.rotated[0 .. take * p], p, self.centroids, k, enc.dots);
                for (0..take) |u| out.lists[r0 + u] = @intCast(nearest(enc.dots[u * k ..][0..k], self.centroid_sq));
                continue;
            }
            for (0..take) |u| {
                const v = r0 + u;
                const l = out.lists[v];

                @memcpy(enc.ws.rotated, enc.rotated[u * p ..][0..p]);
                const res = flat.encodeResidual(&self.quantizer, self.centroid(l), enc.codes, &enc.ws);
                out.scalars[v] = .{ .norm = @floatCast(norms[u]), .gamma = @floatCast(res.factor) };
                out.biases[v] = @floatCast(res.bias);
                const dst = out.codes[v * stride ..][0..stride];
                switch (self.params.residency) {
                    .compact => self.layout.pack(enc.codes, dst),
                    .expanded => {
                        const values: [16]i8 = self.table.values;
                        for (dst[0..enc.codes.len], enc.codes) |*o, code| o.* = @bitCast(values[code]);
                        @memset(dst[enc.codes.len..], 0);
                    },
                }
            }
        }
    }

    // -- searching ----------------------------------------------------------------------

    /// A query probing a list, and its ⟨p, c_list⟩.
    const Member = struct { query: u16, centre_dot: f32 };
    /// A probe before it is sorted by list.
    const Pick = struct { list: u32, member: Member };

    /// Per-batch search state, sized once and reused: a search allocates nothing.
    pub const Searcher = struct {
        batch: usize,
        k: usize,
        /// Lists each query scans. `setProbes` changes it.
        probes: usize,
        query_states: []prod_mod.QueryState,
        scan_queries: []scan.Query,
        heap_store: []Entry,
        table: scan.Table,
        workspace: prod_mod.Workspace,
        /// ⟨p, c_l⟩ for up to four queries and every list, and those queries' rotated
        /// forms side by side, so the centroid ranking runs four queries per pass.
        coarse: []f32,
        stage_rows: []f32,
        /// Queries probing each list: list l's are `members[starts[l]..starts[l + 1]]`.
        starts: []u32,
        members: []Member,
        picks: []Pick,
        /// Probe selection scratch, `probes` long.
        heap_keys: []f32,
        heap_lists: []u32,
        allocator: Allocator,

        pub fn init(allocator: Allocator, index: IvfIndex, batch: usize, k: usize, probes: usize) !Searcher {
            std.debug.assert(batch > 0 and batch <= flat.max_batch and k > 0);
            const lists = index.lists.len;
            const nprobe = std.math.clamp(probes, 1, lists);

            const query_states = try allocator.alloc(prod_mod.QueryState, batch);
            errdefer allocator.free(query_states);
            const scan_queries = try allocator.alloc(scan.Query, batch);
            errdefer allocator.free(scan_queries);
            var built: usize = 0;
            errdefer for (0..built) |i| {
                query_states[i].deinit();
                scan_queries[i].deinit();
            };
            while (built < batch) : (built += 1) {
                query_states[built] = try prod_mod.QueryState.init(allocator, index.quantizer);
                errdefer query_states[built].deinit();
                scan_queries[built] = if (index.params.residency == .expanded)
                    try scan.Query.initSequential(allocator, @intCast(index.padded()))
                else
                    try scan.Query.init(allocator, index.layout);
            }
            const heap_store = try allocator.alloc(Entry, batch * k);
            errdefer allocator.free(heap_store);
            var workspace = try prod_mod.Workspace.init(allocator, index.quantizer);
            errdefer workspace.deinit();
            const coarse = try allocator.alloc(f32, 4 * lists);
            errdefer allocator.free(coarse);
            const stage_rows = try allocator.alloc(f32, 4 * index.padded());
            errdefer allocator.free(stage_rows);
            const starts = try allocator.alloc(u32, lists + 1);

            var self: Searcher = .{
                .batch = batch,
                .k = k,
                .probes = 0,
                .query_states = query_states,
                .scan_queries = scan_queries,
                .heap_store = heap_store,
                .table = if (index.vectorized())
                    scan.Table.init(index.quantizer.mse.codebook.centroids)
                else
                    scan.unusedTable(),
                .workspace = workspace,
                .coarse = coarse,
                .stage_rows = stage_rows,
                .starts = starts,
                .members = &.{},
                .picks = &.{},
                .heap_keys = &.{},
                .heap_lists = &.{},
                .allocator = allocator,
            };
            errdefer allocator.free(starts);
            try self.setProbes(nprobe);
            return self;
        }

        /// Scan `probes` lists per query from now on, clamped to 1..lists. Allocates only
        /// when it grows past every earlier setting. Needs nothing from the index, so a
        /// binding may call it on a searcher whose index is gone.
        pub fn setProbes(self: *Searcher, probes: usize) !void {
            const nprobe = std.math.clamp(probes, 1, self.starts.len - 1);
            if (nprobe > self.heap_keys.len) {
                const members = try self.allocator.alloc(Member, self.batch * nprobe);
                errdefer self.allocator.free(members);
                const picks = try self.allocator.alloc(Pick, self.batch * nprobe);
                errdefer self.allocator.free(picks);
                const keys = try self.allocator.alloc(f32, nprobe);
                errdefer self.allocator.free(keys);
                const lists = try self.allocator.alloc(u32, nprobe);
                self.allocator.free(self.members);
                self.allocator.free(self.picks);
                self.allocator.free(self.heap_keys);
                self.allocator.free(self.heap_lists);
                self.members = members;
                self.picks = picks;
                self.heap_keys = keys;
                self.heap_lists = lists;
            }
            self.probes = nprobe;
        }

        pub fn deinit(self: *Searcher) void {
            for (self.query_states, self.scan_queries) |*a, *b| {
                a.deinit();
                b.deinit();
            }
            self.allocator.free(self.heap_lists);
            self.allocator.free(self.heap_keys);
            self.allocator.free(self.picks);
            self.allocator.free(self.members);
            self.allocator.free(self.starts);
            self.allocator.free(self.stage_rows);
            self.allocator.free(self.coarse);
            self.workspace.deinit();
            self.allocator.free(self.heap_store);
            self.allocator.free(self.scan_queries);
            self.allocator.free(self.query_states);
            self.* = undefined;
        }
    };

    /// Each query's results while a batch is scanned. They outlive any one list.
    const Collect = struct {
        collectors: [flat.max_batch]topk.TopK,
        hot: [flat.max_batch][flat.hot_slots]Entry,
        hot_len: [flat.max_batch]usize,
        /// Cached admission thresholds, as in `FlatIndex.searchBatch`.
        thresholds: [flat.max_batch]f32,
    };

    /// The queries probing one list, gathered so its scan reads them contiguously.
    const Group = struct {
        m: usize,
        queries: [flat.max_batch]scan.Query,
        owner: [flat.max_batch]u16,
        factor: [flat.max_batch]f32,
        offset: [flat.max_batch]f32,
        bias: [flat.max_batch]f32,
        centre: [flat.max_batch]f32,
        thresholds: [flat.max_batch]f32,

        fn terms(g: *const Group, j: usize) flat.QueryTerms {
            return .{ .factor = g.factor[j], .offset = g.offset[j], .bias = g.bias[j], .centre = g.centre[j] };
        }
    };

    /// Search a batch of queries, scanning the `searcher.probes` lists nearest each.
    ///
    /// Query i's results are `result[i*k ..][0..k]`, best first. A slot the probed lists
    /// could not fill holds `no_id` and −inf. Valid until the searcher's next search.
    pub fn searchBatch(self: *const IvfIndex, queries: []const f32, searcher: *Searcher) []Entry {
        const d = self.dim();
        std.debug.assert(queries.len % d == 0);
        const n = queries.len / d;
        std.debug.assert(n > 0 and n <= searcher.batch);
        const lists = self.lists.len;
        const nprobe = searcher.probes;
        const p = self.padded();
        const use_simd = self.vectorized();
        const metric = self.params.metric;

        var query_norms: [flat.max_batch]f32 = undefined;
        for (0..n) |i| {
            const q = queries[i * d ..][0..d];
            const state = &searcher.query_states[i];
            self.quantizer.prepareQueryRotated(q, state, &searcher.workspace);
            if (use_simd) searcher.scan_queries[i].load(state.rotated);
            query_norms[i] = flat.euclideanNorm(q);
        }

        // Choose each query's lists, four queries per pass over the centroids, then sort
        // the choices by list (a counting sort, so a list's queries stay in batch order).
        const starts = searcher.starts;
        @memset(starts, 0);
        const picks = searcher.picks[0 .. n * nprobe];
        var first: usize = 0;
        while (first < n) : (first += 4) {
            const take = @min(4, n - first);
            for (0..take) |u| @memcpy(searcher.stage_rows[u * p ..][0..p], searcher.query_states[first + u].rotated);
            centroidDots(searcher.stage_rows[0 .. take * p], p, self.centroids, lists, searcher.coarse[0 .. take * lists]);
            for (0..take) |u| {
                const i = first + u;
                const dots = searcher.coarse[u * lists ..][0..lists];
                const chosen = searcher.heap_lists[0..nprobe];
                nearestLists(dots, query_norms[i], self.centroid_sq, searcher.heap_keys[0..nprobe], chosen);
                for (chosen, picks[i * nprobe ..][0..nprobe]) |l, *pick| {
                    pick.* = .{ .list = l, .member = .{ .query = @intCast(i), .centre_dot = dots[l] } };
                    starts[l + 1] += 1;
                }
            }
        }
        for (1..lists + 1) |l| starts[l] += starts[l - 1];
        // Place each pick, advancing its list's start; afterwards starts[l] holds where
        // list l + 1 begins, so shift everything back by one.
        for (picks) |pick| {
            searcher.members[starts[pick.list]] = pick.member;
            starts[pick.list] += 1;
        }
        var back: usize = lists;
        while (back > 0) : (back -= 1) starts[back] = starts[back - 1];
        starts[0] = 0;

        var col: Collect = undefined;
        for (0..n) |i| {
            col.collectors[i] = topk.TopK.init(searcher.heap_store[i * searcher.k ..][0..searcher.k]);
            col.hot_len[i] = 0;
            col.thresholds[i] = -std.math.inf(f32);
        }

        var group: Group = undefined;
        const kernels = dispatch.active();
        for (self.lists, 0..) |*list, l| {
            const members = searcher.members[starts[l]..starts[l + 1]];
            if (members.len == 0 or list.len() == 0) continue;
            group.m = members.len;
            for (members, 0..) |member, j| {
                const q = member.query;
                const t = flat.queryTerms(metric, query_norms[q], member.centre_dot);
                group.queries[j] = searcher.scan_queries[q];
                group.owner[j] = q;
                group.factor[j] = t.factor;
                group.offset[j] = t.offset;
                group.bias[j] = t.bias;
                group.centre[j] = t.centre;
                group.thresholds[j] = col.thresholds[q];
            }
            if (!use_simd) {
                self.scanListExact(list, &group, &col, searcher);
            } else if (group.m >= tiles.batch_group) {
                self.scanListBatch(list, &group, &col, searcher, kernels);
            } else for (0..group.m) |j| {
                self.scanListOne(list, &group, j, &col, searcher, kernels);
            }
            for (group.owner[0..group.m], group.thresholds[0..group.m]) |q, th| col.thresholds[q] = th;
        }

        for (0..n) |i| {
            flat.flushHot(&col.collectors[i], &col.hot[i], &col.hot_len[i], &col.thresholds[i]);
            const got = col.collectors[i].drain().len;
            for (searcher.heap_store[i * searcher.k + got ..][0 .. searcher.k - got]) |*e| {
                e.* = .{ .id = no_id, .score = -std.math.inf(f32) };
            }
        }
        return searcher.heap_store[0 .. n * searcher.k];
    }

    /// One query: a batch of one.
    pub fn search(self: *const IvfIndex, query: []const f32, searcher: *Searcher) []Entry {
        return self.searchBatch(query, searcher);
    }

    fn job(self: *const IvfIndex, list: *const List, searcher: *const Searcher) tiles.Job {
        return .{
            .table = &searcher.table,
            .queries = undefined,
            .n = undefined,
            .codes = list.codes.items.ptr,
            .stride = self.codeStride(),
            .padded = @intCast(self.padded()),
            .bits = self.layout.bits,
            .expanded = self.params.residency == .expanded,
            .table_scale = self.table.scale,
            .out = undefined,
            .out_stride = undefined,
        };
    }

    fn canTile(self: *const IvfIndex) bool {
        return self.params.residency == .expanded or scan.canTile(self.layout);
    }

    /// ⟨p, ũ⟩ for vector `v` of `list` against one query, by the one-pair kernels.
    fn mseTerm(self: *const IvfIndex, list: *const List, v: usize, query: scan.Query, searcher: *const Searcher) f32 {
        const stride = self.codeStride();
        const slot = list.codes.items[v * stride ..][0..stride];
        return switch (self.params.residency) {
            .expanded => scan.scoreExpanded(@ptrCast(slot), query, self.table.scale, @intCast(self.padded())),
            .compact => scan.scoreInt8(self.layout, searcher.table, query, slot, @intCast(self.padded())),
        };
    }

    inline fn vectorTermsAt(self: *const IvfIndex, list: *const List, v: usize) flat.VectorTerms {
        const s = list.scalars.items[v];
        return flat.vectorTerms(self.params.metric, s.norm, s.gamma, list.biases.items[v]);
    }

    /// Four or more queries share the list: the tiled multi-query kernel and the lane
    /// epilogue of `FlatIndex.searchBatch`.
    fn scanListBatch(
        self: *const IvfIndex,
        list: *const List,
        group: *Group,
        col: *Collect,
        searcher: *const Searcher,
        kernels: ?*const tiles.Kernels,
    ) void {
        const tile = tiles.batch_tile;
        const lanes = flat.lanes;
        const Lane = flat.Lane;
        const m = group.m;
        const p: u32 = @intCast(self.padded());
        const stride = self.codeStride();
        var mse_block: [tile * flat.max_batch]f32 = undefined;
        var jb = self.job(list, searcher);
        jb.queries = &group.queries;
        jb.n = m;
        jb.out = &mse_block;
        jb.out_stride = flat.max_batch;
        const can_tile = self.canTile();
        const compact_width = scan.multiWidth(self.layout);
        var filled = false;

        const total = list.len();
        var v0: usize = 0;
        while (v0 < total) : (v0 += tile) {
            const take = @min(tile, total - v0);
            if (can_tile and take == tile) {
                dispatch.batchTile(kernels, &jb, v0);
            } else if (self.params.residency == .expanded) {
                for (0..take) |u| for (0..m) |j| {
                    mse_block[u * flat.max_batch + j] = self.mseTerm(list, v0 + u, group.queries[j], searcher);
                };
            } else for (0..take) |u| {
                // The unpack is query-independent, so the group shares it.
                const slot = list.codes.items[(v0 + u) * stride ..][0..stride];
                var j: usize = 0;
                while (j + compact_width <= m) : (j += compact_width) {
                    scan.scoreInt8Multi(self.layout, searcher.table, group.queries[j..m], slot, p, mse_block[u * flat.max_batch + j ..]);
                }
                while (j < m) : (j += 1) {
                    mse_block[u * flat.max_batch + j] = scan.scoreInt8(self.layout, searcher.table, group.queries[j], slot, p);
                }
            }

            // Collectors carry over from earlier lists, so test whether every one is full
            // rather than counting vectors, until they all are.
            if (!filled) {
                filled = true;
                for (group.owner[0..m]) |q| {
                    if (col.collectors[q].len < col.collectors[q].capacity()) filled = false;
                }
            }

            for (0..take) |u| {
                const v = v0 + u;
                const id = list.ids.items[v];
                const vt = self.vectorTermsAt(list, v);
                const block = mse_block[u * flat.max_batch ..][0..m];
                if (!filled) {
                    for (0..m) |j| {
                        const q = group.owner[j];
                        col.collectors[q].offer(flat.scalarScore(true, vt, group.terms(j), block[j]), id);
                        group.thresholds[j] = col.collectors[q].threshold();
                    }
                    continue;
                }
                const av: Lane = @splat(vt.factor);
                const bv: Lane = @splat(vt.offset);
                const biv: Lane = @splat(vt.bias);
                const cv: Lane = @splat(vt.centre);
                var j: usize = 0;
                switch (self.params.metric == .inner_product) {
                    inline else => |plain| while (j + lanes <= m) : (j += lanes) {
                        const mm: Lane = block[j..][0..lanes].*;
                        const th: Lane = group.thresholds[j..][0..lanes].*;
                        // `flat.scalarScore`, lane by lane in the same order, so batched
                        // and single-query scores are bit-identical.
                        const scaled = if (plain) av * mm else blk: {
                            const qf: Lane = group.factor[j..][0..lanes].*;
                            break :blk (av * qf) * mm;
                        };
                        const qb: Lane = group.bias[j..][0..lanes].*;
                        const qc: Lane = group.centre[j..][0..lanes].*;
                        const shared = scaled + (biv * qb + cv * qc);
                        const sc = if (plain) shared else blk: {
                            const qo: Lane = group.offset[j..][0..lanes].*;
                            break :blk shared + (bv + qo);
                        };
                        var mask: flat.LaneMask = @bitCast(sc > th);
                        if (mask == 0) continue;
                        const each: [lanes]f32 = sc;
                        while (mask != 0) {
                            const lane = @ctz(mask);
                            mask &= mask - 1;
                            stage(col, group, j + lane, id, each[lane]);
                        }
                    },
                }
                while (j < m) : (j += 1) {
                    const sc = flat.scalarScore(true, vt, group.terms(j), block[j]);
                    if (sc > group.thresholds[j]) stage(col, group, j, id, sc);
                }
            }
        }
    }

    /// Stage an accepted candidate for the query in group slot `j`, flushing that
    /// query's staging buffer into its collector when it fills.
    inline fn stage(col: *Collect, group: *Group, j: usize, id: u32, score: f32) void {
        const q = group.owner[j];
        col.hot[q][col.hot_len[q]] = .{ .id = id, .score = score };
        col.hot_len[q] += 1;
        if (col.hot_len[q] == flat.hot_slots) {
            flat.flushHot(&col.collectors[q], &col.hot[q], &col.hot_len[q], &group.thresholds[j]);
        }
    }

    /// Fewer than four queries share the list: the one-query tiles, for each.
    fn scanListOne(
        self: *const IvfIndex,
        list: *const List,
        group: *Group,
        j: usize,
        col: *Collect,
        searcher: *const Searcher,
        kernels: ?*const tiles.Kernels,
    ) void {
        const tile = tiles.single_tile;
        var block: [tile * 8]f32 = undefined;
        var jb = self.job(list, searcher);
        jb.queries = &group.queries[j];
        jb.n = 1;
        jb.out = &block;
        jb.out_stride = 1;
        const can_tile = self.canTile();
        const q = group.owner[j];
        const terms = group.terms(j);
        // Candidates staged for this query by an earlier list go in first, so the
        // collector and its threshold are current.
        flat.flushHot(&col.collectors[q], &col.hot[q], &col.hot_len[q], &group.thresholds[j]);
        const collector = &col.collectors[q];
        var threshold = group.thresholds[j];

        const total = list.len();
        var v0: usize = 0;
        while (v0 < total) : (v0 += block.len) {
            const take = @min(block.len, total - v0);
            const tiled = if (can_tile) take / tile * tile else 0;
            if (tiled > 0) dispatch.singleTiles(kernels, &jb, v0, tiled / tile);
            for (tiled..take) |u| block[u] = self.mseTerm(list, v0 + u, group.queries[j], searcher);
            for (0..take) |u| {
                const v = v0 + u;
                const sc = flat.scalarScore(true, self.vectorTermsAt(list, v), terms, block[u]);
                if (sc > threshold or collector.len < collector.capacity()) {
                    collector.offer(sc, list.ids.items[v]);
                    threshold = collector.threshold();
                }
            }
        }
        group.thresholds[j] = threshold;
    }

    /// Codebooks too wide for the int8 kernels: the exact f32 scan, pair by pair.
    fn scanListExact(self: *const IvfIndex, list: *const List, group: *Group, col: *Collect, searcher: *const Searcher) void {
        const stride = self.codeStride();
        const levels = self.quantizer.mse.codebook.centroids;
        for (0..group.m) |j| {
            const q = group.owner[j];
            flat.flushHot(&col.collectors[q], &col.hot[q], &col.hot_len[q], &group.thresholds[j]);
        }
        for (0..list.len()) |v| {
            const slot = list.codes.items[v * stride ..][0..stride];
            const vt = self.vectorTermsAt(list, v);
            for (0..group.m) |j| {
                const q = group.owner[j];
                const t = scan.scoreExact(self.layout, levels, searcher.query_states[q].rotated, slot);
                const sc = flat.scalarScore(true, vt, group.terms(j), t);
                const collector = &col.collectors[q];
                if (sc > group.thresholds[j] or collector.len < collector.capacity()) {
                    collector.offer(sc, list.ids.items[v]);
                    group.thresholds[j] = collector.threshold();
                }
            }
        }
    }

    // -- parallel search ----------------------------------------------------------------

    /// Query-parallel batch search, as `FlatIndex.ParallelSearcher`: one `Searcher` per
    /// thread, each given its own share of the queries.
    pub const ParallelSearcher = struct {
        shards: []Searcher,
        threads: []std.Thread,
        results: []Entry,
        per_thread: usize,
        k: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator, index: IvfIndex, threads: usize, per_thread: usize, k: usize, probes: usize) !ParallelSearcher {
            std.debug.assert(threads > 0 and per_thread > 0 and k > 0);
            const shards = try allocator.alloc(Searcher, threads);
            errdefer allocator.free(shards);
            var built: usize = 0;
            errdefer for (0..built) |i| shards[i].deinit();
            while (built < threads) : (built += 1) {
                shards[built] = try Searcher.init(allocator, index, per_thread, k, probes);
            }
            const handles = try allocator.alloc(std.Thread, threads);
            errdefer allocator.free(handles);
            const results = try allocator.alloc(Entry, threads * per_thread * k);
            return .{
                .shards = shards,
                .threads = handles,
                .results = results,
                .per_thread = per_thread,
                .k = k,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *ParallelSearcher) void {
            for (self.shards) |*s| s.deinit();
            self.allocator.free(self.results);
            self.allocator.free(self.threads);
            self.allocator.free(self.shards);
            self.* = undefined;
        }

        pub fn capacity(self: ParallelSearcher) usize {
            return self.shards.len * self.per_thread;
        }

        pub fn setProbes(self: *ParallelSearcher, probes: usize) !void {
            for (self.shards) |*s| try s.setProbes(probes);
        }
    };

    const ShardWork = struct {
        index: *const IvfIndex,
        queries: []const f32,
        searcher: *Searcher,
        out: []Entry,

        fn run(work: ShardWork) void {
            const found = work.index.searchBatch(work.queries, work.searcher);
            @memcpy(work.out[0..found.len], found);
        }
    };

    /// `searchBatch` spread over the searcher's threads.
    pub fn searchBatchParallel(self: *const IvfIndex, queries: []const f32, searcher: *ParallelSearcher) ![]Entry {
        const d = self.dim();
        std.debug.assert(queries.len % d == 0);
        const n = queries.len / d;
        std.debug.assert(n > 0 and n <= searcher.capacity());
        const threads = @min(searcher.shards.len, n);
        const base = n / threads;
        const extra = n % threads;
        var spawned: usize = 0;
        errdefer for (0..spawned) |i| searcher.threads[i].join();
        var offset: usize = 0;
        while (spawned < threads) : (spawned += 1) {
            const take = base + @intFromBool(spawned < extra);
            const work: ShardWork = .{
                .index = self,
                .queries = queries[offset * d ..][0 .. take * d],
                .searcher = &searcher.shards[spawned],
                .out = searcher.results[offset * searcher.k ..][0 .. take * searcher.k],
            };
            searcher.threads[spawned] = try std.Thread.spawn(.{}, ShardWork.run, .{work});
            offset += take;
        }
        for (searcher.threads[0..threads]) |t| t.join();
        return searcher.results[0 .. n * searcher.k];
    }
};

// -- k-means and centroid ranking ----------------------------------------------------

const V = @Vector(16, f32);

/// ⟨a, b⟩ with a fixed summation order, so the same pair gives the same bits whichever
/// caller computes it and however its rows were blocked.
fn dotExact(a: []const f32, b: []const f32) f32 {
    var acc: V = @splat(0);
    var j: usize = 0;
    while (j + 16 <= a.len) : (j += 16) {
        const x: V = a[j..][0..16].*;
        const y: V = b[j..][0..16].*;
        acc += x * y;
    }
    return finish(acc, a[j..], b[j..]);
}

inline fn finish(acc: V, a_tail: []const f32, b_tail: []const f32) f32 {
    const lanes: [16]f32 = acc;
    var s: f32 = 0;
    for (lanes) |x| s += x;
    for (a_tail, b_tail) |x, y| s += x * y;
    return s;
}

/// `out[r * k + l] = ⟨rows[r], centroids[l]⟩` for every row of `rows` (row-major, `p`
/// wide) and each of `k` centroids. Rows go four at a time so each centroid load serves
/// four dot products; every result is bit-identical to `dotExact`.
fn centroidDots(rows: []const f32, p: usize, centroids: []const f32, k: usize, out: []f32) void {
    const n = rows.len / p;
    var r: usize = 0;
    while (r + 4 <= n) : (r += 4) {
        const a0 = rows[r * p ..][0..p];
        const a1 = rows[(r + 1) * p ..][0..p];
        const a2 = rows[(r + 2) * p ..][0..p];
        const a3 = rows[(r + 3) * p ..][0..p];
        for (0..k) |l| {
            const c = centroids[l * p ..][0..p];
            var s0: V = @splat(0);
            var s1: V = @splat(0);
            var s2: V = @splat(0);
            var s3: V = @splat(0);
            var j: usize = 0;
            while (j + 16 <= p) : (j += 16) {
                const cv: V = c[j..][0..16].*;
                s0 += @as(V, a0[j..][0..16].*) * cv;
                s1 += @as(V, a1[j..][0..16].*) * cv;
                s2 += @as(V, a2[j..][0..16].*) * cv;
                s3 += @as(V, a3[j..][0..16].*) * cv;
            }
            out[r * k + l] = finish(s0, a0[j..], c[j..]);
            out[(r + 1) * k + l] = finish(s1, a1[j..], c[j..]);
            out[(r + 2) * k + l] = finish(s2, a2[j..], c[j..]);
            out[(r + 3) * k + l] = finish(s3, a3[j..], c[j..]);
        }
    }
    while (r < n) : (r += 1) {
        const a = rows[r * p ..][0..p];
        for (0..k) |l| out[r * k + l] = dotExact(a, centroids[l * p ..][0..p]);
    }
}

/// The list nearest a unit direction y, given ⟨y, c_l⟩ for every list: the largest
/// ⟨y, c⟩ − ‖c‖²/2, which is the smallest ‖y − c‖². Ties go to the lower list.
fn nearest(dots: []const f32, centroid_sq: []const f32) usize {
    var best: usize = 0;
    var best_key = -std.math.inf(f32);
    for (dots, centroid_sq, 0..) |dt, sq, l| {
        const key = dt - 0.5 * sq;
        if (key > best_key) {
            best_key = key;
            best = l;
        }
    }
    return best;
}

/// The `out.len` lists nearest a query's direction, given ⟨p, c_l⟩ for every list and
/// ‖p‖, in no particular order. `keys` is scratch of the same length.
fn nearestLists(dots: []const f32, norm: f32, centroid_sq: []const f32, keys: []f32, out: []u32) void {
    const want = out.len;
    std.debug.assert(keys.len == want and want <= dots.len);
    if (want == dots.len) {
        for (out, 0..) |*o, l| o.* = @intCast(l);
        return;
    }
    const inv: f32 = if (norm > 0) 1 / norm else 0;
    // A min-heap of the best `want` keys so far; the root is the worst of them.
    var len: usize = 0;
    for (dots, centroid_sq, 0..) |dt, sq, l| {
        const key = dt * inv - 0.5 * sq;
        if (len < want) {
            var i = len;
            len += 1;
            keys[i] = key;
            out[i] = @intCast(l);
            while (i > 0) {
                const parent = (i - 1) / 2;
                if (!(keys[i] < keys[parent])) break;
                std.mem.swap(f32, &keys[i], &keys[parent]);
                std.mem.swap(u32, &out[i], &out[parent]);
                i = parent;
            }
        } else if (key > keys[0]) {
            keys[0] = key;
            out[0] = @intCast(l);
            var i: usize = 0;
            while (true) {
                const left = 2 * i + 1;
                if (left >= want) break;
                var child = left;
                if (left + 1 < want and keys[left + 1] < keys[left]) child = left + 1;
                if (!(keys[child] < keys[i])) break;
                std.mem.swap(f32, &keys[i], &keys[child]);
                std.mem.swap(u32, &out[i], &out[child]);
                i = child;
            }
        }
    }
}

/// Lloyd's k-means over the rows of `data` (row-major, `p` wide), writing `k` centroids.
/// Seeded from the first `k` rows, which the caller has shuffled. A list left empty takes
/// half of the largest one: its centroid becomes a nudged copy of that one's, and the
/// next assignment splits the members between them. The result is the same whatever
/// `threads` is, since rows are assigned independently and summed in row order.
fn kmeans(a: Allocator, data: []const f32, p: usize, k: usize, threads: usize, centroids: []f32) !void {
    const rows = data.len / p;
    std.debug.assert(rows >= k);
    @memcpy(centroids, data[0 .. k * p]);

    const assign = try a.alloc(u32, rows);
    defer a.free(assign);
    const sums = try a.alloc(f64, k * p);
    defer a.free(sums);
    const sizes = try a.alloc(usize, k);
    defer a.free(sizes);
    const sq = try a.alloc(f32, k);
    defer a.free(sq);

    const workers = @max(1, @min(threads, rows / 1024, 64));
    const dots = try a.alloc(f32, workers * 4 * k);
    defer a.free(dots);

    const Assign = struct {
        data: []const f32,
        p: usize,
        k: usize,
        centroids: []const f32,
        sq: []const f32,
        first: usize,
        last: usize,
        assign: []u32,
        dots: []f32,

        fn run(w: @This()) void {
            var r = w.first;
            while (r < w.last) : (r += 4) {
                const take = @min(4, w.last - r);
                centroidDots(w.data[r * w.p ..][0 .. take * w.p], w.p, w.centroids, w.k, w.dots[0 .. take * w.k]);
                for (0..take) |u| w.assign[r + u] = @intCast(nearest(w.dots[u * w.k ..][0..w.k], w.sq));
            }
        }
    };

    for (0..kmeans_iterations) |_| {
        for (0..k) |l| sq[l] = dotExact(centroids[l * p ..][0..p], centroids[l * p ..][0..p]);
        var handles: [64]?std.Thread = @splat(null);
        const base = rows / workers;
        const rem = rows % workers;
        var start: usize = 0;
        var own: Assign = undefined;
        for (0..workers) |w| {
            const take = base + @intFromBool(w < rem);
            const work: Assign = .{
                .data = data,
                .p = p,
                .k = k,
                .centroids = centroids,
                .sq = sq,
                .first = start,
                .last = start + take,
                .assign = assign,
                .dots = dots[w * 4 * k ..][0 .. 4 * k],
            };
            start += take;
            if (w == 0) {
                own = work;
                continue;
            }
            handles[w] = std.Thread.spawn(.{}, Assign.run, .{work}) catch blk: {
                work.run();
                break :blk null;
            };
        }
        own.run();
        for (handles[0..workers]) |h| if (h) |t| t.join();

        @memset(sums, 0);
        @memset(sizes, 0);
        for (0..rows) |r| {
            const l = assign[r];
            sizes[l] += 1;
            for (sums[l * p ..][0..p], data[r * p ..][0..p]) |*s, x| s.* += x;
        }
        for (0..k) |l| {
            if (sizes[l] == 0) continue;
            const inv = 1 / @as(f64, @floatFromInt(sizes[l]));
            for (centroids[l * p ..][0..p], sums[l * p ..][0..p]) |*c, s| c.* = @floatCast(s * inv);
        }
        for (0..k) |l| {
            if (sizes[l] != 0) continue;
            var largest: usize = 0;
            for (sizes, 0..) |s, i| {
                if (s > sizes[largest]) largest = i;
            }
            const src = centroids[largest * p ..][0..p];
            const dst = centroids[l * p ..][0..p];
            for (dst, src, 0..) |*o, *s, j| {
                const nudge: f32 = if (j % 2 == 0) 1.0 / 1024.0 else -1.0 / 1024.0;
                o.* = s.* * (1 + nudge);
                s.* = s.* * (1 - nudge);
            }
            sizes[largest] -= sizes[largest] / 2;
            sizes[l] = sizes[largest];
        }
    }
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;

/// `rows` vectors around `clusters` random centres, plus a shared offset, with norms
/// spread over [0.5, 2.5) so inner product and cosine rank differently.
fn clusteredRows(a: Allocator, rows: usize, dim: usize, clusters: usize, random: std.Random) ![]f32 {
    const centres = try a.alloc(f32, clusters * dim);
    defer a.free(centres);
    for (centres) |*c| c.* = random.floatNorm(f32);
    const out = try a.alloc(f32, rows * dim);
    for (0..rows) |r| {
        const c = random.uintLessThan(usize, clusters);
        const row = out[r * dim ..][0..dim];
        var norm: f64 = 0;
        for (row, centres[c * dim ..][0..dim]) |*x, cv| {
            x.* = cv + 0.6 * random.floatNorm(f32) + 0.5;
            norm += @as(f64, x.*) * x.*;
        }
        const scale: f32 = @floatCast((0.5 + 2 * random.float(f64)) / @sqrt(norm));
        for (row) |*x| x.* *= scale;
    }
    return out;
}

/// Exact top-k ids of `query` over `rows` by inner product.
fn exactTop(a: Allocator, rows: []const f32, dim: usize, query: []const f32, k: usize) ![]u32 {
    const n = rows.len / dim;
    const buf = try a.alloc(Entry, k);
    defer a.free(buf);
    var heap = topk.TopK.init(buf);
    for (0..n) |r| {
        var s: f32 = 0;
        for (rows[r * dim ..][0..dim], query) |x, q| s += x * q;
        heap.offer(s, @intCast(r));
    }
    const top = heap.drain();
    const ids = try a.alloc(u32, top.len);
    for (ids, top) |*o, e| o.* = e.id;
    return ids;
}

fn hits(found: []const Entry, truth: []const u32) usize {
    var h: usize = 0;
    for (found) |e| {
        for (truth) |t| {
            if (e.id == t) h += 1;
        }
    }
    return h;
}

test "one list scores as a calibrated flat index does" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const dim = 96;
    const data = try clusteredRows(a, 600, dim, 8, prng.random());
    defer a.free(data);
    const queries = try clusteredRows(a, 9, dim, 8, prng.random());
    defer a.free(queries);

    for ([_]Metric{ .inner_product, .cosine, .l2 }) |metric| {
        var fl = try flat.FlatIndex.init(a, .{ .dim = dim, .bits = 3, .metric = metric, .seed = 5 });
        defer fl.deinit();
        try fl.calibrate(data[0 .. 200 * dim]);
        try fl.addBatch(data);

        var ivf = try IvfIndex.init(a, .{ .dim = dim, .bits = 3, .lists = 1, .metric = metric, .seed = 5 });
        defer ivf.deinit();
        try ivf.train(data[0 .. 200 * dim], 1);
        // The k-means mean sums the same rows in another order; take the flat index's
        // centre exactly, so the codes, factors and biases are the same bytes.
        @memcpy(ivf.centroids, fl.centre);
        ivf.centroid_sq[0] = dotExact(ivf.centroids, ivf.centroids);
        try ivf.addBatch(data);

        var fs = try flat.FlatIndex.BatchSearcher.init(a, fl, 9, 10);
        defer fs.deinit();
        var is = try IvfIndex.Searcher.init(a, ivf, 9, 10, 1);
        defer is.deinit();
        const want = fl.searchBatch(queries, &fs);
        const got = ivf.searchBatch(queries, &is);
        for (want, got) |w, g| {
            try testing.expectEqual(w.id, g.id);
            // ⟨p, c⟩ is summed in f64 by the flat index and in f32 here.
            try testing.expectApproxEqAbs(w.score, g.score, 1e-4 * @max(1, @abs(w.score)));
        }
    }
}

test "batched search agrees with one query at a time" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(12);
    const dim = 128;
    const data = try clusteredRows(a, 3000, dim, 24, prng.random());
    defer a.free(data);
    const nq = 37;
    const queries = try clusteredRows(a, nq, dim, 24, prng.random());
    defer a.free(queries);

    const Case = struct { metric: Metric, bits: u6, residency: Residency };
    const cases = [_]Case{
        .{ .metric = .inner_product, .bits = 3, .residency = .compact },
        .{ .metric = .cosine, .bits = 5, .residency = .compact },
        .{ .metric = .l2, .bits = 4, .residency = .compact },
        .{ .metric = .inner_product, .bits = 5, .residency = .expanded },
        .{ .metric = .l2, .bits = 2, .residency = .expanded },
    };
    for (cases) |c| {
        var ivf = try IvfIndex.init(a, .{ .dim = dim, .bits = c.bits, .lists = 16, .metric = c.metric, .seed = 3, .residency = c.residency });
        defer ivf.deinit();
        try ivf.train(data, 2);
        try ivf.addBatch(data);

        const k = 10;
        var batch = try IvfIndex.Searcher.init(a, ivf, nq, k, 5);
        defer batch.deinit();
        var one = try IvfIndex.Searcher.init(a, ivf, 1, k, 5);
        defer one.deinit();
        const together = ivf.searchBatch(queries, &batch);
        for (0..nq) |i| {
            const alone = ivf.search(queries[i * dim ..][0..dim], &one);
            for (alone, together[i * k ..][0..k]) |x, y| {
                try testing.expectEqual(x.id, y.id);
                try testing.expectEqual(x.score, y.score);
            }
        }
    }
}

test "partitioning lifts recall, and more probes lift it further" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(13);
    const dim = 64;
    const n = 4000;
    const data = try clusteredRows(a, n, dim, 32, prng.random());
    defer a.free(data);
    const nq = 40;
    const queries = try clusteredRows(a, nq, dim, 32, prng.random());
    defer a.free(queries);
    const k = 10;

    var fl = try flat.FlatIndex.init(a, .{ .dim = dim, .bits = 2, .seed = 1 });
    defer fl.deinit();
    try fl.calibrate(data);
    try fl.addBatch(data);
    var fs = try flat.FlatIndex.BatchSearcher.init(a, fl, nq, k);
    defer fs.deinit();
    const flat_found = fl.searchBatch(queries, &fs);

    var ivf = try IvfIndex.init(a, .{ .dim = dim, .bits = 2, .lists = 32, .seed = 1 });
    defer ivf.deinit();
    try ivf.train(data, 1);
    try ivf.addBatch(data);

    var flat_hits: usize = 0;
    var by_probes = [_]usize{0} ** 3;
    const probes = [_]usize{ 1, 4, 32 };
    for (probes, &by_probes) |pr, *total| {
        var s = try IvfIndex.Searcher.init(a, ivf, nq, k, pr);
        defer s.deinit();
        const found = ivf.searchBatch(queries, &s);
        for (0..nq) |i| {
            const truth = try exactTop(a, data, dim, queries[i * dim ..][0..dim], k);
            defer a.free(truth);
            total.* += hits(found[i * k ..][0..k], truth);
            if (pr == 1) flat_hits += hits(flat_found[i * k ..][0..k], truth);
        }
    }
    // Scanning every list, the only difference from the flat index is the finer
    // centres. Here that takes 137 of 400 to 197. More probes need not help at every
    // step: a far list's vectors can be overestimated into the top k (4 probes find
    // 198), so only the first step is required to gain.
    try testing.expect(by_probes[2] > flat_hits + flat_hits / 5);
    try testing.expect(by_probes[1] > by_probes[0]);
}

test "short lists pad the results" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(14);
    const dim = 32;
    const data = try clusteredRows(a, 64, dim, 4, prng.random());
    defer a.free(data);
    var ivf = try IvfIndex.init(a, .{ .dim = dim, .bits = 3, .lists = 4, .seed = 2 });
    defer ivf.deinit();
    try ivf.train(data, 1);
    try ivf.addBatch(data[0 .. 3 * dim]);
    var s = try IvfIndex.Searcher.init(a, ivf, 1, 8, 4);
    defer s.deinit();
    const found = ivf.search(data[0..dim], &s);
    for (found[0..3]) |e| try testing.expect(e.id < 3);
    for (found[3..]) |e| {
        try testing.expectEqual(no_id, e.id);
        try testing.expectEqual(-std.math.inf(f32), e.score);
    }
}

test "training and adding give the same bytes on any thread count" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(15);
    const dim = 48;
    const data = try clusteredRows(a, 5000, dim, 12, prng.random());
    defer a.free(data);

    var one = try IvfIndex.init(a, .{ .dim = dim, .bits = 4, .lists = 20, .seed = 9 });
    defer one.deinit();
    try one.train(data, 1);
    try one.addBatchParallel(data, 1);
    var many = try IvfIndex.init(a, .{ .dim = dim, .bits = 4, .lists = 20, .seed = 9 });
    defer many.deinit();
    try many.train(data, 4);
    try many.addBatchParallel(data[0 .. 1200 * dim], 4);
    try many.addBatchParallel(data[1200 * dim ..], 3);

    try testing.expectEqualSlices(f32, one.centroids, many.centroids);
    try testing.expectEqual(one.count(), many.count());
    for (one.lists, many.lists) |x, y| {
        try testing.expectEqualSlices(u8, x.codes.items, y.codes.items);
        try testing.expectEqualSlices(u32, x.ids.items, y.ids.items);
        try testing.expectEqualSlices(f16, x.biases.items, y.biases.items);
        for (x.scalars.items, y.scalars.items) |p, q| try testing.expect(std.meta.eql(p, q));
    }
}

test "misuse is refused" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(16);
    const dim = 16;
    const data = try clusteredRows(a, 40, dim, 2, prng.random());
    defer a.free(data);
    var ivf = try IvfIndex.init(a, .{ .dim = dim, .bits = 3, .lists = 8 });
    defer ivf.deinit();
    try testing.expectError(error.NotTrained, ivf.addBatch(data));
    try testing.expectError(error.EmptySample, ivf.train(data[0 .. 5 * dim], 1));
    try ivf.train(data, 1);
    try ivf.addBatch(data);
    try testing.expectError(error.IndexNotEmpty, ivf.train(data, 1));
    try testing.expectError(error.BadListCount, IvfIndex.init(a, .{ .dim = dim, .bits = 3, .lists = 0 }));
}
