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
//! Per vector: the codes, the f16 norm, factor and bias of a calibrated flat index, a u32
//! id, since lists hold vectors out of insertion order, and a u16 naming the vector's list,
//! so `scoreIds` can find a vector by id. Per index: the centroids, `lists × padded`
//! floats.
//!
//! A list holds its vectors in the order they were added, so its ids rise. Finding a
//! vector by id is a lookup of its list and a binary search there.

const std = @import("std");
const Allocator = std.mem.Allocator;
const coarse_mod = @import("coarse.zig");
const pool = @import("pool.zig");
const dotExact = coarse_mod.dotExact;
const centroidDots = coarse_mod.centroidDots;

const flat = @import("flat.zig");
const prod_mod = @import("../quant/prod.zig");
const packing = @import("../quant/packing.zig");
const scan = @import("../simd/scan.zig");
const tiles = @import("../simd/tiles.zig");
const dispatch = @import("../simd/dispatch.zig");
const columns = @import("../simd/columns.zig");
const Codes = @import("codes.zig").Codes;
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
    /// The list each vector went to, by id.
    list_of: std.ArrayList(u16),
    trained: bool,
    total: usize,
    allocator: Allocator,

    pub const List = struct {
        /// One vector after another, or by column at one bit; see `IvfIndex.byColumn`.
        codes: Codes,
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

        var index: IvfIndex = .{
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
            .list_of = .empty,
            .trained = false,
            .total = 0,
            .allocator = allocator,
        };
        for (lists) |*l| l.* = .{ .codes = Codes.init(index.codeStride(), index.byColumn()) };
        return index;
    }

    /// Whether lists store codes by column, for the one-bit block kernels; the same rule
    /// as `FlatIndex.byColumn`.
    pub fn byColumn(self: IvfIndex) bool {
        return self.params.residency == .compact and self.layout.bits == 1 and
            self.layout.kind() == .sequential and self.vectorized();
    }

    pub fn deinit(self: *IvfIndex) void {
        for (self.lists) |*l| l.deinit(self.allocator);
        self.allocator.free(self.lists);
        self.list_of.deinit(self.allocator);
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

    /// Bytes stored per vector: codes, three f16 scalars, a u32 id and a u16 list. The
    /// centroids are per index; `centroidBytes` reports them.
    pub fn bytesPerVector(self: IvfIndex) usize {
        return self.codeStride() + @sizeOf(StoredScalars) + @sizeOf(f16) + @sizeOf(u32) + @sizeOf(u16);
    }

    /// Where vector `id` is stored: its list, and its slot there.
    pub fn locate(self: IvfIndex, id: u32) struct { list: usize, slot: usize } {
        std.debug.assert(id < self.total);
        const l = self.list_of.items[id];
        const ids = self.lists[l].ids.items;
        const slot = std.sort.binarySearch(u32, ids, id, struct {
            fn order(want: u32, have: u32) std.math.Order {
                return std.math.order(want, have);
            }
        }.order) orelse unreachable;
        return .{ .list = l, .slot = slot };
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
        const rows = try self.trainingRows(sample, order, data, threads);
        if (rows < k) return error.EmptySample;

        try kmeans(a, data[0 .. rows * p], p, k, threads, self.centroids);
        self.useCentroids();
    }

    /// Rotate the rows of `sample` in `order` into `data` until it's full or `order` runs
    /// out, skipping rows of zero, which have no direction. Returns the rows written:
    /// the same rows in the same order whatever `threads` is.
    fn trainingRows(self: *const IvfIndex, sample: []const f32, order: []const u32, data: []f32, threads: usize) !usize {
        const a = self.allocator;
        const p = self.padded();
        const cap = data.len / p;
        const chunk = 256;
        const spread = @max(1, @min(threads, cap / 1024));
        const workspaces = try a.alloc(prod_mod.Workspace, pool.workers(spread, cap, chunk));
        defer a.free(workspaces);
        var made: usize = 0;
        defer for (workspaces[0..made]) |*ws| ws.deinit();
        while (made < workspaces.len) : (made += 1) workspaces[made] = try prod_mod.Workspace.init(a, self.quantizer);
        const kept = try a.alloc(bool, cap);
        defer a.free(kept);

        const Batch = struct {
            index: *const IvfIndex,
            sample: []const f32,
            candidates: []const u32,
            out: []f32,
            kept: []bool,
            workspaces: []prod_mod.Workspace,

            fn run(b: @This(), worker: usize, start: usize, len: usize) void {
                const d_ = b.index.dim();
                const p_ = b.index.padded();
                for (start..start + len) |i| {
                    const row = b.sample[@as(usize, b.candidates[i]) * d_ ..][0..d_];
                    const norm = b.index.quantizer.mse.encodeRotated(row, b.out[i * p_ ..][0..p_], b.workspaces[worker].staging);
                    b.kept[i] = norm != 0;
                }
            }
        };
        // A batch of candidates at a time, encoded in parallel into the free rows, then
        // the zero rows squeezed out.
        var rows: usize = 0;
        var taken: usize = 0;
        while (rows < cap and taken < order.len) {
            const batch = @min(cap - rows, order.len - taken);
            const work: Batch = .{
                .index = self,
                .sample = sample,
                .candidates = order[taken..][0..batch],
                .out = data[rows * p ..][0 .. batch * p],
                .kept = kept[0..batch],
                .workspaces = workspaces,
            };
            pool.forChunks(spread, batch, chunk, work, Batch.run);
            var to = rows;
            for (0..batch) |i| {
                if (!kept[i]) continue;
                if (to != rows + i) @memcpy(data[to * p ..][0..p], data[(rows + i) * p ..][0..p]);
                to += 1;
            }
            rows = to;
            taken += batch;
        }
        return rows;
    }

    /// Take `centroids` as written, by training or by a loader, and mark the index
    /// trained.
    pub fn useCentroids(self: *IvfIndex) void {
        for (0..self.lists.len) |l| self.centroid_sq[l] = dotExact(self.centroid(l), self.centroid(l));
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
        var near = try coarse_mod.Nearest.init(a, self.lists.len, self.padded());
        defer near.deinit(a);
        near.load(self.centroids);
        const workers = @max(1, @min(threads, chunk / min_rows_per_thread));
        const encoders = try a.alloc(Encoder, workers);
        defer a.free(encoders);
        var made: usize = 0;
        defer for (encoders[0..made]) |*e| e.deinit(a);
        while (made < workers) : (made += 1) encoders[made] = try Encoder.init(a, self, &near);

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
            try l.codes.ensureUnusedCapacity(a, extra);
            try l.scalars.ensureUnusedCapacity(a, extra);
            try l.biases.ensureUnusedCapacity(a, extra);
            try l.ids.ensureUnusedCapacity(a, extra);
        }
        try self.list_of.ensureUnusedCapacity(a, n);

        start = 0;
        while (start < n) : (start += chunk) {
            const take = @min(chunk, n - start);
            var out = staged.slice(0, take, stride);
            out.lists = assigned[start..][0..take];
            self.runEncode(vectors[start * d ..][0 .. take * d], out, encoders, .encode);
            for (0..take) |v| {
                const l = &self.lists[assigned[start + v]];
                l.codes.appendAssumeCapacity(staged.codes[v * stride ..][0..stride]);
                l.scalars.appendAssumeCapacity(staged.scalars[v]);
                l.biases.appendAssumeCapacity(staged.biases[v]);
                l.ids.appendAssumeCapacity(@intCast(self.total + start + v));
                self.list_of.appendAssumeCapacity(@intCast(assigned[start + v]));
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

    /// Assign or encode every row of `vectors` into `out`, on up to `encoders.len`
    /// threads taking a block of rows at a time. Each row lands in its own slot.
    fn runEncode(self: *const IvfIndex, vectors: []const f32, out: Staging, encoders: []Encoder, mode: Mode) void {
        const d = self.dim();
        const n = vectors.len / d;
        const Work = struct {
            index: *const IvfIndex,
            vectors: []const f32,
            out: Staging,
            encoders: []Encoder,
            mode: Mode,

            fn run(w: @This(), worker: usize, start: usize, len: usize) void {
                const d_ = w.index.dim();
                const rows = w.vectors[start * d_ ..][0 .. len * d_];
                w.index.encodeRows(rows, w.out.slice(start, len, w.index.codeStride()), &w.encoders[worker], w.mode);
            }
        };
        const spread = @max(1, @min(encoders.len, n / min_rows_per_thread));
        const work: Work = .{ .index = self, .vectors = vectors, .out = out, .encoders = encoders, .mode = mode };
        pool.forChunks(spread, n, Encoder.block, work, Work.run);
    }

    /// Per-thread encoding scratch.
    const Encoder = struct {
        ws: prod_mod.Workspace,
        codes: []u8,
        /// Up to `block` rotated directions.
        rotated: []f32,
        near: *const coarse_mod.Nearest,
        scratch: coarse_mod.Nearest.Scratch,

        /// Rows assigned together: enough that each block of centroids the int8 pass
        /// holds in L1 serves many rows.
        const block = coarse_mod.max_rows;

        fn init(a: Allocator, index: *const IvfIndex, near: *const coarse_mod.Nearest) !Encoder {
            var ws = try prod_mod.Workspace.init(a, index.quantizer);
            errdefer ws.deinit();
            const codes = try a.alloc(u8, index.padded());
            errdefer a.free(codes);
            const rotated = try a.alloc(f32, block * index.padded());
            errdefer a.free(rotated);
            const scratch = try coarse_mod.Nearest.Scratch.init(a, near);
            return .{ .ws = ws, .codes = codes, .rotated = rotated, .near = near, .scratch = scratch };
        }

        fn deinit(self: *Encoder, a: Allocator) void {
            self.scratch.deinit(a);
            a.free(self.rotated);
            a.free(self.codes);
            self.ws.deinit();
        }
    };

    fn encodeRows(self: *const IvfIndex, vectors: []const f32, out: Staging, enc: *Encoder, mode: Mode) void {
        const d = self.dim();
        const p = self.padded();
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
                enc.near.assign(enc.rotated[0 .. take * p], self.centroids, self.centroid_sq, &enc.scratch, out.lists[r0..][0..take]);
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

    /// Score the stored vectors `ids` against one query: `out[j]` is the score a search
    /// probing `ids[j]`'s list would report for it, bit for bit. For ranking candidates
    /// chosen elsewhere, such as by a filter. Probes play no part: every id is scored.
    pub fn scoreIds(
        self: *const IvfIndex,
        query: []const f32,
        ids: []const u32,
        out: []f32,
        searcher: *Searcher,
    ) error{IdOutOfRange}!void {
        std.debug.assert(query.len == self.dim() and ids.len == out.len);
        for (ids) |id| if (id >= self.total) return error.IdOutOfRange;
        if (ids.len == 0) return;
        const state = &searcher.query_states[0];
        self.quantizer.prepareQueryRotated(query, state, &searcher.workspace);
        const use_simd = self.vectorized();
        if (use_simd) searcher.scan_queries[0].load(state.rotated);
        const norm = flat.euclideanNorm(query);
        const levels = self.quantizer.mse.codebook.centroids;

        // ⟨p, c_l⟩ for every list in one pass once there are enough ids to need most of
        // them, else one list at a time. Both give the bits a search uses.
        const lists = self.lists.len;
        const all_dots = ids.len >= lists / 4;
        if (all_dots) centroidDots(state.rotated, self.padded(), self.centroids, lists, searcher.coarse[0..lists]);

        for (ids, out) |id, *o| {
            const at = self.locate(id);
            const list = &self.lists[at.list];
            const centre_dot = if (all_dots) searcher.coarse[at.list] else dotExact(state.rotated, self.centroid(at.list));
            const terms = flat.queryTerms(self.params.metric, norm, centre_dot);
            const mse = if (use_simd)
                self.mseTerm(list, at.slot, searcher.scan_queries[0], searcher)
            else
                scan.scoreExact(self.layout, levels, state.rotated, list.codes.row(at.slot));
            o.* = flat.scalarScore(true, self.vectorTermsAt(list, at.slot), terms, mse);
        }
    }

    fn job(self: *const IvfIndex, list: *const List, searcher: *const Searcher) tiles.Job {
        return .{
            .table = &searcher.table,
            .queries = undefined,
            .n = undefined,
            .codes = list.codes.ptr(),
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
        if (list.codes.by_column) return list.codes.oneBitTerm(v, query, searcher.table);
        const slot = list.codes.row(v);
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
        // 16 vectors at a time for one-bit column blocks, as in `FlatIndex.searchBatch`.
        const by_column = list.codes.by_column;
        const tile: usize = if (by_column) columns.block else tiles.batch_tile;
        const lanes = flat.lanes;
        const Lane = flat.Lane;
        const m = group.m;
        const p: u32 = @intCast(self.padded());
        var mse_block: [columns.block * flat.max_batch]f32 = undefined;
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
            if (by_column) {
                dispatch.columnTile(kernels, &jb, v0 / columns.block);
            } else if (can_tile and take == tile) {
                dispatch.batchTile(kernels, &jb, v0);
            } else if (self.params.residency == .expanded) {
                for (0..take) |u| for (0..m) |j| {
                    mse_block[u * flat.max_batch + j] = self.mseTerm(list, v0 + u, group.queries[j], searcher);
                };
            } else for (0..take) |u| {
                // The unpack is query-independent, so the group shares it.
                const slot = list.codes.row(v0 + u);
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
        comptime std.debug.assert(block.len % columns.block == 0);
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
            if (list.codes.by_column) {
                // The list's last block is stored whole, so it is scored whole.
                dispatch.columnSingle(kernels, &jb, v0 / columns.block, (take + columns.block - 1) / columns.block);
            } else {
                const tiled = if (can_tile) take / tile * tile else 0;
                if (tiled > 0) dispatch.singleTiles(kernels, &jb, v0, tiled / tile);
                for (tiled..take) |u| block[u] = self.mseTerm(list, v0 + u, group.queries[j], searcher);
            }
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
        const levels = self.quantizer.mse.codebook.centroids;
        for (0..group.m) |j| {
            const q = group.owner[j];
            flat.flushHot(&col.collectors[q], &col.hot[q], &col.hot_len[q], &group.thresholds[j]);
        }
        for (0..list.len()) |v| {
            const slot = list.codes.row(v);
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
    return kmeansWith(a, data, p, k, threads, centroids, !coarse_mod.bounded);
}

/// `kmeans`, with the assignment's path chosen: exhaustive, or by `coarse_mod.Nearest`'s
/// int8 pre-pass. Both give the same centroids, bit for bit.
fn kmeansWith(a: Allocator, data: []const f32, p: usize, k: usize, threads: usize, centroids: []f32, exhaustive: bool) !void {
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

    // Below about 1024 rows a thread, spawning costs more than it saves.
    const spread = @max(1, @min(threads, rows / 1024));
    const workers = pool.workers(spread, rows, coarse_mod.max_rows);
    var near = try coarse_mod.Nearest.init(a, k, p);
    defer near.deinit(a);
    near.exhaustive = near.exhaustive or exhaustive;
    const scratch = try a.alloc(coarse_mod.Nearest.Scratch, workers);
    defer a.free(scratch);
    var made: usize = 0;
    defer for (scratch[0..made]) |*s| s.deinit(a);
    while (made < workers) : (made += 1) scratch[made] = try coarse_mod.Nearest.Scratch.init(a, &near);

    const Step = struct {
        data: []const f32,
        p: usize,
        near: *const coarse_mod.Nearest,
        centroids: []f32,
        sq: []const f32,
        assign: []u32,
        scratch: []coarse_mod.Nearest.Scratch,
        sums: []f64,
        sizes: []const usize,

        /// Rows `start..start + len` to their nearest lists.
        fn nearestRows(w: @This(), worker: usize, start: usize, len: usize) void {
            const rows_ = w.data[start * w.p ..][0 .. len * w.p];
            w.near.assign(rows_, w.centroids, w.sq, &w.scratch[worker], w.assign[start..][0..len]);
        }

        /// Columns `j0..j0 + width` of every list's mean. Each sum adds its rows in row
        /// order, as one thread would.
        fn means(w: @This(), _: usize, j0: usize, width: usize) void {
            const k_ = w.sizes.len;
            for (0..k_) |l| @memset(w.sums[l * w.p + j0 ..][0..width], 0);
            for (w.assign, 0..) |l, r| {
                const src = w.data[r * w.p + j0 ..][0..width];
                for (w.sums[l * w.p + j0 ..][0..width], src) |*acc, x| acc.* += x;
            }
            for (0..k_) |l| {
                if (w.sizes[l] == 0) continue;
                const inv = 1 / @as(f64, @floatFromInt(w.sizes[l]));
                for (w.centroids[l * w.p + j0 ..][0..width], w.sums[l * w.p + j0 ..][0..width]) |*c, acc| c.* = @floatCast(acc * inv);
            }
        }
    };
    const step: Step = .{
        .data = data,
        .p = p,
        .near = &near,
        .centroids = centroids,
        .sq = sq,
        .assign = assign,
        .scratch = scratch,
        .sums = sums,
        .sizes = sizes,
    };

    for (0..kmeans_iterations) |_| {
        for (0..k) |l| sq[l] = dotExact(centroids[l * p ..][0..p], centroids[l * p ..][0..p]);
        near.load(centroids);
        pool.forChunks(spread, rows, coarse_mod.max_rows, step, Step.nearestRows);

        @memset(sizes, 0);
        for (assign) |l| sizes[l] += 1;
        pool.forChunks(spread, p, 16, step, Step.means);
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

test "k-means gives the same centroids by int8 pre-pass as exhaustively" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x4B4D);
    // Clustered rows, normalised as training's directions are, at a width that isn't a
    // multiple of 16 and a list count that isn't a multiple of 4.
    for ([_][2]usize{ .{ 96, 37 }, .{ 200, 10 } }) |shape| {
        const p = shape[0];
        const k = shape[1];
        const data = try clusteredRows(a, 5000, p, k / 2 + 1, prng.random());
        defer a.free(data);
        for (0..5000) |r| {
            const row = data[r * p ..][0..p];
            const inv = 1 / @sqrt(dotExact(row, row));
            for (row) |*x| x.* *= inv;
        }
        const want = try a.alloc(f32, k * p);
        defer a.free(want);
        try kmeansWith(a, data, p, k, 1, want, true);
        for ([_]usize{ 1, 4 }) |threads| {
            const got = try a.alloc(f32, k * p);
            defer a.free(got);
            try kmeansWith(a, data, p, k, threads, got, false);
            try testing.expectEqualSlices(f32, want, got);
        }
    }
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
        ivf.useCentroids();
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
        try testing.expectEqualSlices(u8, x.codes.bytes.items, y.codes.bytes.items);
        try testing.expectEqualSlices(u32, x.ids.items, y.ids.items);
        try testing.expectEqualSlices(f16, x.biases.items, y.biases.items);
        for (x.scalars.items, y.scalars.items) |p, q| try testing.expect(std.meta.eql(p, q));
    }
}

test "training takes the rows the serial loop did, zero rows skipped" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(21);
    const random = prng.random();
    const dim = 40;
    const n = 3000;
    const data = try clusteredRows(a, n, dim, 6, random);
    defer a.free(data);
    // Every fifth row zero, so each batch leaves gaps for the next to fill.
    var r: usize = 0;
    while (r < n) : (r += 5) @memset(data[r * dim ..][0..dim], 0);
    const order = try a.alloc(u32, n);
    defer a.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    random.shuffle(u32, order);

    var index = try IvfIndex.init(a, .{ .dim = dim, .bits = 3, .lists = 8 });
    defer index.deinit();
    const p = index.padded();
    for ([_]usize{ 2048, 2900 }) |cap| {
        // The loop trainingRows replaced.
        const want = try a.alloc(f32, cap * p);
        defer a.free(want);
        var ws = try prod_mod.Workspace.init(a, index.quantizer);
        defer ws.deinit();
        var rows: usize = 0;
        for (order) |o| {
            if (rows == cap) break;
            const row = data[@as(usize, o) * dim ..][0..dim];
            if (index.quantizer.mse.encodeRotated(row, want[rows * p ..][0..p], ws.staging) == 0) continue;
            rows += 1;
        }
        for ([_]usize{ 1, 2, 4 }) |threads| {
            const got = try a.alloc(f32, cap * p);
            defer a.free(got);
            try testing.expectEqual(rows, try index.trainingRows(data, order, got, threads));
            try testing.expectEqualSlices(f32, want[0 .. rows * p], got[0 .. rows * p]);
        }
    }
}

test "scoreIds reports what search does, for any id and any probe count" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(17);
    const dim = 96;
    const data = try clusteredRows(a, 2000, dim, 16, prng.random());
    defer a.free(data);
    const nq = 6;
    const queries = try clusteredRows(a, nq, dim, 16, prng.random());
    defer a.free(queries);

    const Case = struct { metric: Metric, bits: u6, residency: Residency };
    for ([_]Case{
        .{ .metric = .inner_product, .bits = 2, .residency = .compact },
        .{ .metric = .cosine, .bits = 4, .residency = .compact },
        .{ .metric = .l2, .bits = 3, .residency = .expanded },
        .{ .metric = .inner_product, .bits = 6, .residency = .compact }, // exact scan
    }) |c| {
        var ivf = try IvfIndex.init(a, .{ .dim = dim, .bits = c.bits, .lists = 24, .metric = c.metric, .seed = 4, .residency = c.residency });
        defer ivf.deinit();
        try ivf.train(data, 1);
        try ivf.addBatch(data[0 .. 1500 * dim]);
        try ivf.addBatch(data[1500 * dim ..]);

        const k = 20;
        var s = try IvfIndex.Searcher.init(a, ivf, nq, k, 5);
        defer s.deinit();
        var scorer = try IvfIndex.Searcher.init(a, ivf, 1, 1, 1);
        defer scorer.deinit();
        const found = ivf.searchBatch(queries, &s);
        for (0..nq) |i| {
            const q = queries[i * dim ..][0..dim];
            const row = found[i * k ..][0..k];
            var ids: [k]u32 = undefined;
            var want: [k]f32 = undefined;
            for (row, &ids, &want) |e, *id, *w| {
                id.* = e.id;
                w.* = e.score;
            }
            // Few ids score their lists one at a time, many score every list at once;
            // both must give search's bits.
            var got: [k]f32 = undefined;
            try ivf.scoreIds(q, ids[0..3], got[0..3], &scorer);
            try testing.expectEqualSlices(f32, want[0..3], got[0..3]);
            try ivf.scoreIds(q, &ids, &got, &scorer);
            try testing.expectEqualSlices(f32, &want, &got);
        }

        // Every id is found where it was stored.
        for (0..ivf.count()) |id| {
            const at = ivf.locate(@intCast(id));
            try testing.expectEqual(@as(u32, @intCast(id)), ivf.lists[at.list].ids.items[at.slot]);
        }
        var out: [1]f32 = undefined;
        try testing.expectError(error.IdOutOfRange, ivf.scoreIds(queries[0..dim], &.{@intCast(ivf.count())}, &out, &scorer));
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

test "one-bit lists stored by column: batched, single, by id and reloaded agree" {
    // Lists hold one-bit codes by column, each with its own partly used last block. A
    // list read by fewer than four queries takes the one-query path and the rest the
    // batched one; both, scoreIds and a reloaded index must report the same bits.
    const a = testing.allocator;
    const persist = @import("persist.zig");
    var prng = std.Random.DefaultPrng.init(0x1B1F);
    const dim = 256;
    const data = try clusteredRows(a, 1500, dim, 12, prng.random());
    defer a.free(data);
    const nq = 9;
    const queries = try clusteredRows(a, nq, dim, 12, prng.random());
    defer a.free(queries);

    for ([_]Metric{ .inner_product, .cosine, .l2 }) |metric| {
        var ivf = try IvfIndex.init(a, .{ .dim = dim, .bits = 2, .lists = 10, .metric = metric, .seed = 9 });
        defer ivf.deinit();
        try testing.expect(ivf.byColumn());
        try ivf.train(data, 1);
        try ivf.addBatch(data[0 .. 1000 * dim]);
        try ivf.addBatch(data[1000 * dim ..]);

        const buf = try a.alloc(u8, persist.serializedSizePartitioned(&ivf));
        defer a.free(buf);
        _ = try persist.writePartitioned(&ivf, buf);
        var loaded = try persist.readPartitioned(a, buf);
        defer loaded.deinit();

        const k = 12;
        for ([_]usize{ 1, 3, 10 }) |probes| {
            var batch = try IvfIndex.Searcher.init(a, ivf, nq, k, probes);
            defer batch.deinit();
            var one = try IvfIndex.Searcher.init(a, ivf, 1, k, probes);
            defer one.deinit();
            var again = try IvfIndex.Searcher.init(a, loaded, 1, k, probes);
            defer again.deinit();
            const together = ivf.searchBatch(queries, &batch);
            for (0..nq) |i| {
                const q = queries[i * dim ..][0..dim];
                var alone: [k]Entry = undefined;
                const found = ivf.search(q, &one);
                @memcpy(alone[0..found.len], found);
                const reloaded = loaded.search(q, &again);
                var ids: [k]u32 = undefined;
                for (alone[0..found.len], 0..) |e, j| ids[j] = e.id;
                var scores: [k]f32 = undefined;
                try ivf.scoreIds(q, ids[0..found.len], scores[0..found.len], &one);
                for (alone[0..found.len], together[i * k ..][0..found.len], reloaded, scores[0..found.len]) |x, y, z, w| {
                    try testing.expectEqual(x.id, y.id);
                    try testing.expectEqual(x.score, y.score);
                    try testing.expectEqual(x.id, z.id);
                    try testing.expectEqual(x.score, z.score);
                    try testing.expectEqual(x.score, w);
                }
            }
        }
    }
}
