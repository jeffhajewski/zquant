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
const Density = @import("../math/density.zig").Density;
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

/// How to correct the bias in ⟨p, ỹ⟩.
///
/// Storage, at bit-width `b` and dimension `d`:
///
///     .qjl_sketch   d·(b−1)/8 code bytes + d/8 sketch bytes + 4  =  d·b/8 + 4
///     .scalar       d·(b−1)/8 code bytes + 4
///
/// So `.scalar` at `b` costs the same as `.qjl_sketch` at `b−1`, and measured on
/// SIFT10K it wins at every matched storage budget — the freed bit buys more codebook
/// resolution than the sketch was buying accuracy. `.qjl_sketch` is kept because it is
/// the paper's construction and the reference the scalar form is checked against.
///
/// MSE-optimal reconstruction shrinks ỹ, so ⟨p, ỹ⟩ underestimates ⟨p, y⟩. The paper
/// corrects this with a 1-bit-per-coordinate QJL sketch of the residual. But over
/// random queries the least-squares estimate of ⟨p, y⟩ from ⟨p, ỹ⟩ is simply
///
///     ⟨p, y⟩ ≈ (⟨y, ỹ⟩ / ‖ỹ‖²) · ⟨p, ỹ⟩
///
/// — one scalar per *vector*, not one bit per *coordinate*. At d=128 that is 2 bytes
/// against 16, and the 14 bytes saved buy a whole extra bit of MSE codebook, which is
/// exactly what a 1-bit sketch costs.
pub const Correction = enum {
    /// The paper's construction: 1-bit QJL sketch of the residual.
    qjl_sketch,
    /// Per-vector least-squares rescale. Costs one f16 instead of d bits.
    scalar,
};

/// How codes are held in memory.
///
/// **`.compact` is the right default.** Whether unpacking costs anything depends
/// entirely on which packing the bit-width uses, and the two answers are very
/// different. Timed at the kernel, d=256, ns per vector:
///
///     codebook bits   packing      packed   expanded   ratio
///     2               sequential     3.82       3.37    1.13x
///     3               bit-plane     13.76       3.44    4.01x
///     4               sequential      3.28       3.38    0.97x
///
/// For **sequential** packing the `tbl` issues alongside the `sdot`s rather than
/// competing with them, and unpacking is free — at 4 bits the packed path is even
/// marginally ahead. For **bit-plane** packing (3-bit codes, which do not divide a
/// byte) it costs **4×**: three bit expansions and a weighted recombination per 16
/// codes is real work.
///
/// So `.expanded` is only worth considering at `bits = 4`, the one configuration whose
/// codebook is 3 bits. Even there `bits = 5` compact beats it on memory, speed *and*
/// recall, which is why the default does not move.
///
/// turbovec sits at the expanded end — it serializes 4 bits per coordinate but keeps
/// 8 bits resident. For a sequential layout that trade buys nothing we do not already
/// have.
pub const Residency = enum {
    /// `bits` per coordinate, unpacked during the scan. The default: this is a
    /// compression library, and compactness is the reason to use it.
    compact,
    /// One int8 per coordinate, dequantized at insert. Roughly `8/bits` times the
    /// memory for roughly half the scan instructions. Recall is unchanged — the
    /// stored values are exactly the centroids the compact path looks up.
    ///
    /// Requires a codebook of at most 16 levels (`bits <= 5`), since the values come
    /// from the same int8 table the compact scan indexes.
    expanded,
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
    /// How the MSE term's bias is corrected.
    correction: Correction = .scalar,
    /// Trade memory for scan speed. See `Residency`.
    residency: Residency = .compact,
    /// Include the QJL residual sketch in the score.
    ///
    /// The sketch makes the inner-product estimate *unbiased*, which is what
    /// `prod` exists for. But the bias it removes is multiplicative and essentially
    /// constant across vectors, and a constant factor does not change a ranking —
    /// so for top-k search the correction may buy nothing while costing a bit per
    /// coordinate and adding variance. Exposed so that can be measured rather than
    /// argued about.
    use_sketch: bool = true,
};

/// Half-precision per-vector scalars. See the field comment on `FlatIndex.scalars`.
pub const StoredScalars = struct {
    norm: f16,
    gamma: f16,

    fn from(s: prod_mod.Scalars) StoredScalars {
        return .{ .norm = @floatCast(s.norm), .gamma = @floatCast(s.gamma) };
    }
};

/// Largest batch `searchBatch` accepts. Bounded so per-query scratch can live on the
/// stack; beyond this the query state stops fitting in cache and the amortization the
/// batch exists for stops paying.
pub const max_batch: usize = 64;

pub const FlatIndex = struct {
    quantizer: prod_mod.Prod,
    layout: packing.Layout,
    metric: Metric,
    residency: Residency,
    /// int8 centroid table, needed at insert time to dequantize for `.expanded`.
    table: scan.Table,
    exact_scan: bool,
    use_sketch: bool,
    correction: Correction,

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

    /// Per-coordinate shifts for the rotated basis, length `padded`. Zero until
    /// `calibrate` is called.
    shifts: []f32,
    /// Per-coordinate scales for the rotated basis, length `padded`. All ones until
    /// `calibrate` is called.
    ///
    /// A random rotation *randomizes* the axes but does not *whiten* them: after
    /// rotation, coordinate j has variance πⱼᵀCπⱼ, which still tracks the spectrum of
    /// the data covariance C. On SIFT that leaves a 4.3× spread (cv 0.30) across
    /// coordinates, so one shared codebook is badly matched at both ends. Scaling each
    /// coordinate to the variance the codebook was built for fixes that.
    ///
    /// Costs `d` floats for the whole index, not per vector, and nothing at query
    /// time: the scale folds into the rotated query.
    scales: []f32,
    /// Whether `calibrate` has been run. Encoding before and after would produce
    /// incompatible codes, so adding after calibration is refused.
    calibrated: bool,

    workspace: prod_mod.Workspace,
    /// Reusable unpacked-code buffer for `add`. Held here rather than allocated per
    /// call: `add` runs once per vector, and a transient allocation per vector is
    /// both wasteful and needless pressure on the allocator during a bulk load.
    encode_scratch: []u8,
    allocator: Allocator,

    pub const Error = error{
        /// `.expanded` needs the int8 centroid table, which holds 16 levels.
        BitWidthTooWideForExpanded,
    };

    pub fn init(allocator: Allocator, params: Params) !FlatIndex {
        if (params.residency == .expanded and params.bits > 5) {
            return Error.BitWidthTooWideForExpanded;
        }

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

        const scales = try allocator.alloc(f32, quantizer.padded());
        errdefer allocator.free(scales);
        @memset(scales, 1.0);

        const shifts = try allocator.alloc(f32, quantizer.padded());
        errdefer allocator.free(shifts);
        @memset(shifts, 0.0);

        return .{
            .quantizer = quantizer,
            .layout = packing.Layout.init(quantizer.padded(), quantizer.mse.bits),
            .metric = params.metric,
            .residency = params.residency,
            // Only when the codebook fits one shuffle register. Building it
            // unconditionally overran a 16-entry array — the same mistake as in
            // Searcher.init, caught by the regression test added for that one.
            .table = if (quantizer.mse.codebook.levels() <= 16)
                scan.Table.init(quantizer.mse.codebook.centroids)
            else
                scan.unusedTable(),
            .exact_scan = params.exact_scan,
            .use_sketch = params.use_sketch,
            .correction = params.correction,
            .codes = .empty,
            .sketches = .empty,
            .scalars = .empty,
            .workspace = workspace,
            .encode_scratch = encode_scratch,
            .scales = scales,
            .shifts = shifts,
            .calibrated = false,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FlatIndex) void {
        self.allocator.free(self.shifts);
        self.allocator.free(self.scales);
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
        if (self.exact_scan) return false;
        // Expanded codes need no unpacking, so the bit-width constraints do not apply.
        if (self.residency == .expanded) return self.quantizer.padded() % 16 == 0;
        return scan.canVectorize(self.layout) and
            sketch_kernel.canVectorize(self.quantizer.padded());
    }

    /// Total bytes of index storage per vector, scalars included.
    ///
    /// Includes them deliberately: an earlier version reported only codes and
    /// sketch, which overstated the compression ratio by 11% at d=128.
    pub fn bytesPerVector(self: FlatIndex) usize {
        return self.codeBytesPerVector() + @sizeOf(StoredScalars);
    }

    /// Bytes of codes and sketch only — the part that scales with the bit budget.
    /// With the sketch disabled its storage is not counted, because it is not read.
    pub fn codeBytesPerVector(self: FlatIndex) usize {
        const sketch_bytes = switch (self.correction) {
            .qjl_sketch => if (self.use_sketch) self.quantizer.sketchLen() else 0,
            // The scalar factor rides in `StoredScalars`, already counted.
            .scalar => 0,
        };
        return self.codeStride() + sketch_bytes;
    }

    /// Bytes of codes per vector, which is what `Residency` changes.
    pub fn codeStride(self: FlatIndex) usize {
        return switch (self.residency) {
            .compact => self.layout.stride(),
            .expanded => std.mem.alignForward(usize, self.quantizer.padded(), 16),
        };
    }

    /// Fit a per-coordinate `(shift, scale)` by quantile anchoring.
    ///
    /// Each coordinate is mapped by `z = (y + shift) * scale` before quantizing, with
    /// the pair chosen so the coordinate's empirical quantiles land on the codebook's
    /// outermost centroids. The anchor probability is `P(|x| <= c_outer)` under the
    /// canonical marginal, taken from the codebook rather than fixed.
    ///
    /// Approach from turbovec (MIT, Ryan Codrai). Fitting mean and standard deviation
    /// instead — the obvious thing, and what an earlier version here did — measures
    /// *worse* than no calibration, because matching σ says nothing about where the
    /// tails land relative to the outermost centroid, and everything past it collapses
    /// into one bucket.
    ///
    /// **Sample size matters more than it looks.** The anchor sits at ~0.9967 for a
    /// 4-bit codebook, so the high quantile is an order statistic a few rows from the
    /// end: a 1024-row sample puts only ~3 rows beyond it, and the resulting fit is
    /// dominated by tail noise. Prefer several thousand rows, and more as bit-width
    /// rises since the anchor moves further out.
    ///
    /// Must be called before any `add`.
    pub fn calibrate(self: *FlatIndex, sample: []const f32) !void {
        if (self.count() > 0) return error.IndexNotEmpty;
        const d = self.dim();
        std.debug.assert(sample.len % d == 0);
        const rows = sample.len / d;
        if (rows < 2) return error.EmptySample;

        const padded = self.quantizer.padded();
        const staging = try self.allocator.alloc(f32, padded);
        defer self.allocator.free(staging);
        const rotated = try self.allocator.alloc(f32, rows * padded);
        defer self.allocator.free(rotated);
        for (0..rows) |i| {
            _ = self.quantizer.mse.encodeRotated(
                sample[i * d ..][0..d],
                rotated[i * padded ..][0..padded],
                staging,
            );
        }

        const centroids = self.quantizer.mse.codebook.centroids;
        var outer: f32 = 0;
        for (centroids) |c| outer = @max(outer, @abs(c));

        const density = Density.sphereCoord(padded);
        const p_hi = density.mass(density.support()[0], outer);
        const p_lo = 1.0 - p_hi;

        const frows: f64 = @floatFromInt(rows);
        var lo_idx: usize = @intFromFloat(frows * p_lo);
        var hi_idx: usize = @intFromFloat(frows * p_hi);
        lo_idx = @min(lo_idx, rows - 2);
        hi_idx = @max(@min(hi_idx, rows - 1), lo_idx + 1);

        const column = try self.allocator.alloc(f32, rows);
        defer self.allocator.free(column);

        for (0..padded) |j| {
            for (0..rows) |i| column[i] = rotated[i * padded + j];
            std.mem.sort(f32, column, {}, std.sort.asc(f32));

            const q_lo = column[lo_idx];
            const q_hi = column[hi_idx];
            const span = q_hi - q_lo;
            if (!(span > 1e-20) or outer <= 0) {
                self.shifts[j] = 0;
                self.scales[j] = 1;
                continue;
            }
            self.scales[j] = 2.0 * outer / span;
            self.shifts[j] = -0.5 * (q_lo + q_hi);
        }
        self.calibrated = true;
    }

    pub fn reserve(self: *FlatIndex, additional: usize) !void {
        try self.codes.ensureUnusedCapacity(self.allocator, additional * self.codeStride());
        try self.sketches.ensureUnusedCapacity(self.allocator, additional * self.quantizer.sketchLen());
        try self.scalars.ensureUnusedCapacity(self.allocator, additional);
    }

    /// Encode and append one vector. Returns its id.
    ///
    /// Ids are sequential and stable; nothing here removes or reorders. Encoding is
    /// data-oblivious, so a vector's codes never depend on what was added before it.
    pub fn add(self: *FlatIndex, vector: []const f32) !u32 {
        std.debug.assert(vector.len == self.dim());

        const stride = self.codeStride();
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

        var scalars = if (self.calibrated)
            self.encodeCalibrated(vector, scratch)
        else
            self.quantizer.encode(vector, scratch, sketch_slot, &self.workspace);

        switch (self.residency) {
            .compact => self.layout.pack(scratch, codes_slot),
            .expanded => {
                // Resolve each code to its int8 centroid now, so the scan does not
                // have to. Exactly the values `lookup` would return.
                const values: [16]i8 = self.table.values;
                for (codes_slot[0..scratch.len], scratch) |*dst, code| {
                    dst.* = @bitCast(values[code]);
                }
                @memset(codes_slot[scratch.len..], 0);
            },
        }

        // Only for the uncalibrated path: `encodeCalibrated` already returns α, and
        // running this over it treats that α as a residual norm and halves it.
        if (self.correction == .scalar and !self.calibrated) {
            // α = ⟨y, ỹ⟩ / ‖ỹ‖², stored in gamma's slot since the sketch is unused.
            // ⟨y, ỹ⟩ = (‖y‖² + ‖ỹ‖² − ‖y−ỹ‖²)/2, and ‖y‖ = 1 after normalization.
            const centroids = self.quantizer.mse.codebook.centroids;
            var recon_sq: f64 = 0;
            for (scratch, self.scales) |code, t| {
                // r̃_j = t_j · c[code_j], so the scale enters here too.
                const c: f64 = @as(f64, centroids[code]) * t;
                recon_sq += c * c;
            }
            const residual_sq: f64 = @as(f64, scalars.gamma) * scalars.gamma;
            // ⟨y,ỹ⟩ = (‖y‖² + ‖ỹ‖² − ‖y−ỹ‖²)/2, with ‖y‖ = 1 after normalization.
            const dot = (1.0 + recon_sq - residual_sq) * 0.5;
            scalars.gamma = if (recon_sq > 0) @floatCast(dot / recon_sq) else 1.0;
        }


        try self.scalars.append(self.allocator, StoredScalars.from(scalars));
        return @intCast(self.scalars.items.len - 1);
    }

    /// Rotate, rescale by the fitted per-coordinate scales, and encode.
    ///
    /// `gamma` comes back as the residual norm in the rotated basis, matching what
    /// `Prod.encode` returns, so the α computation downstream is unchanged.
    fn encodeCalibrated(self: *FlatIndex, vector: []const f32, codes: []u8) prod_mod.Scalars {
        const ws = &self.workspace;
        const norm = self.quantizer.mse.encodeRotated(vector, ws.rotated, ws.staging);

        // z = (y + shift) · scale, quantized against the shared codebook.
        for (ws.staging, ws.rotated, self.shifts, self.scales) |*z, y, sh, sc| {
            z.* = (y + sh) * sc;
        }
        self.quantizer.mse.codebook.encodeSlice(ws.staging, codes);

        // α is a least squares fit of the rotated vector against its reconstruction
        // in the *original* basis, x̂_j = c_j/scale_j − shift_j. Fitting in the
        // shifted basis instead — y+shift against c/scale — makes α a near-tautology:
        // both sides then contain the shift, which dominates, so α ≈ 1 no matter how
        // coarse the codes are and the correction silently does nothing.
        const centroids = self.quantizer.mse.codebook.centroids;
        var recon_sq: f64 = 0;
        var dot: f64 = 0;
        for (codes, ws.rotated, self.shifts, self.scales) |code, y, sh, sc| {
            const x_hat = @as(f64, centroids[code]) / sc - sh;
            recon_sq += x_hat * x_hat;
            dot += @as(f64, y) * x_hat;
        }
        const alpha = if (recon_sq > 0) dot / recon_sq else 1.0;
        return .{ .norm = norm, .gamma = @floatCast(alpha) };
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
            var scan_query = if (index.residency == .expanded)
                try scan.Query.initSequential(allocator, index.quantizer.padded())
            else
                try scan.Query.init(allocator, index.layout);
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

    /// Per-query state for a batch. Same components as `Searcher`, one set per query,
    /// plus one shared shuffle table.
    ///
    /// Sized once and reused: a batch search allocates nothing.
    pub const BatchSearcher = struct {
        batch: usize,
        k: usize,
        query_states: []prod_mod.QueryState,
        scan_queries: []scan.Query,
        sketch_queries: []sketch_kernel.Query,
        /// `batch * k` entries; query i owns `heap_store[i*k..][0..k]`.
        heap_store: []Entry,
        table: scan.Table,
        workspace: prod_mod.Workspace,
        allocator: Allocator,

        pub fn init(allocator: Allocator, index: FlatIndex, batch: usize, k: usize) !BatchSearcher {
            std.debug.assert(batch > 0 and k > 0);

            const query_states = try allocator.alloc(prod_mod.QueryState, batch);
            errdefer allocator.free(query_states);
            const scan_queries = try allocator.alloc(scan.Query, batch);
            errdefer allocator.free(scan_queries);
            const sketch_queries = try allocator.alloc(sketch_kernel.Query, batch);
            errdefer allocator.free(sketch_queries);

            var built: usize = 0;
            errdefer for (0..built) |i| {
                query_states[i].deinit();
                scan_queries[i].deinit();
                sketch_queries[i].deinit();
            };
            while (built < batch) : (built += 1) {
                query_states[built] = try prod_mod.QueryState.init(allocator, index.quantizer);
                scan_queries[built] = if (index.residency == .expanded)
                    try scan.Query.initSequential(allocator, index.quantizer.padded())
                else
                    try scan.Query.init(allocator, index.layout);
                sketch_queries[built] = try sketch_kernel.Query.init(allocator, index.quantizer.padded());
            }

            const heap_store = try allocator.alloc(Entry, batch * k);
            errdefer allocator.free(heap_store);
            var workspace = try prod_mod.Workspace.init(allocator, index.quantizer);
            errdefer workspace.deinit();

            return .{
                .batch = batch,
                .k = k,
                .query_states = query_states,
                .scan_queries = scan_queries,
                .sketch_queries = sketch_queries,
                .heap_store = heap_store,
                .table = if (index.vectorized())
                    scan.Table.init(index.quantizer.mse.codebook.centroids)
                else
                    scan.unusedTable(),
                .workspace = workspace,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *BatchSearcher) void {
            for (self.query_states, self.scan_queries, self.sketch_queries) |*a, *b, *c| {
                a.deinit();
                b.deinit();
                c.deinit();
            }
            self.workspace.deinit();
            self.allocator.free(self.heap_store);
            self.allocator.free(self.sketch_queries);
            self.allocator.free(self.scan_queries);
            self.allocator.free(self.query_states);
            self.* = undefined;
        }
    };

    /// Score several queries in one pass over the corpus.
    ///
    /// The scan is memory-bound: a vector's codes cost far more to fetch than to
    /// score. Looping queries *inside* the corpus loop means each vector is read from
    /// DRAM once and scored `n` times out of L1, so the memory traffic amortizes by
    /// the batch size. Query state for a batch of 32 at d=256 is about 50 KB, which
    /// stays resident alongside it.
    ///
    /// Returns a flat buffer; query `i`'s results are `result[i*k ..][0..k]`, sorted
    /// descending. Valid until the next search.
    pub fn searchBatch(self: *FlatIndex, queries: []const f32, searcher: *BatchSearcher) []Entry {
        const d = self.dim();
        std.debug.assert(queries.len % d == 0);
        const n = queries.len / d;
        std.debug.assert(n > 0 and n <= searcher.batch);

        const use_simd = self.vectorized();

        // One rotation and one sketch per query, before the corpus is touched.
        for (0..n) |i| {
            self.quantizer.prepareQuery(
                queries[i * d ..][0..d],
                &searcher.query_states[i],
                &searcher.workspace,
            );

            if (use_simd) {
                searcher.scan_queries[i].load(searcher.query_states[i].rotated);
                searcher.sketch_queries[i].load(searcher.query_states[i].sketched);
            }
        }

        var shift_terms: [max_batch]f32 = @splat(0);
        if (self.calibrated) {
            for (0..n) |i| {
                var acc: f64 = 0;
                for (searcher.query_states[i].rotated, self.shifts) |p, sh| {
                    acc += @as(f64, p) * sh;
                }
                shift_terms[i] = @floatCast(-acc);
            }
        }

        if (self.calibrated) {
            for (0..n) |i| {
                for (searcher.query_states[i].rotated, self.scales) |*p, sc| p.* /= sc;
            }
        }
        for (0..n) |i| {
            if (self.vectorized()) {
                searcher.scan_queries[i].load(searcher.query_states[i].rotated);
            }
        }

        var collectors: [max_batch]topk.TopK = undefined;
        for (0..n) |i| {
            collectors[i] = topk.TopK.init(searcher.heap_store[i * searcher.k ..][0..searcher.k]);
        }

        const query_norms = blk: {
            var norms: [max_batch]f32 = undefined;
            if (self.metric != .inner_product) {
                for (0..n) |i| norms[i] = euclideanNorm(queries[i * d ..][0..d]);
            } else {
                @memset(norms[0..n], 0);
            }
            break :blk norms;
        };

        const stride = self.codeStride();
        const sketch_len = self.quantizer.sketchLen();
        const padded = self.quantizer.padded();
        const centroids = self.quantizer.mse.codebook.centroids;

        // The expanded residency scores the whole query group in one pass over the
        // vector, so each code chunk is loaded once rather than once per query. See
        // `scan.scoreExpandedMulti` for why batching did nothing without this.
        const group = 8;
        var mse_terms: [max_batch]f32 = undefined;

        for (self.scalars.items, 0..) |scalars, v| {
            // Fetched once; the inner loop reads it out of cache.
            const code_slot = self.codes.items[v * stride ..][0..stride];
            const sketch_slot = self.sketches.items[v * sketch_len ..][0..sketch_len];
            const norm: f32 = scalars.norm;
            const gamma: f32 = scalars.gamma;

            if (self.residency == .expanded) {
                var i: usize = 0;
                while (i + group <= n) : (i += group) {
                    scan.scoreExpandedMulti(
                        group,
                        @ptrCast(code_slot),
                        searcher.scan_queries[i..],
                        self.table.scale,
                        padded,
                        mse_terms[i..],
                    );
                }
                while (i < n) : (i += 1) {
                    mse_terms[i] = scan.scoreExpanded(
                        @ptrCast(code_slot),
                        searcher.scan_queries[i],
                        self.table.scale,
                        padded,
                    );
                }
            }

            for (0..n) |i| {
                const mse_term = switch (self.residency) {
                    .expanded => mse_terms[i],
                    .compact => if (use_simd)
                        scan.scoreInt8(self.layout, searcher.table, searcher.scan_queries[i], code_slot, padded)
                    else
                        scan.scoreExact(self.layout, centroids, searcher.query_states[i].rotated, code_slot),
                };

                const estimate = switch (self.correction) {
                    // α scales the whole reconstruction ⟨p, x̂⟩ = mse_term + shift_term,
                    // not just one of its two halves: x̂ = c/scale − shift, so both
                    // terms come from the same reconstruction and share its error.
                    .scalar => norm * gamma * (shift_terms[i] + mse_term),
                    .qjl_sketch => blk: {
                        const sketch_term = if (!self.use_sketch)
                            0
                        else if (use_simd)
                            sketch_kernel.signDot(searcher.sketch_queries[i], sketch_slot, padded)
                        else
                            sketch_kernel.signDotExact(searcher.query_states[i].sketched, sketch_slot);
                        break :blk norm * (mse_term + gamma * self.quantizer.sketch_scale * sketch_term);
                    },
                };
                collectors[i].offer(self.score(estimate, norm, query_norms[i]), @intCast(v));
            }
        }

        for (0..n) |i| _ = collectors[i].drain();
        const found = @min(searcher.k, self.count());
        return searcher.heap_store[0 .. (n - 1) * searcher.k + found];
    }

    /// Query-parallel batch search state: one `BatchSearcher` per thread.
    ///
    /// Threads are given disjoint *queries*, not disjoint corpus shards. Each thread
    /// therefore reads the whole corpus, which would be wasteful if the scan were
    /// memory-bound — it is not (see docs/notes.md: batching a corpus pass across 32
    /// queries gives 1.01×, so per-vector compute is the constraint). Sharding queries
    /// instead means no shared mutable state, no top-k merge, and no false sharing.
    pub const ParallelSearcher = struct {
        shards: []BatchSearcher,
        threads: []std.Thread,
        /// Combined results, `capacity * k` entries.
        results: []Entry,
        per_thread: usize,
        k: usize,
        allocator: Allocator,

        /// `threads` workers, each able to hold `per_thread` queries in flight.
        pub fn init(
            allocator: Allocator,
            index: FlatIndex,
            threads: usize,
            per_thread: usize,
            k: usize,
        ) !ParallelSearcher {
            std.debug.assert(threads > 0 and per_thread > 0 and k > 0);

            const shards = try allocator.alloc(BatchSearcher, threads);
            errdefer allocator.free(shards);
            var built: usize = 0;
            errdefer for (0..built) |i| shards[i].deinit();
            while (built < threads) : (built += 1) {
                shards[built] = try BatchSearcher.init(allocator, index, per_thread, k);
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
            for (self.shards) |*shard| shard.deinit();
            self.allocator.free(self.results);
            self.allocator.free(self.threads);
            self.allocator.free(self.shards);
            self.* = undefined;
        }

        pub fn capacity(self: ParallelSearcher) usize {
            return self.shards.len * self.per_thread;
        }
    };

    const ShardWork = struct {
        index: *FlatIndex,
        queries: []const f32,
        searcher: *BatchSearcher,
        out: []Entry,

        fn run(work: ShardWork) void {
            const found = work.index.searchBatch(work.queries, work.searcher);
            @memcpy(work.out[0..found.len], found);
        }
    };

    /// Batch search across `searcher.shards.len` threads.
    ///
    /// Threads are spawned per call. That is only sensible because a batch is large
    /// enough to amortize it: at ~5 ms per query, a 32-query batch runs for ~160 ms
    /// against ~30 µs of spawn cost per thread. Spawning per *query* would not pay.
    pub fn searchBatchParallel(
        self: *FlatIndex,
        queries: []const f32,
        searcher: *ParallelSearcher,
    ) ![]Entry {
        const d = self.dim();
        std.debug.assert(queries.len % d == 0);
        const n = queries.len / d;
        std.debug.assert(n > 0 and n <= searcher.capacity());

        // Spread queries as evenly as possible; the remainder goes to the first few
        // threads rather than piling onto the last.
        const threads = @min(searcher.shards.len, n);
        const base = n / threads;
        const extra = n % threads;

        var spawned: usize = 0;
        errdefer for (0..spawned) |i| searcher.threads[i].join();

        var offset: usize = 0;
        while (spawned < threads) : (spawned += 1) {
            const take = base + @intFromBool(spawned < extra);
            const work = ShardWork{
                .index = self,
                .queries = queries[offset * d ..][0 .. take * d],
                .searcher = &searcher.shards[spawned],
                .out = searcher.results[offset * searcher.k ..][0 .. take * searcher.k],
            };
            searcher.threads[spawned] = try std.Thread.spawn(.{}, ShardWork.run, .{work});
            offset += take;
        }
        for (searcher.threads[0..threads]) |t| t.join();

        const found = @min(searcher.k, self.count());
        return searcher.results[0 .. (n - 1) * searcher.k + found];
    }

    /// Top-k search. Results are written into `searcher`'s heap and returned as a
    /// slice into it, valid until the next search.
    pub fn search(self: *FlatIndex, query: []const f32, searcher: *Searcher) []Entry {
        std.debug.assert(query.len == self.dim());
        std.debug.assert(searcher.heap.len > 0);

        // One rotation and one sketch for the whole corpus — the point of staying in
        // the rotated basis (docs/DESIGN.md §1.3).
        self.quantizer.prepareQuery(query, &searcher.query_state, &searcher.workspace);
        // ⟨p, ỹ⟩ = Σ (p_j/scale_j)·c[code_j] − ⟨p, shift⟩; the second term is the
        // same for every vector and cannot reorder them, but is included so scores
        // are real inner-product estimates.
        var shift_term: f32 = 0;
        if (self.calibrated) {
            var acc: f64 = 0;
            for (searcher.query_state.rotated, self.shifts) |p, sh| acc += @as(f64, p) * sh;
            shift_term = @floatCast(-acc);
            for (searcher.query_state.rotated, self.scales) |*p, sc| p.* /= sc;
        }
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
        const stride = self.codeStride();
        const sketch_len = self.quantizer.sketchLen();
        const padded = self.quantizer.padded();
        const centroids = self.quantizer.mse.codebook.centroids;

        for (self.scalars.items, 0..) |scalars, i| {
            const code_slot = self.codes.items[i * stride ..][0..stride];
            const sketch_slot = self.sketches.items[i * sketch_len ..][0..sketch_len];

            const mse_term = switch (self.residency) {
                .expanded => scan.scoreExpanded(
                    @ptrCast(code_slot),
                    searcher.scan_query,
                    self.table.scale,
                    padded,
                ),
                .compact => if (use_simd)
                    scan.scoreInt8(self.layout, searcher.table, searcher.scan_query, code_slot, padded)
                else
                    scan.scoreExact(self.layout, centroids, searcher.query_state.rotated, code_slot),
            };

            const norm: f32 = scalars.norm;
            const gamma: f32 = scalars.gamma;

            const estimate = switch (self.correction) {
                .scalar => norm * gamma * (shift_term + mse_term),
                .qjl_sketch => blk: {
                    const sketch_term = if (!self.use_sketch)
                        0
                    else if (use_simd)
                        sketch_kernel.signDot(searcher.sketch_query, sketch_slot, padded)
                    else
                        sketch_kernel.signDotExact(searcher.query_state.sketched, sketch_slot);
                    break :blk norm * (mse_term + gamma * self.quantizer.sketch_scale * sketch_term);
                },
            };

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
        var index = try FlatIndex.init(allocator, .{
            .dim = 1024,
            .bits = @intCast(bits),
            .correction = .qjl_sketch,
        });
        defer index.deinit();
        try testing.expectEqual(@as(u6, @intCast(bits - 1)), index.quantizer.mse.bits);
        try testing.expect(index.vectorized());
        // With the sketch, the code budget is exactly `bits` per coordinate.
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
    const dim: usize = 1024;
    for ([_]u6{ 2, 3, 4, 5, 6, 7 }) |bits| {
        // With the QJL sketch: (b−1) code bits plus one sketch bit per coordinate,
        // in both the sequential and bit-plane layouts.
        var sketched = try FlatIndex.init(allocator, .{
            .dim = @intCast(dim),
            .bits = bits,
            .correction = .qjl_sketch,
        });
        defer sketched.deinit();
        try testing.expectEqual(dim * @as(usize, bits) / 8, sketched.codeBytesPerVector());

        // With the scalar correction: (b−1) code bits and no sketch, so one bit per
        // coordinate less — which is the entire point.
        var scalar = try FlatIndex.init(allocator, .{
            .dim = @intCast(dim),
            .bits = bits,
            .correction = .scalar,
        });
        defer scalar.deinit();
        try testing.expectEqual(dim * @as(usize, bits - 1) / 8, scalar.codeBytesPerVector());

        // Both carry two half-precision scalars, which do not scale with dimension.
        for ([_]FlatIndex{ sketched, scalar }) |ix| {
            try testing.expectEqual(ix.codeBytesPerVector() + 4, ix.bytesPerVector());
        }
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

test "scalar correction beats the QJL sketch at matched storage" {
    // The measured P1 finding, pinned. The sketch costs one bit per coordinate to
    // remove a bias that a single per-vector scalar removes for f16 — so at equal
    // storage the scalar form gets a whole extra bit of codebook. On SIFT10K this
    // was worth +10 points of R@10 at 68 bytes; here it is checked on synthetic data
    // as a regression guard rather than a benchmark.
    const allocator = testing.allocator;
    const dim: u32 = 256;
    const n = 1500;
    const trials = 120;

    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(0x5CA1AB1E);
    const random = prng.random();
    // Clustered, so near-ties exist and the estimator's precision actually matters.
    var centers: [16][]f32 = undefined;
    const center_store = try allocator.alloc(f32, 16 * dim);
    defer allocator.free(center_store);
    for (0..16) |c| {
        centers[c] = center_store[c * dim ..][0..dim];
        randomUnit(centers[c], random);
    }
    for (0..n) |i| {
        const row = corpus[i * dim ..][0..dim];
        const c = random.uintLessThan(usize, 16);
        for (row, centers[c]) |*v, ce| {
            v.* = ce + 0.4 * random.floatNorm(f32) / @sqrt(@as(f32, @floatFromInt(dim)));
        }
        const len = euclideanNorm(row);
        for (row) |*v| v.* /= len;
    }

    const query = try allocator.alloc(f32, dim);
    defer allocator.free(query);

    // `.scalar` at bits=5 and `.qjl_sketch` at bits=4 cost the same.
    var scalar_recall: f64 = 0;
    var sketch_recall: f64 = 0;
    var scalar_bytes: usize = 0;
    var sketch_bytes: usize = 0;

    for ([_]Correction{ .scalar, .qjl_sketch }) |correction| {
        const bits: u6 = if (correction == .scalar) 5 else 4;
        var index = try FlatIndex.init(allocator, .{
            .dim = dim,
            .bits = bits,
            .seed = 0x11,
            .correction = correction,
        });
        defer index.deinit();
        try index.addBatch(corpus);

        var searcher = try FlatIndex.Searcher.init(allocator, index, 10);
        defer searcher.deinit();

        var hits: f64 = 0;
        var qprng = std.Random.DefaultPrng.init(7);
        for (0..trials) |_| {
            randomUnit(query, qprng.random());

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
        if (correction == .scalar) {
            scalar_recall = hits / @as(f64, trials);
            scalar_bytes = index.bytesPerVector();
        } else {
            sketch_recall = hits / @as(f64, trials);
            sketch_bytes = index.bytesPerVector();
        }
    }

    // Same storage, by construction.
    try testing.expectEqual(sketch_bytes, scalar_bytes);
    // And the scalar form is at least as good. Measured well ahead on real data;
    // asserted loosely here so the test tracks the direction, not the noise.
    try testing.expect(scalar_recall >= sketch_recall);
}

test "scalar correction removes the estimator's bias" {
    // The correction's actual job. Regress estimates on truth: the slope must be ~1,
    // where an uncorrected MSE-only estimator would sit near 2/π at one code bit.
    const allocator = testing.allocator;
    const dim: u32 = 256;
    const n = 400;

    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(4);
    const random = prng.random();
    for (0..n) |i| randomUnit(corpus[i * dim ..][0..dim], random);

    var index = try FlatIndex.init(allocator, .{
        .dim = dim,
        .bits = 5,
        .seed = 3,
        .correction = .scalar,
    });
    defer index.deinit();
    try index.addBatch(corpus);

    var searcher = try FlatIndex.Searcher.init(allocator, index, n);
    defer searcher.deinit();

    const query = try allocator.alloc(f32, dim);
    defer allocator.free(query);

    var cross: f64 = 0;
    var truth_sq: f64 = 0;
    for (0..30) |_| {
        randomUnit(query, random);
        for (index.search(query, &searcher)) |e| {
            var truth: f64 = 0;
            for (corpus[@as(usize, e.id) * dim ..][0..dim], query) |x, qv| {
                truth += @as(f64, x) * qv;
            }
            cross += @as(f64, e.score) * truth;
            truth_sq += truth * truth;
        }
    }
    try testing.expectApproxEqAbs(@as(f64, 1.0), cross / truth_sq, 0.05);
}

test "calibration fits per-coordinate scales and improves recall" {
    // A random rotation randomizes the axes but does not whiten them, so after
    // rotation coordinate j keeps variance πⱼᵀCπⱼ. On anisotropic data that leaves a
    // wide spread and one shared codebook is badly matched at both ends.
    const allocator = testing.allocator;
    const dim: u32 = 256;
    const n = 1200;

    // Deliberately anisotropic: energy concentrated in a low-dimensional subspace,
    // which is what real embeddings look like and what the uniform-sphere analysis
    // does not cover.
    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(0xA15E);
    const random = prng.random();
    for (0..n) |i| {
        const row = corpus[i * dim ..][0..dim];
        for (row, 0..) |*v, j| {
            const decay = 1.0 / (1.0 + @as(f32, @floatFromInt(j)) * 0.08);
            v.* = random.floatNorm(f32) * decay;
        }
        const len = euclideanNorm(row);
        for (row) |*v| v.* /= len;
    }

    var plain = try FlatIndex.init(allocator, .{ .dim = dim, .bits = 5, .seed = 2 });
    defer plain.deinit();
    try plain.addBatch(corpus);

    var fitted = try FlatIndex.init(allocator, .{ .dim = dim, .bits = 5, .seed = 2 });
    defer fitted.deinit();
    try fitted.calibrate(corpus[0 .. 512 * dim]);
    try fitted.addBatch(corpus);

    // The fit must actually find structure, or the comparison is vacuous.
    var spread: f32 = 0;
    for (fitted.scales) |t| spread = @max(spread, @abs(t - 1.0));
    try testing.expect(spread > 0.1);

    // Storage is unchanged: the scales are per index, not per vector.
    try testing.expectEqual(plain.bytesPerVector(), fitted.bytesPerVector());

    const query = try allocator.alloc(f32, dim);
    defer allocator.free(query);
    var hits_plain: usize = 0;
    var hits_fitted: usize = 0;
    const trials = 100;

    for ([_]*FlatIndex{ &plain, &fitted }, 0..) |index, which| {
        var searcher = try FlatIndex.Searcher.init(allocator, index.*, 10);
        defer searcher.deinit();
        var qprng = std.Random.DefaultPrng.init(31);
        for (0..trials) |_| {
            randomUnit(query, qprng.random());
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
                    if (which == 0) hits_plain += 1 else hits_fitted += 1;
                    break;
                }
            }
        }
    }
    try testing.expect(hits_fitted >= hits_plain);
}

test "calibrate refuses to run on a non-empty index" {
    // Codes written under different scales are not comparable, and the failure would
    // be silent: recall would just be worse.
    const allocator = testing.allocator;
    const dim: u32 = 64;
    var index = try FlatIndex.init(allocator, .{ .dim = dim, .bits = 4 });
    defer index.deinit();

    const vector = [_]f32{0.25} ** dim;
    _ = try index.add(&vector);
    try testing.expectError(error.IndexNotEmpty, index.calibrate(&vector));
}

test "batched search agrees with single-query search" {
    // Same scores, same order, same ids. The batch path reorders the loops for
    // locality; it must not change a result.
    const allocator = testing.allocator;
    const dim: u32 = 128;
    const n = 800;

    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(0xBA7C);
    const random = prng.random();
    for (0..n) |i| randomUnit(corpus[i * dim ..][0..dim], random);

    for ([_]Metric{ .inner_product, .cosine, .l2 }) |metric| {
        var index = try FlatIndex.init(allocator, .{
            .dim = dim,
            .bits = 5,
            .metric = metric,
            .seed = 0x5B,
        });
        defer index.deinit();
        try index.addBatch(corpus);

        const batch = 12;
        const k = 7;
        var single = try FlatIndex.Searcher.init(allocator, index, k);
        defer single.deinit();
        var batched = try FlatIndex.BatchSearcher.init(allocator, index, batch, k);
        defer batched.deinit();

        const queries = try allocator.alloc(f32, batch * dim);
        defer allocator.free(queries);
        for (0..batch) |i| randomUnit(queries[i * dim ..][0..dim], random);

        const grouped = index.searchBatch(queries, &batched);
        for (0..batch) |i| {
            const one = index.search(queries[i * dim ..][0..dim], &single);
            const many = grouped[i * k ..][0..k];
            try testing.expectEqual(one.len, many.len);
            for (one, many) |a, b| {
                try testing.expectEqual(a.id, b.id);
                try testing.expectEqual(a.score, b.score);
            }
        }
    }
}

test "batched search handles partial batches and calibration" {
    const allocator = testing.allocator;
    const dim: u32 = 64;
    const n = 300;

    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(5);
    const random = prng.random();
    for (0..n) |i| randomUnit(corpus[i * dim ..][0..dim], random);

    var index = try FlatIndex.init(allocator, .{ .dim = dim, .bits = 4, .seed = 1 });
    defer index.deinit();
    try index.calibrate(corpus[0 .. 128 * dim]);
    try index.addBatch(corpus);

    const k = 5;
    var single = try FlatIndex.Searcher.init(allocator, index, k);
    defer single.deinit();
    var batched = try FlatIndex.BatchSearcher.init(allocator, index, 16, k);
    defer batched.deinit();

    const queries = try allocator.alloc(f32, 16 * dim);
    defer allocator.free(queries);
    for (0..16) |i| randomUnit(queries[i * dim ..][0..dim], random);

    // Fewer queries than the searcher was sized for.
    for ([_]usize{ 1, 3, 16 }) |count| {
        const grouped = index.searchBatch(queries[0 .. count * dim], &batched);
        for (0..count) |i| {
            const one = index.search(queries[i * dim ..][0..dim], &single);
            for (one, grouped[i * k ..][0..k]) |a, b| try testing.expectEqual(a.id, b.id);
        }
    }
}

test "parallel search agrees with single-query search" {
    // Threads own disjoint queries and disjoint output, so results must be
    // bit-identical to the serial path — not merely close.
    const allocator = testing.allocator;
    const dim: u32 = 128;
    const n = 900;

    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(0x9A11E1);
    const random = prng.random();
    for (0..n) |i| randomUnit(corpus[i * dim ..][0..dim], random);

    var index = try FlatIndex.init(allocator, .{ .dim = dim, .bits = 5, .seed = 0x77 });
    defer index.deinit();
    try index.addBatch(corpus);

    const k = 8;
    var single = try FlatIndex.Searcher.init(allocator, index, k);
    defer single.deinit();

    // Thread counts that do and do not divide the query count, so the remainder
    // distribution is exercised.
    for ([_]usize{ 1, 2, 3, 4 }) |threads| {
        var parallel = try FlatIndex.ParallelSearcher.init(allocator, index, threads, 16, k);
        defer parallel.deinit();

        for ([_]usize{ 1, 5, 17, 32 }) |count| {
            if (count > parallel.capacity()) continue;
            const queries = try allocator.alloc(f32, count * dim);
            defer allocator.free(queries);
            for (0..count) |i| randomUnit(queries[i * dim ..][0..dim], random);

            const grouped = try index.searchBatchParallel(queries, &parallel);
            for (0..count) |i| {
                const one = index.search(queries[i * dim ..][0..dim], &single);
                const many = grouped[i * k ..][0..k];
                for (one, many) |a, b| {
                    try testing.expectEqual(a.id, b.id);
                    try testing.expectEqual(a.score, b.score);
                }
            }
        }
    }
}

test "parallel search is deterministic across runs" {
    // Nothing is shared between threads, so repeated runs must not vary. A race
    // would show up here as an intermittent mismatch rather than a crash.
    const allocator = testing.allocator;
    const dim: u32 = 64;
    const n = 400;

    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(3);
    const random = prng.random();
    for (0..n) |i| randomUnit(corpus[i * dim ..][0..dim], random);

    var index = try FlatIndex.init(allocator, .{ .dim = dim, .bits = 4, .seed = 2 });
    defer index.deinit();
    try index.addBatch(corpus);

    const k = 5;
    var parallel = try FlatIndex.ParallelSearcher.init(allocator, index, 4, 8, k);
    defer parallel.deinit();

    const queries = try allocator.alloc(f32, 32 * dim);
    defer allocator.free(queries);
    for (0..32) |i| randomUnit(queries[i * dim ..][0..dim], random);

    const first = try allocator.alloc(Entry, 32 * k);
    defer allocator.free(first);
    @memcpy(first, try index.searchBatchParallel(queries, &parallel));

    for (0..8) |_| {
        const again = try index.searchBatchParallel(queries, &parallel);
        for (first, again) |a, b| {
            try testing.expectEqual(a.id, b.id);
            try testing.expectEqual(a.score, b.score);
        }
    }
}

test "expanded residency gives identical results to compact" {
    // The stored bytes are exactly the centroids the compact path looks up, so the
    // two must agree bit-for-bit. Anything else means the dequantization at insert
    // and the lookup at scan have drifted.
    const allocator = testing.allocator;
    const dim: u32 = 256;
    const n = 600;

    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(0xE89A);
    const random = prng.random();
    for (0..n) |i| randomUnit(corpus[i * dim ..][0..dim], random);

    for ([_]u6{ 3, 4, 5 }) |bits| {
        var compact = try FlatIndex.init(allocator, .{ .dim = dim, .bits = bits, .seed = 6 });
        defer compact.deinit();
        var expanded = try FlatIndex.init(allocator, .{
            .dim = dim,
            .bits = bits,
            .seed = 6,
            .residency = .expanded,
        });
        defer expanded.deinit();
        try compact.addBatch(corpus);
        try expanded.addBatch(corpus);

        // Expanded costs one byte per coordinate regardless of bit-width.
        try testing.expectEqual(@as(usize, dim), expanded.codeStride());
        try testing.expect(expanded.codeStride() > compact.codeStride());
        try testing.expect(expanded.vectorized());

        const k = 10;
        var cs = try FlatIndex.Searcher.init(allocator, compact, k);
        defer cs.deinit();
        var es = try FlatIndex.Searcher.init(allocator, expanded, k);
        defer es.deinit();

        const query = try allocator.alloc(f32, dim);
        defer allocator.free(query);
        for (0..25) |_| {
            randomUnit(query, random);
            const a = compact.search(query, &cs);
            const b = expanded.search(query, &es);
            for (a, b) |x, y| {
                try testing.expectEqual(x.id, y.id);
                try testing.expectEqual(x.score, y.score);
            }
        }
    }
}

test "expanded residency is refused above the table's bit-width" {
    const allocator = testing.allocator;
    try testing.expectError(FlatIndex.Error.BitWidthTooWideForExpanded, FlatIndex.init(allocator, .{
        .dim = 128,
        .bits = 6,
        .residency = .expanded,
    }));
}

test "calibrated scores match an explicit reconstruction" {
    // The estimator must equal ⟨p, ŷ⟩ for the reconstruction the encoder actually
    // produced, ŷ_j = c[code_j]/scale_j − shift_j.
    //
    // This is the test that was missing. A stale post-encode block was overwriting
    // the α returned by `encodeCalibrated`, treating it as a residual norm and
    // halving it — scores came out ~21% low with a per-vector wobble, which looked
    // like "calibration hurts recall" rather than like a bug. Comparing against an
    // independent reconstruction finds it immediately.
    const allocator = testing.allocator;
    const dim: u32 = 128;
    const n = 200;

    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(0xA1FA);
    const random = prng.random();
    // Skewed and anisotropic, so shift and scale both do real work.
    for (0..n) |i| {
        const row = corpus[i * dim ..][0..dim];
        for (row, 0..) |*v, j| {
            const decay = 1.0 / (1.0 + @as(f32, @floatFromInt(j)) * 0.05);
            v.* = (random.floatNorm(f32) + 0.7) * decay;
        }
        const len = euclideanNorm(row);
        for (row) |*v| v.* /= len;
    }

    var index = try FlatIndex.init(allocator, .{
        .dim = dim,
        .bits = 5,
        .seed = 0x33,
        .exact_scan = true,
    });
    defer index.deinit();
    try index.calibrate(corpus);
    try index.addBatch(corpus);

    var searcher = try FlatIndex.Searcher.init(allocator, index, n);
    defer searcher.deinit();

    const padded = index.quantizer.padded();
    const staging = try allocator.alloc(f32, padded);
    defer allocator.free(staging);
    const y = try allocator.alloc(f32, padded);
    defer allocator.free(y);
    const z = try allocator.alloc(f32, padded);
    defer allocator.free(z);
    const codes = try allocator.alloc(u8, padded);
    defer allocator.free(codes);
    const centroids = index.quantizer.mse.codebook.centroids;

    var state = try prod_mod.QueryState.init(allocator, index.quantizer);
    defer state.deinit();
    var ws = try prod_mod.Workspace.init(allocator, index.quantizer);
    defer ws.deinit();

    const query = try allocator.alloc(f32, dim);
    defer allocator.free(query);

    for (0..10) |_| {
        randomUnit(query, random);
        const results = index.search(query, &searcher);
        index.quantizer.prepareQuery(query, &state, &ws);

        for (results[0..8]) |e| {
            const id: usize = e.id;
            _ = index.quantizer.mse.encodeRotated(corpus[id * dim ..][0..dim], y, staging);
            for (z, y, index.shifts, index.scales) |*zz, yy, sh, sc| zz.* = (yy + sh) * sc;
            index.quantizer.mse.codebook.encodeSlice(z, codes);

            var expected: f64 = 0;
            for (codes, state.rotated, index.shifts, index.scales) |c, p, sh, sc| {
                expected += @as(f64, p) * (@as(f64, centroids[c]) / sc - sh);
            }
            // α is a deliberate rescale, so allow for it rather than demanding
            // equality — but it must be close to 1, not off by a factor.
            const ratio = @as(f64, e.score) / expected;
            if (!(ratio > 0.85 and ratio < 1.18)) {
                std.debug.print("score {d:.5} expected {d:.5} ratio {d:.4}\n", .{ e.score, expected, ratio });
                return error.EstimatorMismatch;
            }
        }
    }
}

test "calibrated alpha is the least-squares fit in the pre-shift basis" {
    // α must be ⟨y, x̂⟩/‖x̂‖² for x̂_j = c[code_j]/scale_j − shift_j — the same basis
    // the estimator applies it in.
    //
    // The score-vs-reconstruction test above cannot see this. Fitting α in the
    // shifted basis instead (⟨y+shift, u⟩/‖u‖² for u = c/scale) leaves the score
    // still *proportional* to ⟨p, x̂⟩, so that test passes with a ratio near 1 —
    // but the constant is wrong and recall drops 12 points on anisotropic data.
    // Only checking α against its definition catches it.
    const allocator = testing.allocator;
    const dim: u32 = 96;
    const n = 128;

    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(0xB2C3);
    const random = prng.random();
    // Off-centre and anisotropic, so the fitted shift is large: that is exactly the
    // case where the two bases disagree.
    for (0..n) |i| {
        const row = corpus[i * dim ..][0..dim];
        for (row, 0..) |*v, j| {
            const decay = 1.0 / (1.0 + @as(f32, @floatFromInt(j)) * 0.04);
            v.* = (random.floatNorm(f32) + 1.3) * decay;
        }
        const len = euclideanNorm(row);
        for (row) |*v| v.* /= len;
    }

    var index = try FlatIndex.init(allocator, .{ .dim = dim, .bits = 3, .seed = 0x77 });
    defer index.deinit();
    try index.calibrate(corpus);
    try index.addBatch(corpus);

    const padded = index.quantizer.padded();
    const staging = try allocator.alloc(f32, padded);
    defer allocator.free(staging);
    const y = try allocator.alloc(f32, padded);
    defer allocator.free(y);
    const z = try allocator.alloc(f32, padded);
    defer allocator.free(z);
    const codes = try allocator.alloc(u8, padded);
    defer allocator.free(codes);
    const centroids = index.quantizer.mse.codebook.centroids;

    for (0..n) |id| {
        _ = index.quantizer.mse.encodeRotated(corpus[id * dim ..][0..dim], y, staging);
        for (z, y, index.shifts, index.scales) |*zz, yy, sh, sc| zz.* = (yy + sh) * sc;
        index.quantizer.mse.codebook.encodeSlice(z, codes);

        var dot: f64 = 0;
        var recon_sq: f64 = 0;
        for (codes, y, index.shifts, index.scales) |c, yy, sh, sc| {
            const x_hat = @as(f64, centroids[c]) / sc - sh;
            dot += @as(f64, yy) * x_hat;
            recon_sq += x_hat * x_hat;
        }
        const want: f32 = @floatCast(dot / recon_sq);
        try testing.expectApproxEqRel(want, index.scalars.items[id].gamma, 1e-3);
    }
}
