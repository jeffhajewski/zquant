//! Exhaustive-scan index over quantized codes.
//!
//! Owns a `Prod` quantizer and the packed corpus, and answers top-k queries by
//! scoring every vector. No partitioning — that is `index/ivf.zig` later. The point
//! of this layer is that the quantizer, packing, and both kernels compose into
//! something callable, and that the full estimator is used rather than half of it.
//!
//! ## The score
//!
//! Under the default `.scalar` correction, for a stored direction y = R·x/‖x‖ and the
//! rotated query p = R·q:
//!
//!     ⟨q, x⟩ ≈ ‖x‖ · [ f·⟨p, ũ⟩  +  ‖q‖·b  +  ⟨p, c⟩ ]
//!                       └ simd/scan ┘
//!
//! c is the corpus centre fitted by `calibrate` (zero without it), ũ the codebook
//! reconstruction of the unit residual (y − c)/‖y − c‖, and f and b per-vector scalars
//! (`encodeScalar`). Only the scan term is estimated; the centre's share is exact.
//!
//! Under `.qjl_sketch`, the paper's construction:
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
const tiles = @import("../simd/tiles.zig");
const dispatch = @import("../simd/dispatch.zig");
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
/// corrects this with a 1-bit-per-coordinate QJL sketch of the residual. One scalar
/// per *vector* does it instead:
///
///     ⟨p, y⟩ ≈ (‖y‖² / ⟨y, ỹ⟩) · ⟨p, ỹ⟩
///
/// At d=128 that is 2 bytes against 16, and the 14 bytes saved buy a whole extra bit
/// of MSE codebook, which is exactly what a 1-bit sketch costs.
///
/// **Why this scalar and not the least-squares one.** ⟨y, ỹ⟩/‖ỹ‖² is the best fit of
/// y from ỹ over random queries, and was used here first. The scalar above instead
/// makes the estimate exact when p points along y. A ranking is decided among the
/// vectors that point roughly where the query does, so that is the direction to be
/// exact in: at bits 2 and 3 it measured up to 7 points of R@10 better, and level on
/// nytimes-256 (docs/notes.md, "Partitioning and the low-bit gap"). It is also the unbiased one in the sense
/// the test "scalar correction removes the estimator's bias" checks: regressing the
/// estimate on the truth gives a slope of exactly 1. RaBitQ normalises the same way.
pub const Correction = enum {
    /// The paper's construction: 1-bit QJL sketch of the residual.
    qjl_sketch,
    /// Per-vector rescale. Costs one f16 instead of d bits.
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
    /// Under `.scalar`, the factor f on the scanned term. Under `.qjl_sketch`, the
    /// residual norm γ that weights the sketch.
    gamma: f16,

    fn from(s: prod_mod.Scalars) StoredScalars {
        return .{ .norm = @floatCast(s.norm), .gamma = @floatCast(s.gamma) };
    }
};

/// What `encodeScalar` produces for one vector.
const ScalarCode = struct {
    norm: f32,
    factor: f32,
    bias: f32,
};

/// Largest batch `searchBatch` accepts. Bounded so per-query scratch can live on the
/// stack; beyond this the query state stops fitting in cache and the amortization the
/// batch exists for stops paying.
pub const max_batch: usize = 128;

/// Lane count for the batch epilogue's early-reject compare. Once the heaps fill,
/// almost every candidate is rejected, so scoring a lane group and testing one mask
/// replaces `lanes` scalar compares and branches.
pub const lanes = 8;
pub const Lane = @Vector(lanes, f32);
pub const LaneMask = std.meta.Int(.unsigned, lanes);

/// Accepted candidates are staged here per query and flushed into the collector in
/// batches.
///
/// Accepts are rare but expensive — measured at ~55 ns each, far more than a heap sift
/// costs — because between two accepts for the same query the scan walks thousands of
/// vectors and evicts that query's heap. The cost therefore tracks the collectors'
/// total footprint, `batch * k`: 2.5 KB at k=10 but 51 KB at k=200, well past L1. That
/// is why retrieval depth costs throughput at all, and why a wider reservoir did not
/// help — it made the footprint larger, not smaller.
///
/// A staging buffer of fixed width does not grow with k. `max_batch * hot_slots` is
/// 8 KB whatever the depth, so it stays resident, and the cold heap is touched once
/// per flush rather than once per accept.
pub const hot_slots = 8;

pub const FlatIndex = struct {
    /// What the index was created with. Kept so it can be saved and recreated
    /// (`persist.zig`): the seed in particular is consumed by the quantizer's setup and
    /// not otherwise recoverable.
    params: Params,
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
    /// QJL sign bitmaps, `sketchStride()` bytes per vector — none at all unless the
    /// score reads them (see `storesSketch`).
    sketches: std.ArrayList(u8),
    /// ‖x‖ and the correction scalar per vector, at half precision.
    ///
    /// f16 rather than f32 because these are per *vector*, not per coordinate, so
    /// they do not shrink with the bit budget: 8 bytes of f32 is under 2% of a
    /// d=1024 vector but 11% of a d=128 one, which is exactly the KV-cache regime.
    /// f16 carries ~3 decimal digits, far more than needed for a scale factor whose
    /// own inputs are quantized to 4 bits.
    scalars: std.ArrayList(StoredScalars),
    /// The per-vector bias b of a calibrated index, at half precision; empty otherwise.
    /// Held apart from `scalars` so an uncalibrated index does not pay for it.
    biases: std.ArrayList(f16),

    /// The corpus centre c in the rotated basis, length `padded`: the mean of the
    /// sample's rotated directions. Zero until `calibrate` is called.
    ///
    /// Each vector is stored as c plus a quantized residual, and the centre's share of
    /// every score is computed exactly rather than estimated, so only the residual's
    /// share carries quantization error. When the corpus sits well off the origin, as
    /// real embeddings do, the residual is much shorter than the vector and the error
    /// shrinks with it.
    ///
    /// Costs `d` floats for the whole index, one dot product per query, and one f16
    /// per vector (`biases`).
    centre: []f32,
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

        const centre = try allocator.alloc(f32, quantizer.padded());
        errdefer allocator.free(centre);
        @memset(centre, 0.0);

        return .{
            .params = params,
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
            .biases = .empty,
            .workspace = workspace,
            .encode_scratch = encode_scratch,
            .centre = centre,
            .calibrated = false,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FlatIndex) void {
        self.allocator.free(self.centre);
        self.allocator.free(self.encode_scratch);
        self.codes.deinit(self.allocator);
        self.sketches.deinit(self.allocator);
        self.scalars.deinit(self.allocator);
        self.biases.deinit(self.allocator);
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
    /// sketch, which overstated the compression ratio by 11% at d=128. Calibrating
    /// adds two, for the per-vector bias.
    pub fn bytesPerVector(self: FlatIndex) usize {
        return self.codeBytesPerVector() + @sizeOf(StoredScalars) + self.biasStride();
    }

    /// Bytes of bias per vector: two once calibrated, otherwise none.
    pub fn biasStride(self: FlatIndex) usize {
        return if (self.calibrated) @sizeOf(f16) else 0;
    }

    /// Bytes of codes and sketch only — the part that scales with the bit budget.
    /// The scalar factor rides in `StoredScalars`, already counted by `bytesPerVector`.
    pub fn codeBytesPerVector(self: FlatIndex) usize {
        return self.codeStride() + self.sketchStride();
    }

    /// Whether the score reads the QJL sketch, and so whether it is computed and kept.
    ///
    /// Only `.qjl_sketch` with `use_sketch` does. Every vector used to get a sketch
    /// regardless, which under the default `.scalar` correction was a second rotation
    /// per insert and per query, and d/8 bytes per vector that nothing read — 164
    /// resident bytes against the 132 reported, at d=256 and bits=5.
    pub fn storesSketch(self: FlatIndex) bool {
        return self.correction == .qjl_sketch and self.use_sketch;
    }

    /// Bytes of stored sketch per vector: `sketchLen()` when kept, otherwise zero.
    pub fn sketchStride(self: FlatIndex) usize {
        return if (self.storesSketch()) self.quantizer.sketchLen() else 0;
    }

    /// Bytes of codes per vector, which is what `Residency` changes.
    pub fn codeStride(self: FlatIndex) usize {
        return switch (self.residency) {
            .compact => self.layout.stride(),
            .expanded => std.mem.alignForward(usize, self.quantizer.padded(), 16),
        };
    }

    /// Centroid norm at or above which `calibrateAuto` calibrates.
    ///
    /// Centring removes the part of every vector the corpus shares, so what it buys
    /// tracks how much is shared: the centroid norm. At 0.12 (nytimes-256) it gained
    /// about a point of R@10, which does not pay for its two bytes per vector; at 0.65
    /// and above (SIFT, fashion-mnist, Cohere, OpenAI) it gained 9 to 22 points at
    /// bits 2 and 3. Measured through the library; see docs/notes.md, "Partitioning and
    /// the low-bit gap".
    pub const auto_calibration_threshold: f32 = 0.3;

    pub const AutoCalibration = struct {
        /// `centroidNorm` of the sample.
        centroid_norm: f32,
        /// Whether the index was calibrated.
        applied: bool,
    };

    /// Calibrate if the sample says it will help: when its `centroidNorm` is at least
    /// `auto_calibration_threshold`. Fails exactly as `calibrate` would, whichever
    /// way the measurement goes, so a caller's error handling does not depend on data.
    pub fn calibrateAuto(self: *FlatIndex, sample: []const f32) !AutoCalibration {
        if (self.count() > 0) return error.IndexNotEmpty;
        if (self.correction != .scalar) return error.CalibrationNeedsScalarCorrection;
        const d = self.dim();
        std.debug.assert(sample.len % d == 0);
        if (!allFinite(sample)) return error.NonFiniteInput;
        if (sample.len / d < 2) return error.EmptySample;

        const c = try centroidNorm(self.allocator, sample, d);
        if (c < auto_calibration_threshold) return .{ .centroid_norm = c, .applied = false };
        try self.calibrate(sample);
        return .{ .centroid_norm = c, .applied = true };
    }

    /// Fit the corpus centre from a sample: the mean of its rotated directions.
    ///
    /// Afterwards each vector is encoded as the centre plus a quantized residual, and
    /// the centre's share of every score is computed exactly (see `centre`). Rows of
    /// zero have no direction and are skipped.
    ///
    /// This replaced a per-coordinate (shift, scale) fitted by quantile anchoring, an
    /// approach from turbovec. Centring with the exact centre term matched or beat it
    /// on every corpus measured, by up to 22 points of R@10 (fashion-mnist, bits=2), and
    /// the quantile fit on top of centring gained at most 2 points anywhere and lost up
    /// to 3 elsewhere. A mean also needs far fewer rows than a tail quantile did: 100
    /// rows came within a point of 4,096.
    ///
    /// Must be called before any `add`, and only with the `.scalar` correction.
    pub fn calibrate(self: *FlatIndex, sample: []const f32) !void {
        if (self.count() > 0) return error.IndexNotEmpty;
        // The sketch path has no centre term and stores a residual norm where the
        // scalar path stores its factor. Mixing them once silently cost a third of
        // recall, so the combination is refused rather than half-supported.
        if (self.correction != .scalar) return error.CalibrationNeedsScalarCorrection;
        const d = self.dim();
        std.debug.assert(sample.len % d == 0);
        // One NaN would poison the mean, and with it every score.
        if (!allFinite(sample)) return error.NonFiniteInput;
        const rows = sample.len / d;
        if (rows < 2) return error.EmptySample;

        const padded = self.quantizer.padded();
        const staging = try self.allocator.alloc(f32, padded);
        defer self.allocator.free(staging);
        const rotated = try self.allocator.alloc(f32, padded);
        defer self.allocator.free(rotated);
        const sum = try self.allocator.alloc(f64, padded);
        defer self.allocator.free(sum);
        @memset(sum, 0);

        var used: usize = 0;
        for (0..rows) |i| {
            const norm = self.quantizer.mse.encodeRotated(sample[i * d ..][0..d], rotated, staging);
            if (norm == 0) continue;
            for (sum, rotated) |*acc, y| acc.* += y;
            used += 1;
        }
        const scale: f64 = if (used > 0) 1 / @as(f64, @floatFromInt(used)) else 0;
        for (self.centre, sum) |*c, acc| c.* = @floatCast(acc * scale);
        self.calibrated = true;
    }

    pub fn reserve(self: *FlatIndex, additional: usize) !void {
        try self.codes.ensureUnusedCapacity(self.allocator, additional * self.codeStride());
        try self.sketches.ensureUnusedCapacity(self.allocator, additional * self.sketchStride());
        try self.scalars.ensureUnusedCapacity(self.allocator, additional);
        if (self.calibrated) try self.biases.ensureUnusedCapacity(self.allocator, additional);
    }

    /// Encode and append one vector. Returns its id.
    ///
    /// Ids are sequential and stable; nothing here removes or reorders. Encoding is
    /// data-oblivious, so a vector's codes never depend on what was added before it.
    pub fn add(self: *FlatIndex, vector: []const f32) !u32 {
        std.debug.assert(vector.len == self.dim());
        // A NaN or infinite coordinate encodes to a NaN norm and score, and a NaN in the
        // collector breaks its ordering for every query, not just for this vector.
        if (!allFinite(vector)) return error.NonFiniteInput;

        // Reserved first, so nothing after this point can fail half-way through.
        try self.reserve(1);
        const id = self.count();
        self.growAssumeCapacity(1);
        self.encodeAt(id, vector, self.encode_scratch, &self.workspace);
        return @intCast(id);
    }

    /// Grow storage by `n` zeroed vectors. Capacity must already be reserved.
    fn growAssumeCapacity(self: *FlatIndex, n: usize) void {
        @memset(self.codes.addManyAsSliceAssumeCapacity(n * self.codeStride()), 0);
        @memset(self.sketches.addManyAsSliceAssumeCapacity(n * self.sketchStride()), 0);
        _ = self.scalars.addManyAsSliceAssumeCapacity(n);
        if (self.calibrated) _ = self.biases.addManyAsSliceAssumeCapacity(n);
    }

    /// Encode `vector` into the already-allocated slot `id`.
    ///
    /// Reads only state that is fixed once vectors are being added (quantizer,
    /// calibration, layout) and writes only slot `id`, `scratch` and `ws`. So threads
    /// with their own scratch and workspace can fill disjoint slots concurrently, and
    /// a vector's bytes do not depend on which thread encoded it.
    fn encodeAt(self: *const FlatIndex, id: usize, vector: []const f32, scratch: []u8, ws: *prod_mod.Workspace) void {
        const stride = self.codeStride();
        const sketch_len = self.sketchStride();
        const codes_slot = self.codes.items[id * stride ..][0..stride];
        const sketch_slot = self.sketches.items[id * sketch_len ..][0..sketch_len];

        // Encode into scratch, then pack. The quantizer emits one byte per
        // coordinate; storage wants them bit-packed.
        switch (self.correction) {
            .scalar => {
                const code = self.encodeScalar(vector, scratch, ws);
                self.scalars.items[id] = .{ .norm = @floatCast(code.norm), .gamma = @floatCast(code.factor) };
                if (self.calibrated) self.biases.items[id] = @floatCast(code.bias);
            },
            .qjl_sketch => {
                const scalars = if (self.storesSketch())
                    self.quantizer.encode(vector, scratch, sketch_slot, ws)
                else
                    self.quantizer.encodeCodes(vector, scratch, ws);
                self.scalars.items[id] = StoredScalars.from(scalars);
            },
        }

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
    }

    /// Encode one vector for the `.scalar` correction.
    ///
    /// With y = R·x/‖x‖ and the centre c (zero unless calibrated), the residual
    /// r = y − c is quantized as a unit vector, u = r/‖r‖, giving codes whose
    /// reconstruction is ũ. Then
    ///
    ///     ⟨p, y⟩ = ⟨p, c⟩ + ⟨c, r⟩ + ⟨p − c, r⟩
    ///            ≈ ⟨p, c⟩ + ⟨c, r⟩ + f·⟨p − c, ũ⟩          f = ‖r‖² / ⟨r, ũ⟩
    ///            = f·⟨p, ũ⟩ + b + ⟨p, c⟩                   b = ⟨c, r⟩ − f·⟨c, ũ⟩
    ///
    /// The first two terms of the split are exact: one per query, one per vector. Only
    /// ⟨p − c, r⟩ is estimated, and f makes that estimate exact when p − c points
    /// along r. For a query of any length the split uses p/‖p‖, so the score scales
    /// b by ‖q‖ (`queryTerms`).
    ///
    /// Uncalibrated, c = 0 and b = 0, and y is quantized as it stands rather than
    /// renormalized, so the codes are exactly the quantizer's.
    fn encodeScalar(self: *const FlatIndex, vector: []const f32, codes: []u8, ws: *prod_mod.Workspace) ScalarCode {
        const norm = self.quantizer.mse.encodeRotated(vector, ws.rotated, ws.staging);
        const res = encodeResidual(&self.quantizer, if (self.calibrated) self.centre else null, codes, ws);
        return .{ .norm = norm, .factor = res.factor, .bias = res.bias };
    }

    /// Append `n` vectors from a row-major buffer.
    pub fn addBatch(self: *FlatIndex, vectors: []const f32) !void {
        const d = self.dim();
        std.debug.assert(vectors.len % d == 0);
        const n = vectors.len / d;
        // Checked up front so a bad row late in the batch does not leave the rows
        // before it added: a batch either goes in whole or not at all.
        if (!allFinite(vectors)) return error.NonFiniteInput;
        try self.reserve(n);
        const first = self.count();
        self.growAssumeCapacity(n);
        for (0..n) |i| self.encodeAt(first + i, vectors[i * d ..][0..d], self.encode_scratch, &self.workspace);
    }

    /// Below this many vectors per thread, spawning costs more than it saves: a thread
    /// takes ~30 µs to start and a vector a few µs to encode.
    const min_rows_per_thread = 256;

    /// `addBatch` on up to `threads` threads.
    ///
    /// The result is byte-for-byte what `addBatch` produces — same ids, codes and
    /// scalars — because each vector encodes independently into its own slot, so how
    /// the batch is split cannot show. Like `addBatch`, the batch goes in whole or
    /// not at all; a thread that fails to start has its share encoded on this one.
    pub fn addBatchParallel(self: *FlatIndex, vectors: []const f32, threads: usize) !void {
        const d = self.dim();
        std.debug.assert(vectors.len % d == 0);
        const n = vectors.len / d;
        const workers = @max(1, @min(threads, n / min_rows_per_thread));
        if (workers <= 1) return self.addBatch(vectors);
        if (!allFinite(vectors)) return error.NonFiniteInput;

        // Every allocation before the index changes, so a failure leaves it as it was.
        // Worker 0 is this thread and uses the index's own scratch and workspace.
        const extra = workers - 1;
        const scratch = try self.allocator.alloc(u8, extra * self.encode_scratch.len);
        defer self.allocator.free(scratch);
        const spaces = try self.allocator.alloc(prod_mod.Workspace, extra);
        defer self.allocator.free(spaces);
        var made: usize = 0;
        defer for (spaces[0..made]) |*ws| ws.deinit();
        while (made < extra) : (made += 1) spaces[made] = try prod_mod.Workspace.init(self.allocator, self.quantizer);
        const handles = try self.allocator.alloc(?std.Thread, extra);
        defer self.allocator.free(handles);
        try self.reserve(n);

        const first = self.count();
        self.growAssumeCapacity(n);

        const Chunk = struct {
            index: *const FlatIndex,
            vectors: []const f32,
            first: usize,
            scratch: []u8,
            ws: *prod_mod.Workspace,

            fn run(c: @This()) void {
                const dd = c.index.dim();
                const rows = c.vectors.len / dd;
                for (0..rows) |i| c.index.encodeAt(c.first + i, c.vectors[i * dd ..][0..dd], c.scratch, c.ws);
            }
        };
        const base = n / workers;
        const rem = n % workers;
        var start: usize = 0;
        var own: Chunk = undefined;
        for (0..workers) |w| {
            const take = base + @intFromBool(w < rem);
            const chunk = Chunk{
                .index = self,
                .vectors = vectors[start * d ..][0 .. take * d],
                .first = first + start,
                .scratch = if (w == 0) self.encode_scratch else scratch[(w - 1) * self.encode_scratch.len ..][0..self.encode_scratch.len],
                .ws = if (w == 0) &self.workspace else &spaces[w - 1],
            };
            start += take;
            if (w == 0) {
                // This thread's share runs once the others have started.
                own = chunk;
                continue;
            }
            handles[w - 1] = std.Thread.spawn(.{}, Chunk.run, .{chunk}) catch blk: {
                chunk.run();
                break :blk null;
            };
        }
        own.run();
        for (handles) |h| if (h) |t| t.join();
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

        // One rotation per query before the corpus is touched, plus a sketch rotation
        // only where the score reads it. The int8 scan query is loaded below, once any
        // calibration scaling has been applied.
        const sketched = self.storesSketch();
        for (0..n) |i| {
            const query = queries[i * d ..][0..d];
            const state = &searcher.query_states[i];
            if (sketched) {
                self.quantizer.prepareQuery(query, state, &searcher.workspace);
                if (use_simd) searcher.sketch_queries[i].load(state.sketched);
            } else {
                self.quantizer.prepareQueryRotated(query, state, &searcher.workspace);
            }
        }

        for (0..n) |i| {
            if (use_simd) searcher.scan_queries[i].load(searcher.query_states[i].rotated);
        }

        var collectors: [max_batch]topk.TopK = undefined;
        // Mirror of each collector's admission threshold. Reading it from the heap
        // costs three loads — `len`, `heap.ptr`, `heap.len` — for every (vector,
        // query) pair, and at d=256 the scan itself is only about ten loads per pair,
        // so the epilogue was a large fraction of the loop it followed. Almost every
        // candidate is rejected, so the cache is refreshed only on the rare accept.
        var thresholds: [max_batch]f32 = @splat(-std.math.inf(f32));
        var hot: [max_batch][hot_slots]Entry = undefined;
        var hot_len: [max_batch]usize = @splat(0);
        for (0..n) |i| {
            collectors[i] = topk.TopK.init(searcher.heap_store[i * searcher.k ..][0..searcher.k]);
        }

        const query_norms = blk: {
            var norms: [max_batch]f32 = undefined;
            if (self.metric != .inner_product or self.calibrated) {
                for (0..n) |i| norms[i] = euclideanNorm(queries[i * d ..][0..d]);
            } else {
                @memset(norms[0..n], 0);
            }
            break :blk norms;
        };
        // The per-query half of `scalarScore`, laid out as lanes for the epilogue.
        var query_factor: [max_batch]f32 = undefined;
        var query_offset: [max_batch]f32 = undefined;
        var query_bias: [max_batch]f32 = @splat(0);
        var query_centre: [max_batch]f32 = @splat(0);
        for (0..n) |i| {
            const terms = self.queryTermsFor(query_norms[i], searcher.query_states[i].rotated);
            query_factor[i] = terms.factor;
            query_offset[i] = terms.offset;
            query_bias[i] = terms.bias;
            query_centre[i] = terms.centre;
        }

        const stride = self.codeStride();
        const sketch_len = self.sketchStride();
        const padded = self.quantizer.padded();
        const centroids = self.quantizer.mse.codebook.centroids;

        // Vectors are tiled as well as queries: the group's query chunks stay in
        // registers across the tile, so the loop issues group + tile loads against
        // tile*group SDOTs. See `scan.scoreExpandedTiled` for the accounting. The
        // tiles go through `dispatch`, which may run a copy of the kernels compiled for
        // a wider instruction set than this build's (AVX2 in an x86_64_v2 release).
        const tile = tiles.batch_tile;
        // Everything the inner loop would otherwise re-test per (vector, query) pair:
        // all three are fixed for the whole index.
        //
        // Every metric qualifies: under the scalar correction each one is an affine map
        // of the scanned term with per-vector and per-query coefficients
        // (`scalarScore`), so cosine and L2 reject a lane group as cheaply as inner
        // product does. They used to take the per-pair path, and measured about half
        // inner product's throughput for it.
        const fast_path = self.correction == .scalar and
            (self.residency == .expanded or use_simd);
        const compact_width = if (use_simd) scan.multiWidth(self.layout) else 1;
        var mse_block: [tile * max_batch]f32 = undefined;

        const can_tile = self.residency == .expanded or (use_simd and scan.canTile(self.layout));
        const kernels = dispatch.active();
        const job: tiles.Job = .{
            .table = &searcher.table,
            .queries = searcher.scan_queries.ptr,
            .n = n,
            .codes = self.codes.items.ptr,
            .stride = stride,
            .padded = @intCast(padded),
            .bits = self.layout.bits,
            .expanded = self.residency == .expanded,
            .table_scale = self.table.scale,
            .out = &mse_block,
            .out_stride = max_batch,
        };

        const total = self.scalars.items.len;
        var v0: usize = 0;
        while (v0 < total) : (v0 += tile) {
            const take = @min(tile, total - v0);

            if (can_tile and take == tile) {
                // Both amortizations at once: shared unpack and register-resident
                // query chunks. Query tails are handled inside.
                dispatch.batchTile(kernels, &job, v0);
            } else if (self.residency == .expanded) {
                // The last, partial tile.
                for (0..n) |i| {
                    for (0..take) |u| {
                        mse_block[u * max_batch + i] = scan.scoreExpanded(
                            @ptrCast(self.codes.items[(v0 + u) * stride ..][0..stride]),
                            searcher.scan_queries[i],
                            self.table.scale,
                            padded,
                        );
                    }
                }
            } else if (use_simd) {
                // The unpack is query-independent, so the group shares it.
                for (0..take) |u| {
                    const slot = self.codes.items[(v0 + u) * stride ..][0..stride];
                    var i: usize = 0;
                    while (i + compact_width <= n) : (i += compact_width) {
                        scan.scoreInt8Multi(
                            self.layout,
                            searcher.table,
                            searcher.scan_queries[i..],
                            slot,
                            padded,
                            mse_block[u * max_batch + i ..],
                        );
                    }
                    while (i < n) : (i += 1) {
                        mse_block[u * max_batch + i] = scan.scoreInt8(
                            self.layout,
                            searcher.table,
                            searcher.scan_queries[i],
                            slot,
                            padded,
                        );
                    }
                }
            }

            // Until every collector holds k candidates the threshold is -inf, and a
            // -inf score — a zero vector under cosine — would be rejected by a strict
            // compare even though the heap has room, returning short results. Once
            // filled, the cached threshold is exactly `heap[0].score`, so the fast
            // reject is identical to the test inside `offer`.
            const filled = v0 >= searcher.k;

            for (0..take) |u| {
                const v = v0 + u;
                const scalars = self.scalars.items[v];
                const code_slot = self.codes.items[v * stride ..][0..stride];
                const sketch_slot = self.sketches.items[v * sketch_len ..][0..sketch_len];
                const norm: f32 = scalars.norm;
                const gamma: f32 = scalars.gamma;
                const bias: f32 = if (self.calibrated) self.biases.items[v] else 0;

                // Split on the correction outside the query loop rather than
                // switching per pair: it is fixed for the whole index.
                if (fast_path and filled) {
                    // Once the heaps are full almost every candidate is rejected, so
                    // the epilogue is a long run of compares that do nothing. Scoring
                    // and rejecting a whole lane group at a time turns `lanes` scalar
                    // compares and branches into one vector compare and one mask test,
                    // and the scalar path below is reached only on the rare accept.
                    //
                    // This is where the batch loop's remaining cost sits: measured by
                    // ablation, the scan kernel alone runs at 175 G dim/s against
                    // `searchBatch`'s 133, and the gap is the per-pair scalar work
                    // rather than the top-k structure.
                    const block = mse_block[u * max_batch ..][0..n];
                    const vt = vectorTerms(self.metric, norm, gamma, bias);
                    const av: Lane = @splat(vt.factor);
                    const bv: Lane = @splat(vt.offset);
                    const biv: Lane = @splat(vt.bias);
                    const cv: Lane = @splat(vt.centre);

                    var i: usize = 0;
                    // Inner product's query factor and offset are 1 and 0, so its lanes
                    // skip loading them, and an uncalibrated index has no centre terms;
                    // specialized at compile time, the plainest case is the loop it was
                    // before the other metrics joined (which cost it 3.5% unspecialized).
                    const plain_now = self.metric == .inner_product;
                    switch (plain_now) {
                        inline else => |plain| switch (self.calibrated) {
                            inline else => |centred| while (i + lanes <= n) : (i += lanes) {
                                const m: Lane = block[i..][0..lanes].*;
                                const th: Lane = thresholds[i..][0..lanes].*;
                                // `scalarScore`, lane by lane and in the same order, so
                                // the result is bit-identical to the scalar paths. For
                                // inner product (f·1)·t + (0+0) is f·t exactly.
                                const scaled = if (plain) av * m else blk: {
                                    const qf: Lane = query_factor[i..][0..lanes].*;
                                    break :blk (av * qf) * m;
                                };
                                const shared = if (centred) blk: {
                                    const qb: Lane = query_bias[i..][0..lanes].*;
                                    const qc: Lane = query_centre[i..][0..lanes].*;
                                    break :blk scaled + (biv * qb + cv * qc);
                                } else scaled;
                                const sc = if (plain) shared else blk: {
                                    const qo: Lane = query_offset[i..][0..lanes].*;
                                    break :blk shared + (bv + qo);
                                };
                                // Visit only the lanes that actually passed. Descending
                                // into all `lanes` on any hit costs eight unpredictable
                                // branches to service what is almost always a single
                                // accept, and hits get steadily more frequent as k rises
                                // and thresholds fall — which is where the cost of
                                // retrieval depth lives.
                                var mask: LaneMask = @bitCast(sc > th);
                                if (mask == 0) continue;
                                const each: [lanes]f32 = sc;
                                while (mask != 0) {
                                    const j = @ctz(mask);
                                    mask &= mask - 1;
                                    const q = i + j;
                                    hot[q][hot_len[q]] = .{ .id = @intCast(v), .score = each[j] };
                                    hot_len[q] += 1;
                                    if (hot_len[q] == hot_slots) {
                                        flushHot(&collectors[q], &hot[q], &hot_len[q], &thresholds[q]);
                                    }
                                }
                            },
                        },
                    }
                    while (i < n) : (i += 1) {
                        const sc = self.scalarScoreFor(vt, .{
                            .factor = query_factor[i],
                            .offset = query_offset[i],
                            .bias = query_bias[i],
                            .centre = query_centre[i],
                        }, block[i]);
                        if (sc > thresholds[i]) {
                            collectors[i].offer(sc, @intCast(v));
                            thresholds[i] = collectors[i].threshold();
                        }
                    }
                } else if (fast_path) {
                    // Heaps not yet full: everything is admitted, so there is nothing
                    // to reject early and the scalar path is the whole job.
                    const block = mse_block[u * max_batch ..][0..n];
                    const vt = vectorTerms(self.metric, norm, gamma, bias);
                    for (0..n) |i| {
                        const sc = self.scalarScoreFor(vt, .{
                            .factor = query_factor[i],
                            .offset = query_offset[i],
                            .bias = query_bias[i],
                            .centre = query_centre[i],
                        }, block[i]);
                        collectors[i].offer(sc, @intCast(v));
                        thresholds[i] = collectors[i].threshold();
                    }
                } else for (0..n) |i| {
                    const mse_term = if (self.residency == .expanded or use_simd)
                        mse_block[u * max_batch + i]
                    else
                        scan.scoreExact(self.layout, centroids, searcher.query_states[i].rotated, code_slot);

                    const sc = switch (self.correction) {
                        .scalar => self.scalarScoreFor(
                            vectorTerms(self.metric, norm, gamma, bias),
                            .{
                                .factor = query_factor[i],
                                .offset = query_offset[i],
                                .bias = query_bias[i],
                                .centre = query_centre[i],
                            },
                            mse_term,
                        ),
                        .qjl_sketch => blk: {
                            const sketch_term = if (!self.use_sketch)
                                0
                            else if (use_simd)
                                sketch_kernel.signDot(searcher.sketch_queries[i], sketch_slot, padded)
                            else
                                sketch_kernel.signDotExact(searcher.query_states[i].sketched, sketch_slot);
                            const estimate = norm * (mse_term + gamma * self.quantizer.sketch_scale * sketch_term);
                            break :blk self.score(estimate, norm, query_norms[i]);
                        },
                    };
                    if (!filled or sc > thresholds[i]) {
                        collectors[i].offer(sc, @intCast(v));
                        thresholds[i] = collectors[i].threshold();
                    }
                }
            }
        }
        for (0..n) |i| flushHot(&collectors[i], &hot[i], &hot_len[i], &thresholds[i]);
        for (0..n) |i| _ = collectors[i].drain();
        const found = @min(searcher.k, self.count());
        return searcher.heap_store[0 .. (n - 1) * searcher.k + found];
    }

    /// Query-parallel batch search state: one `BatchSearcher` per thread.
    ///
    /// Threads are given disjoint *queries*, not disjoint corpus shards, so each
    /// thread reads the whole corpus. That costs no extra bandwidth: total traffic is
    /// one corpus pass per batch either way, since every thread's batch has to see
    /// every vector regardless of how the work is divided. Sharding queries instead
    /// means no shared mutable state, no top-k merge, and no false sharing.
    ///
    /// This once cited "batching gives 1.01×, so per-vector compute is the
    /// constraint" as the justification. That measurement was real but the inference
    /// was wrong: batching did nothing because the batch loop called the one-query
    /// kernel per query rather than because the corpus pass was free. It now gives
    /// ~1.9× (see docs/notes.md).
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
        std.debug.assert(searcher.heap.len > 0);
        const prepared = self.prepareSingle(query, searcher);
        const use_simd = prepared.use_simd;

        var collector = topk.TopK.init(searcher.heap);
        const stride = self.codeStride();
        const padded = self.quantizer.padded();

        // A single query cannot tile over queries, but it can still tile over stored
        // vectors: the one query chunk stays in a register and is reused across the
        // tile, so the loop issues V + 1 loads against V SDOTs — 1.125 at V=8 against
        // the per-vector kernel's 2.0. Q=1 leaves registers nearly free, so the tile
        // is wider here than in the batch path. Tiles are scored a block at a time,
        // so `dispatch` costs one indirect call per `block_tiles` of them.
        const tile = tiles.single_tile;
        const block_tiles = 8;
        const queries: [1]scan.Query = .{searcher.scan_query};
        const can_tile = use_simd and
            (self.residency == .expanded or scan.canTile(self.layout));
        var block: [tile * block_tiles]f32 = undefined;
        const kernels = dispatch.active();
        const job: tiles.Job = .{
            .table = &searcher.table,
            .queries = &queries,
            .n = 1,
            .codes = self.codes.items.ptr,
            .stride = stride,
            .padded = @intCast(padded),
            .bits = self.layout.bits,
            .expanded = self.residency == .expanded,
            .table_scale = self.table.scale,
            .out = &block,
            .out_stride = 1,
        };

        const total = self.scalars.items.len;
        var v0: usize = 0;
        while (v0 < total) : (v0 += block.len) {
            const take = @min(block.len, total - v0);
            const tiled = if (can_tile) take / tile * tile else 0;
            if (tiled > 0) dispatch.singleTiles(kernels, &job, v0, tiled / tile);
            for (tiled..take) |u| {
                block[u] = self.mseTermAt(v0 + u, prepared, searcher);
            }

            // Specialized on calibration at compile time: an uncalibrated index has no
            // bias to load or centre terms to add, and testing for them per vector
            // cost this loop 10–15% at d=128, where a vector is only a few SDOTs.
            switch (self.calibrated) {
                inline else => |centred| for (0..take) |u| {
                    const i = v0 + u;
                    collector.offer(self.finishScore(centred, i, block[u], prepared, searcher), @intCast(i));
                },
            }
        }

        return collector.drain();
    }

    /// Score the stored vectors `ids` against one query, writing `out[j]` for `ids[j]`.
    ///
    /// The scores are exactly the ones `search` reports for those vectors, so this is
    /// the way to rank a candidate set from elsewhere — a metadata filter, another
    /// index — with this index's estimates, without scanning the rest of the corpus.
    pub fn scoreIds(
        self: *FlatIndex,
        query: []const f32,
        ids: []const u32,
        out: []f32,
        searcher: *Searcher,
    ) error{IdOutOfRange}!void {
        std.debug.assert(ids.len == out.len);
        for (ids) |id| if (id >= self.count()) return error.IdOutOfRange;
        const prepared = self.prepareSingle(query, searcher);
        switch (self.calibrated) {
            inline else => |centred| for (ids, out) |id, *o| {
                o.* = self.finishScore(centred, id, self.mseTermAt(id, prepared, searcher), prepared, searcher);
            },
        }
    }

    /// Per-query state shared by `search` and `scoreIds`, beyond what `searcher` holds.
    const Prepared = struct {
        use_simd: bool,
        query_norm: f32,
        query_terms: QueryTerms,
    };

    fn prepareSingle(self: *FlatIndex, query: []const f32, searcher: *Searcher) Prepared {
        std.debug.assert(query.len == self.dim());

        // One rotation and one sketch for the whole corpus — the point of staying in
        // the rotated basis (docs/DESIGN.md §1.3).
        if (self.storesSketch()) {
            self.quantizer.prepareQuery(query, &searcher.query_state, &searcher.workspace);
        } else {
            self.quantizer.prepareQueryRotated(query, &searcher.query_state, &searcher.workspace);
        }
        const use_simd = self.vectorized();
        if (use_simd) {
            searcher.scan_query.load(searcher.query_state.rotated);
            if (self.storesSketch()) searcher.sketch_query.load(searcher.query_state.sketched);
        }

        const query_norm = if (self.metric != .inner_product or self.calibrated)
            euclideanNorm(query)
        else
            0;
        return .{
            .use_simd = use_simd,
            .query_norm = query_norm,
            .query_terms = self.queryTermsFor(query_norm, searcher.query_state.rotated),
        };
    }

    /// ⟨p, ỹ_i⟩ for one stored vector, by the per-vector kernel. The tiled kernels in
    /// `search` compute the same integer dot products, so the results agree exactly.
    inline fn mseTermAt(self: *const FlatIndex, i: usize, prepared: Prepared, searcher: *const Searcher) f32 {
        const stride = self.codeStride();
        const padded = self.quantizer.padded();
        const slot = self.codes.items[i * stride ..][0..stride];
        return switch (self.residency) {
            .expanded => scan.scoreExpanded(
                @ptrCast(slot),
                searcher.scan_query,
                self.table.scale,
                padded,
            ),
            .compact => if (prepared.use_simd)
                scan.scoreInt8(self.layout, searcher.table, searcher.scan_query, slot, padded)
            else
                scan.scoreExact(self.layout, self.quantizer.mse.codebook.centroids, searcher.query_state.rotated, slot),
        };
    }

    /// The reported score of stored vector `i`, given its MSE term.
    /// `centred` must equal `self.calibrated`; it is a parameter so callers can hoist
    /// the test out of their loop.
    inline fn finishScore(self: *const FlatIndex, comptime centred: bool, i: usize, mse_term: f32, prepared: Prepared, searcher: *const Searcher) f32 {
        std.debug.assert(centred == self.calibrated);
        const scalars = self.scalars.items[i];
        const norm: f32 = scalars.norm;
        const gamma: f32 = scalars.gamma;
        return switch (self.correction) {
            .scalar => scalarScore(
                centred,
                vectorTerms(self.metric, norm, gamma, if (centred) self.biases.items[i] else 0),
                prepared.query_terms,
                mse_term,
            ),
            .qjl_sketch => blk: {
                const sketch_len = self.sketchStride();
                const sketch_slot = self.sketches.items[i * sketch_len ..][0..sketch_len];
                const sketch_term = if (!self.use_sketch)
                    0
                else if (prepared.use_simd)
                    sketch_kernel.signDot(searcher.sketch_query, sketch_slot, self.quantizer.padded())
                else
                    sketch_kernel.signDotExact(searcher.query_state.sketched, sketch_slot);
                const estimate = norm * (mse_term + gamma * self.quantizer.sketch_scale * sketch_term);
                break :blk self.score(estimate, norm, prepared.query_norm);
            },
        };
    }

    /// The per-query coefficients of `scalarScore` for a query of norm `query_norm`
    /// whose rotation is `rotated`. The centre's share, ⟨p, c⟩, is one dot product
    /// per query, and only a calibrated index has one.
    fn queryTermsFor(self: *const FlatIndex, query_norm: f32, rotated: []const f32) QueryTerms {
        var centre_dot: f32 = 0;
        if (self.calibrated) {
            var acc: f64 = 0;
            for (rotated, self.centre) |p, c| acc += @as(f64, p) * c;
            centre_dot = @floatCast(acc);
        }
        return queryTerms(self.metric, query_norm, centre_dot);
    }

    /// `scalarScore`, with the centre terms included only when the index has them.
    inline fn scalarScoreFor(self: *const FlatIndex, v: VectorTerms, q: QueryTerms, t: f32) f32 {
        return if (self.calibrated) scalarScore(true, v, q, t) else scalarScore(false, v, q, t);
    }

    /// Map an inner-product estimate onto the metric, always larger-is-better. The
    /// `.qjl_sketch` paths use this; `.scalar` uses `scalarScore`, which is the same
    /// map rearranged so the batch epilogue can evaluate it a lane group at a time.
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

/// `‖mean(x/‖x‖)‖` over the rows of `sample`: how far off the origin the corpus's
/// directions sit, from 0 (directions spread evenly) to 1 (all the same). It predicts
/// whether `calibrate` helps; see `FlatIndex.auto_calibration_threshold`. Zero rows
/// have no direction and are skipped.
pub fn centroidNorm(allocator: Allocator, sample: []const f32, dim: usize) Allocator.Error!f32 {
    std.debug.assert(dim > 0 and sample.len % dim == 0);
    const mean = try allocator.alloc(f64, dim);
    defer allocator.free(mean);
    @memset(mean, 0);
    var rows: usize = 0;
    var i: usize = 0;
    while (i < sample.len) : (i += dim) {
        const row = sample[i..][0..dim];
        var sq: f64 = 0;
        for (row) |v| sq += @as(f64, v) * v;
        if (sq == 0) continue;
        const inv = 1 / @sqrt(sq);
        for (mean, row) |*m, v| m.* += v * inv;
        rows += 1;
    }
    if (rows == 0) return 0;
    var sq: f64 = 0;
    for (mean) |m| sq += m * m;
    return @floatCast(@sqrt(sq) / @as(f64, @floatFromInt(rows)));
}

/// Per-vector coefficients of `scalarScore`.
pub const VectorTerms = struct { factor: f32, offset: f32, bias: f32, centre: f32 };
/// Per-query coefficients of `scalarScore`.
pub const QueryTerms = struct { factor: f32, offset: f32, bias: f32, centre: f32 };

/// The `.scalar`-correction score for any metric, given the scanned term t = ⟨p, ũ⟩:
///
///     score = (v.factor·q.factor)·t + (v.bias·q.bias + v.centre·q.centre)
///             + (v.offset + q.offset)
///
/// The estimate of ⟨q, x/‖x‖⟩ is e = f·t + ‖q‖·b + ⟨p, c⟩ (`encodeScalar`), and
/// with n = ‖x‖:
///
///     inner product   n·e                   factor n·f, bias n·b, centre n
///     cosine          e / ‖q‖               factor f/‖q‖, bias b, centre ⟨p,c⟩/‖q‖
///     l2              −(‖q‖² + n² − 2n·e)   factor 2n·f, bias 2n·b, centre 2n,
///                                           offset −n² − ‖q‖²
///
/// with each coefficient split between the vector and the query that own it, and the
/// query side holding ‖q‖ (as `bias`) and ⟨p, c⟩ (as `centre`). Splitting them by owner
/// is what lets the batch epilogue score a whole lane group of queries against one
/// vector with a few multiplies and adds, for every metric rather than inner product
/// alone. Every path that scores `.scalar` calls this, or evaluates it lane-wise in this
/// exact order, so batched and single-query scores stay bit-identical.
///
/// Uncalibrated, b and c are zero, and `centred = false` drops their terms rather than
/// adding zeros: the score is then the affine map it was before centring existed.
///
/// A zero vector or zero query has no direction under cosine and scores −inf, as
/// `score` does: its factor and centre terms are zero and its offset −inf.
pub inline fn scalarScore(comptime centred: bool, v: VectorTerms, q: QueryTerms, t: f32) f32 {
    const scaled = (v.factor * q.factor) * t;
    const shared = if (centred) scaled + (v.bias * q.bias + v.centre * q.centre) else scaled;
    return shared + (v.offset + q.offset);
}

pub inline fn vectorTerms(metric: Metric, norm: f32, factor: f32, bias: f32) VectorTerms {
    return switch (metric) {
        .inner_product => .{ .factor = norm * factor, .offset = 0, .bias = norm * bias, .centre = norm },
        .cosine => if (norm > 0)
            .{ .factor = factor, .offset = 0, .bias = bias, .centre = 1 }
        else
            .{ .factor = 0, .offset = -std.math.inf(f32), .bias = 0, .centre = 0 },
        .l2 => .{
            .factor = 2 * norm * factor,
            .offset = -(norm * norm),
            .bias = 2 * norm * bias,
            .centre = 2 * norm,
        },
    };
}

/// `centre_dot` is ⟨p, c⟩, zero for an uncalibrated index.
pub fn queryTerms(metric: Metric, query_norm: f32, centre_dot: f32) QueryTerms {
    return switch (metric) {
        .inner_product => .{ .factor = 1, .offset = 0, .bias = query_norm, .centre = centre_dot },
        .cosine => if (query_norm > 0)
            .{ .factor = 1 / query_norm, .offset = 0, .bias = 1, .centre = centre_dot / query_norm }
        else
            .{ .factor = 0, .offset = -std.math.inf(f32), .bias = 0, .centre = 0 },
        .l2 => .{ .factor = 1, .offset = -(query_norm * query_norm), .bias = query_norm, .centre = centre_dot },
    };
}

/// Drain staged candidates into a collector and refresh its cached threshold.
pub inline fn flushHot(
    collector: *topk.TopK,
    staged: *[hot_slots]Entry,
    len: *usize,
    threshold: *f32,
) void {
    for (staged[0..len.*]) |e| collector.offer(e.score, e.id);
    len.* = 0;
    threshold.* = collector.threshold();
}

/// The per-vector half of a `.scalar` code: the factor f and the bias b.
pub const Residual = struct { factor: f32, bias: f32 };

/// Quantize the rotated direction y in `ws.rotated` against `centre` (see
/// `FlatIndex.encodeScalar` for the split), writing one code per padded dimension to
/// `codes`. With no centre, y is quantized as it stands and the bias is zero: the
/// uncalibrated encoding, whose codes predate centring and must not change. With one,
/// the residual is renormalized first, since the codebook is built for unit vectors.
/// `IvfIndex` calls this with each vector's list centroid.
pub fn encodeResidual(
    quantizer: *const prod_mod.Prod,
    centre: ?[]const f32,
    codes: []u8,
    ws: *prod_mod.Workspace,
) Residual {
    const residual = if (centre != null) ws.residual else ws.rotated;
    if (centre) |c| {
        for (ws.residual, ws.rotated, c) |*r, y, cv| r.* = y - cv;
    }
    var rr: f64 = 0;
    for (residual) |r| rr += @as(f64, r) * r;

    if (centre != null) {
        const inv: f32 = if (rr > 0) @floatCast(1 / @sqrt(rr)) else 0;
        for (ws.staging, residual) |*u, r| u.* = r * inv;
        quantizer.mse.codebook.encodeSlice(ws.staging, codes);
    } else {
        quantizer.mse.codebook.encodeSlice(residual, codes);
    }

    const levels = quantizer.mse.codebook.centroids;
    var ru: f64 = 0;
    var cr: f64 = 0;
    var cu: f64 = 0;
    if (centre) |c| {
        for (codes, residual, c) |code, r, cv| {
            const u: f64 = levels[code];
            ru += r * u;
            cr += @as(f64, cv) * r;
            cu += @as(f64, cv) * u;
        }
    } else {
        for (codes, residual) |code, r| ru += r * @as(f64, levels[code]);
    }
    // ⟨r, ũ⟩ > 0 for any nonzero r: each coordinate's nearest level has its sign.
    const factor: f64 = if (rr > 0 and ru > 0) rr / ru else 0;
    return .{ .factor = @floatCast(factor), .bias = @floatCast(cr - factor * cu) };
}

/// Whether every value is finite. `v * 0` is zero for a finite `v` and NaN for an
/// infinity or a NaN, so the sum is zero exactly when all are finite — a branch-free
/// pass that costs far less than encoding the same values.
pub fn allFinite(x: []const f32) bool {
    const width = 16;
    const V = @Vector(width, f32);
    const zero: V = @splat(0);
    var acc: V = zero;
    var i: usize = 0;
    while (i + width <= x.len) : (i += width) {
        const v: V = x[i..][0..width].*;
        acc += v * zero;
    }
    var tail: f32 = 0;
    for (x[i..]) |v| tail += v * 0;
    return @reduce(.Add, acc) + tail == 0;
}

pub fn euclideanNorm(x: []const f32) f32 {
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

test "centring improves recall on an off-centre corpus" {
    // Real embeddings sit well off the origin, so much of every vector is shared.
    // Centring stores only the part that is not, and computes the shared part's share
    // of each score exactly.
    const allocator = testing.allocator;
    const dim: u32 = 256;
    const n = 1200;

    // Off-centre and anisotropic: a common offset plus energy concentrated in a
    // low-dimensional subspace, which is what real embeddings look like.
    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(0xA15E);
    const random = prng.random();
    for (0..n) |i| {
        const row = corpus[i * dim ..][0..dim];
        for (row, 0..) |*v, j| {
            const decay = 1.0 / (1.0 + @as(f32, @floatFromInt(j)) * 0.08);
            v.* = random.floatNorm(f32) * decay + 0.15;
        }
        const len = euclideanNorm(row);
        for (row) |*v| v.* /= len;
    }

    var plain = try FlatIndex.init(allocator, .{ .dim = dim, .bits = 3, .seed = 2 });
    defer plain.deinit();
    try plain.addBatch(corpus);

    var fitted = try FlatIndex.init(allocator, .{ .dim = dim, .bits = 3, .seed = 2 });
    defer fitted.deinit();
    try fitted.calibrate(corpus[0 .. 512 * dim]);
    try fitted.addBatch(corpus);

    // The fit must actually find the offset, or the comparison is vacuous. The centre
    // is a mean of unit directions, so its norm is the sample's centroid norm.
    var sq: f64 = 0;
    for (fitted.centre) |c| sq += @as(f64, c) * c;
    const want = try centroidNorm(allocator, corpus[0 .. 512 * dim], dim);
    try testing.expectApproxEqAbs(want, @as(f32, @floatCast(@sqrt(sq))), 1e-4);
    try testing.expect(want > 0.5);

    // Two bytes more per vector: the bias.
    try testing.expectEqual(plain.bytesPerVector() + 2, fitted.bytesPerVector());

    const query = try allocator.alloc(f32, dim);
    defer allocator.free(query);
    var hits = [2]usize{ 0, 0 };
    const trials = 100;

    for ([_]*FlatIndex{ &plain, &fitted }, 0..) |index, which| {
        var searcher = try FlatIndex.Searcher.init(allocator, index.*, 10);
        defer searcher.deinit();
        var qprng = std.Random.DefaultPrng.init(31);
        for (0..trials) |_| {
            // Queries from the same distribution as the corpus.
            const src = corpus[qprng.random().uintLessThan(usize, n) * dim ..][0..dim];
            for (query, src) |*q, x| q.* = x + 0.05 * qprng.random().floatNorm(f32);
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
                    hits[which] += 1;
                    break;
                }
            }
        }
    }
    try testing.expect(hits[1] >= hits[0]);
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
    // locality and scores lane groups at a time; it must not change a result.
    const allocator = testing.allocator;
    const dim: u32 = 128;
    const n = 800;

    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(0xBA7C);
    const random = prng.random();
    // Off-centre, with varied norms, so every term of the score is live.
    offCentreRows(corpus, dim, random);

    for ([_]Metric{ .inner_product, .cosine, .l2 }) |metric| for ([_]bool{ false, true }) |calibrate| {
        var index = try FlatIndex.init(allocator, .{
            .dim = dim,
            .bits = 5,
            .metric = metric,
            .seed = 0x5B,
        });
        defer index.deinit();
        if (calibrate) try index.calibrate(corpus);
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
    };
}

test "batched search handles partial batches and calibration" {
    const allocator = testing.allocator;
    const dim: u32 = 64;
    const n = 300;

    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(5);
    const random = prng.random();
    offCentreRows(corpus, dim, random);

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
            for (one, grouped[i * k ..][0..k]) |a, b| {
                try testing.expectEqual(a.id, b.id);
                try testing.expectEqual(a.score, b.score);
            }
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

/// Off-centre, anisotropic rows with norms spread over [0.5, 2.5): the case where the
/// centre, the factor and the bias all do real work.
fn offCentreRows(rows: []f32, dim: usize, random: std.Random) void {
    for (0..rows.len / dim) |i| {
        const row = rows[i * dim ..][0..dim];
        for (row, 0..) |*v, j| {
            const decay = 1.0 / (1.0 + @as(f32, @floatFromInt(j)) * 0.05);
            v.* = (random.floatNorm(f32) + 0.7) * decay;
        }
        const len = euclideanNorm(row);
        const want = 0.5 + 2.0 * random.float(f32);
        for (row) |*v| v.* *= want / len;
    }
}

test "a stored vector's estimate against itself is its squared norm" {
    // The defining property of the factor: the estimate is exact when the query points
    // along the stored vector. Querying with x itself, the inner-product estimate must
    // be ‖x‖², cosine 1, and L2 0, calibrated or not.
    //
    // This checks the whole score, the centre term, the bias and its ‖q‖ weighting
    // included, against something no formula in the index supplies. A stale or
    // misplaced term — the bias left unscaled by ‖q‖, say, or the least-squares factor
    // in place of this one — misses by percent, not by rounding.
    const allocator = testing.allocator;
    const dim: u32 = 128;
    const n = 64;
    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(0xA1FA);
    offCentreRows(corpus, dim, prng.random());

    for ([_]bool{ false, true }) |calibrate| {
        for ([_]Metric{ .inner_product, .cosine, .l2 }) |metric| {
            for ([_]bool{ true, false }) |exact| {
                var index = try FlatIndex.init(allocator, .{
                    .dim = dim,
                    .bits = 3,
                    .seed = 0x33,
                    .metric = metric,
                    .exact_scan = exact,
                });
                defer index.deinit();
                if (calibrate) try index.calibrate(corpus);
                try index.addBatch(corpus);
                var searcher = try FlatIndex.Searcher.init(allocator, index, 1);
                defer searcher.deinit();

                // Half-precision scalars bound the exact scan; the int8 query adds a
                // little more.
                const tolerance: f32 = if (exact) 4e-3 else 3e-2;
                for (0..n) |i| {
                    const x = corpus[i * dim ..][0..dim];
                    const id = [1]u32{@intCast(i)};
                    var got: [1]f32 = undefined;
                    try index.scoreIds(x, &id, &got, &searcher);
                    const sq = euclideanNorm(x) * euclideanNorm(x);
                    switch (metric) {
                        .inner_product => try testing.expectApproxEqRel(sq, got[0], tolerance),
                        .cosine => try testing.expectApproxEqAbs(@as(f32, 1), got[0], tolerance),
                        .l2 => try testing.expectApproxEqAbs(@as(f32, 0), got[0], 2 * tolerance * sq),
                    }
                }
            }
        }
    }
}

test "the stored factor and bias are the ones the split defines" {
    // f = ‖r‖²/⟨r, ũ⟩ and b = ⟨c, r⟩ − f·⟨c, ũ⟩, for r = y − c and ũ the reconstruction
    // of r/‖r‖, recomputed here from the definitions in f64.
    //
    // The self-score test cannot tell every wrong definition apart: any factor that is
    // exact along the vector passes it. Fitting f by least squares instead, ⟨r, ũ⟩/‖ũ‖²,
    // is the earlier choice and fails it, but a sign slip in b that happens to cancel
    // would not. Checking each scalar against its definition does.
    const allocator = testing.allocator;
    const dim: u32 = 96;
    const n = 128;
    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    var prng = std.Random.DefaultPrng.init(0xB2C3);
    offCentreRows(corpus, dim, prng.random());

    var index = try FlatIndex.init(allocator, .{ .dim = dim, .bits = 3, .seed = 0x77 });
    defer index.deinit();
    try index.calibrate(corpus);
    try index.addBatch(corpus);

    const padded = index.quantizer.padded();
    const staging = try allocator.alloc(f32, padded);
    defer allocator.free(staging);
    const y = try allocator.alloc(f32, padded);
    defer allocator.free(y);
    const u = try allocator.alloc(f32, padded);
    defer allocator.free(u);
    const codes = try allocator.alloc(u8, padded);
    defer allocator.free(codes);
    const levels = index.quantizer.mse.codebook.centroids;

    for (0..n) |id| {
        _ = index.quantizer.mse.encodeRotated(corpus[id * dim ..][0..dim], y, staging);
        var rr: f64 = 0;
        for (y, index.centre) |yy, c| rr += (@as(f64, yy) - c) * (@as(f64, yy) - c);
        for (u, y, index.centre) |*uu, yy, c| uu.* = @floatCast((@as(f64, yy) - c) / @sqrt(rr));
        index.quantizer.mse.codebook.encodeSlice(u, codes);

        var ru: f64 = 0;
        var cr: f64 = 0;
        var cu: f64 = 0;
        for (codes, y, index.centre) |code, yy, c| {
            const r = @as(f64, yy) - c;
            ru += r * levels[code];
            cr += c * r;
            cu += c * @as(f64, levels[code]);
        }
        const f = rr / ru;
        try testing.expectApproxEqRel(@as(f32, @floatCast(f)), @as(f32, index.scalars.items[id].gamma), 1e-3);
        try testing.expectApproxEqAbs(@as(f32, @floatCast(cr - f * cu)), @as(f32, index.biases.items[id]), 1e-3);
    }
}

test "batch search fills k results when some vectors score -inf" {
    // A zero vector has no direction, so under cosine it scores -inf. The batch
    // loop caches each collector's admission threshold to avoid three loads per
    // (vector, query) pair, and that cache starts at -inf — a strict compare would
    // reject a -inf score even while the heap still has room, and the returned
    // slice would be shorter than the k entries per query that callers index by.
    const allocator = testing.allocator;
    const dim: u32 = 64;
    const n = 8;
    const k = 8;

    const corpus = try allocator.alloc(f32, n * dim);
    defer allocator.free(corpus);
    @memset(corpus, 0);
    var prng = std.Random.DefaultPrng.init(0x2E20);
    const random = prng.random();
    // Half the corpus is left as zero vectors.
    for (0..n / 2) |i| randomUnit(corpus[i * dim ..][0..dim], random);

    var index = try FlatIndex.init(allocator, .{
        .dim = dim,
        .bits = 5,
        .seed = 0x11,
        .metric = .cosine,
    });
    defer index.deinit();
    try index.addBatch(corpus);

    const nq = 3;
    const queries = try allocator.alloc(f32, nq * dim);
    defer allocator.free(queries);
    for (0..nq) |i| randomUnit(queries[i * dim ..][0..dim], random);

    var searcher = try FlatIndex.BatchSearcher.init(allocator, index, nq, k);
    defer searcher.deinit();
    const res = index.searchBatch(queries, &searcher);

    try testing.expectEqual(@as(usize, nq * k), res.len);
    for (0..nq) |q| {
        for (res[q * k ..][0..k]) |e| try testing.expect(e.id < n);
    }
}

test "allFinite flags NaN and both infinities anywhere in the slice" {
    var buf: [37]f32 = undefined;
    for (&buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) - 18.5;
    try testing.expect(allFinite(&buf));
    try testing.expect(allFinite(buf[0..0]));
    // Positions inside the vector body and in the scalar tail.
    for ([_]usize{ 0, 15, 16, 31, 32, 36 }) |pos| {
        for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| {
            const saved = buf[pos];
            buf[pos] = bad;
            try testing.expect(!allFinite(&buf));
            buf[pos] = saved;
        }
    }
}

test "non-finite vectors are refused and leave the index unchanged" {
    var index = try FlatIndex.init(testing.allocator, .{ .dim = 32, .bits = 4 });
    defer index.deinit();

    var good: [32]f32 = @splat(0.25);
    _ = try index.add(&good);

    var bad = good;
    bad[7] = std.math.nan(f32);
    try testing.expectError(error.NonFiniteInput, index.add(&bad));
    try testing.expectEqual(@as(usize, 1), index.count());

    // A bad row at the end of a batch rejects the whole batch, not just the tail.
    var batch: [3 * 32]f32 = @splat(0.25);
    batch[2 * 32 + 5] = std.math.inf(f32);
    try testing.expectError(error.NonFiniteInput, index.addBatch(&batch));
    try testing.expectEqual(@as(usize, 1), index.count());

    var fresh = try FlatIndex.init(testing.allocator, .{ .dim = 32, .bits = 4 });
    defer fresh.deinit();
    try testing.expectError(error.NonFiniteInput, fresh.calibrate(&batch));
}

test "calibration is refused with the QJL sketch correction" {
    var index = try FlatIndex.init(testing.allocator, .{ .dim = 32, .bits = 4, .correction = .qjl_sketch });
    defer index.deinit();
    const sample: [8 * 32]f32 = @splat(0.5);
    try testing.expectError(error.CalibrationNeedsScalarCorrection, index.calibrate(&sample));
    try testing.expect(!index.calibrated);
}

test "resident storage is what bytesPerVector reports" {
    // The sketch used to be computed and stored for every vector even under the
    // scalar correction, which never reads it: 32 resident bytes per vector at d=256
    // that the reported figure did not include.
    var prng = std.Random.DefaultPrng.init(0x51DE);
    const random = prng.random();
    for ([_]Correction{ .scalar, .qjl_sketch }) |correction| for ([_]bool{ false, true }) |calibrate| {
        if (calibrate and correction != .scalar) continue;
        for ([_]u6{ 3, 5 }) |bits| {
            var index = try FlatIndex.init(testing.allocator, .{
                .dim = 256,
                .bits = bits,
                .correction = correction,
            });
            defer index.deinit();
            var v: [256]f32 = undefined;
            if (calibrate) {
                var sample: [2 * 256]f32 = undefined;
                for (0..2) |i| randomUnit(sample[i * 256 ..][0..256], random);
                try index.calibrate(&sample);
            }
            for (0..20) |_| {
                randomUnit(&v, random);
                _ = try index.add(&v);
            }
            const n = index.count();
            const resident = index.codes.items.len + index.sketches.items.len +
                index.scalars.items.len * @sizeOf(StoredScalars) + index.biases.items.len * @sizeOf(f16);
            try testing.expectEqual(index.bytesPerVector() * n, resident);
            try testing.expectEqual(correction == .qjl_sketch, index.sketches.items.len > 0);
            try testing.expectEqual(calibrate, index.biases.items.len > 0);
        }
    };
}

test "parallel add produces exactly the bytes of a serial add" {
    // Each configuration that encodes differently: scalar and sketch corrections,
    // calibrated codes, expanded residency, and a row count no thread count divides.
    var prng = std.Random.DefaultPrng.init(0x9A2A);
    const random = prng.random();
    const dim = 96;
    const n = 1500;
    const data = try testing.allocator.alloc(f32, n * dim);
    defer testing.allocator.free(data);
    for (0..n) |i| randomUnit(data[i * dim ..][0..dim], random);

    const Case = struct { correction: Correction, residency: Residency, calibrate: bool };
    for ([_]Case{
        .{ .correction = .scalar, .residency = .compact, .calibrate = false },
        .{ .correction = .scalar, .residency = .compact, .calibrate = true },
        .{ .correction = .scalar, .residency = .expanded, .calibrate = false },
        .{ .correction = .qjl_sketch, .residency = .compact, .calibrate = false },
    }) |case| {
        for ([_]usize{ 2, 3, 7, 64 }) |threads| {
            var serial = try FlatIndex.init(testing.allocator, .{ .dim = dim, .bits = 4, .correction = case.correction, .residency = case.residency });
            defer serial.deinit();
            var parallel = try FlatIndex.init(testing.allocator, .{ .dim = dim, .bits = 4, .correction = case.correction, .residency = case.residency });
            defer parallel.deinit();
            if (case.calibrate) {
                try serial.calibrate(data[0 .. 200 * dim]);
                try parallel.calibrate(data[0 .. 200 * dim]);
            }

            // Two batches, so the second starts at a non-zero id.
            try serial.addBatch(data[0 .. 100 * dim]);
            try serial.addBatch(data[100 * dim ..]);
            try parallel.addBatchParallel(data[0 .. 100 * dim], threads);
            try parallel.addBatchParallel(data[100 * dim ..], threads);

            try testing.expectEqual(serial.count(), parallel.count());
            try testing.expectEqualSlices(u8, serial.codes.items, parallel.codes.items);
            try testing.expectEqualSlices(u8, serial.sketches.items, parallel.sketches.items);
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(serial.scalars.items), std.mem.sliceAsBytes(parallel.scalars.items));
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(serial.biases.items), std.mem.sliceAsBytes(parallel.biases.items));
        }
    }
}

test "parallel add refuses a non-finite batch whole" {
    var index = try FlatIndex.init(testing.allocator, .{ .dim = 32, .bits = 4 });
    defer index.deinit();
    const batch = try testing.allocator.alloc(f32, 2000 * 32);
    defer testing.allocator.free(batch);
    @memset(batch, 0.25);
    batch[1999 * 32] = std.math.nan(f32);
    try testing.expectError(error.NonFiniteInput, index.addBatchParallel(batch, 4));
    try testing.expectEqual(@as(usize, 0), index.count());
}

test "scoreIds reports exactly the scores search does" {
    var prng = std.Random.DefaultPrng.init(0x5C0E);
    const random = prng.random();
    const dim = 128;
    const n = 300;
    var data: [n * dim]f32 = undefined;
    for (0..n) |i| randomUnit(data[i * dim ..][0..dim], random);

    const Case = struct { params: Params, calibrate: bool };
    for ([_]Case{
        .{ .params = .{ .dim = dim, .bits = 5 }, .calibrate = false },
        .{ .params = .{ .dim = dim, .bits = 4, .metric = .cosine }, .calibrate = true },
        .{ .params = .{ .dim = dim, .bits = 4, .metric = .l2, .residency = .expanded }, .calibrate = false },
        .{ .params = .{ .dim = dim, .bits = 3, .correction = .qjl_sketch }, .calibrate = false },
        .{ .params = .{ .dim = dim, .bits = 4, .exact_scan = true }, .calibrate = false },
    }) |case| {
        var index = try FlatIndex.init(testing.allocator, case.params);
        defer index.deinit();
        if (case.calibrate) try index.calibrate(&data);
        try index.addBatch(&data);
        var searcher = try FlatIndex.Searcher.init(testing.allocator, index, 20);
        defer searcher.deinit();

        const q = data[17 * dim ..][0..dim];
        var found: [20]Entry = undefined;
        @memcpy(&found, index.search(q, &searcher));
        var ids: [20]u32 = undefined;
        for (&ids, found) |*id, e| id.* = e.id;
        var scores: [20]f32 = undefined;
        try index.scoreIds(q, &ids, &scores, &searcher);
        for (found, scores) |e, got| try testing.expectEqual(e.score, got);

        const bad = [_]u32{ 0, n };
        var out: [2]f32 = undefined;
        try testing.expectError(error.IdOutOfRange, index.scoreIds(q, &bad, &out, &searcher));
    }
}

test "centroidNorm: 0 for balanced directions, 1 for identical ones" {
    const a = [_]f32{ 1, 0, -1, 0, 0, 2, 0, -2 };
    try testing.expectApproxEqAbs(@as(f32, 0), try centroidNorm(testing.allocator, &a, 2), 1e-6);
    const b = [_]f32{ 3, 4, 6, 8, 0, 0 }; // the zero row is skipped
    try testing.expectApproxEqAbs(@as(f32, 1), try centroidNorm(testing.allocator, &b, 2), 1e-6);
    const z = [_]f32{ 0, 0, 0, 0 };
    try testing.expectEqual(@as(f32, 0), try centroidNorm(testing.allocator, &z, 2));
}

test "calibrateAuto calibrates an off-centre corpus and leaves a centred one alone" {
    var prng = std.Random.DefaultPrng.init(0xA070);
    const random = prng.random();
    const dim = 64;
    const n = 2000;
    const data = try testing.allocator.alloc(f32, n * dim);
    defer testing.allocator.free(data);

    for ([_]f32{ 0, 0.5 }) |offset| {
        for (data) |*v| v.* = random.floatNorm(f32) + offset;
        var index = try FlatIndex.init(testing.allocator, .{ .dim = dim, .bits = 4 });
        defer index.deinit();
        const got = try index.calibrateAuto(data);
        try testing.expectEqual(offset > 0, got.applied);
        try testing.expectEqual(got.applied, index.calibrated);
        // Offset o in d dims: o·√d / √(d·(1 + o²)) = 0.447 at o = 0.5; ~1/√n at 0.
        if (offset == 0)
            try testing.expect(got.centroid_norm < 0.1)
        else
            try testing.expectApproxEqAbs(@as(f32, 0.447), got.centroid_norm, 0.03);
    }

    // The same errors as calibrate, whatever the data.
    var full = try FlatIndex.init(testing.allocator, .{ .dim = dim, .bits = 4 });
    defer full.deinit();
    _ = try full.add(data[0..dim]);
    try testing.expectError(error.IndexNotEmpty, full.calibrateAuto(data));
    var sketch = try FlatIndex.init(testing.allocator, .{ .dim = dim, .bits = 4, .correction = .qjl_sketch });
    defer sketch.deinit();
    try testing.expectError(error.CalibrationNeedsScalarCorrection, sketch.calibrateAuto(data));
}
