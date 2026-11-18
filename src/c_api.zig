//! C ABI for zquant.
//!
//! The foundation the Python, JavaScript and Go clients all sit on, so the shape here is
//! chosen for binding authors rather than for Zig callers:
//!
//! * Opaque handles. Nothing in `zquant.h` depends on Zig struct layout, so the header
//!   stays valid across changes that do not alter behaviour.
//! * Status codes, never panics. A binding cannot recover from a Zig panic, so every
//!   fallible entry point returns `zq_status` and the library owns no path that aborts on
//!   bad input from the caller.
//! * Explicit searcher handles. Search needs per-query scratch, and hiding that inside
//!   `zq_search` would either allocate on every call or hold shared mutable state that no
//!   binding could safely thread. Making it a handle lets a caller own one per thread.
//! * Caller-provided output buffers. No memory crosses the boundary that the caller did
//!   not allocate, which removes the "who frees this" question from every binding.

const std = @import("std");
const zq = @import("root.zig");

const allocator = std.heap.c_allocator;

pub const ZQ_OK: c_int = 0;
pub const ZQ_ERR_ALLOC: c_int = -1;
pub const ZQ_ERR_INVALID: c_int = -2;
pub const ZQ_ERR_STATE: c_int = -3;
pub const ZQ_ERR_UNSUPPORTED: c_int = -4;
pub const ZQ_ERR_CPU: c_int = -5;
pub const ZQ_ERR_FORMAT: c_int = -6;

/// From src/cpu_probe.zig, compiled for the architecture baseline so it runs anywhere.
extern fn zq_cpu_supported() c_int;

/// Largest dimension the library accepts. Far above any embedding or head dimension in
/// use, and low enough that `dim` rounded up for the rotation, and every per-vector
/// size derived from it, fits comfortably in the integer types that hold them.
pub const ZQ_MAX_DIM: u32 = 1 << 16;

/// Largest worker count a searcher accepts. A guard against a garbage argument
/// becoming a request for millions of threads, not a tuning limit.
pub const ZQ_MAX_THREADS: usize = 1024;

pub const ZQ_METRIC_INNER_PRODUCT: c_int = 0;
pub const ZQ_METRIC_COSINE: c_int = 1;
pub const ZQ_METRIC_L2: c_int = 2;

pub const ZqConfig = extern struct {
    dim: u32,
    /// Total bit budget per coordinate, 2..6. The scalar codebook uses `bits - 1`, so
    /// 1 would leave it no bits at all.
    bits: u8,
    metric: c_int,
    seed: u64,
    /// Non-zero keeps codes packed (smaller); zero dequantizes to int8 (faster, ~2x memory).
    compact: c_int,
};

const Index = struct {
    inner: zq.flat.FlatIndex,
};

const Searcher = struct {
    /// The index this searcher's scratch was sized for. Query state depends on the
    /// index's dimension, layout and residency, so a searcher used on another index
    /// indexes past its buffers — a heap overflow in ReleaseFast, with OK returned.
    owner: *const Index,
    batch: zq.flat.FlatIndex.BatchSearcher,
    parallel: ?zq.flat.FlatIndex.ParallelSearcher,
    /// Single-query state for `zq_score_ids`.
    scorer: zq.flat.FlatIndex.Searcher,
    k: usize,
    capacity: usize,
};

/// `n` rows of `d` floats as a slice, or null if `n * d` does not fit in a `usize`.
/// Without the check a huge `n` wraps and the slice silently covers the wrong length.
fn rowsOf(ptr: [*]const f32, n: usize, d: usize) ?[]const f32 {
    const len = std.math.mul(usize, n, d) catch return null;
    return ptr[0..len];
}

fn metricFrom(m: c_int) ?zq.flat.Metric {
    return switch (m) {
        ZQ_METRIC_INNER_PRODUCT => .inner_product,
        ZQ_METRIC_COSINE => .cosine,
        ZQ_METRIC_L2 => .l2,
        else => null,
    };
}

export fn zq_version() [*:0]const u8 {
    return "0.1.0";
}

/// Which scan kernels searches use: native instructions or the portable scalar
/// fallback, and on x86_64 whether the 256-bit (AVX2) ones were chosen for this CPU
/// (`simd/dispatch.zig`). Each candidate's string sits in the binary as a literal, so
/// CI checks a cross-compiled release with `grep -a` instead of a disassembler for a
/// foreign architecture.
export fn zq_build_kernels() [*:0]const u8 {
    return zq.simd_dispatch.description();
}

export fn zq_status_string(status: c_int) [*:0]const u8 {
    return switch (status) {
        ZQ_OK => "ok",
        ZQ_ERR_ALLOC => "out of memory",
        ZQ_ERR_INVALID => "invalid argument",
        ZQ_ERR_STATE => "operation not valid in this state",
        ZQ_ERR_UNSUPPORTED => "unsupported configuration",
        ZQ_ERR_CPU => "this CPU lacks instructions this build of the library needs; use a portable build",
        ZQ_ERR_FORMAT => "not a zquant index, damaged, or saved by an incompatible version",
        else => "unknown status",
    };
}

export fn zq_index_create(config: ?*const ZqConfig, out: ?**Index) c_int {
    // First, before any code that might use the instructions in question. Bindings
    // check zq_cpu_supported themselves on load; this catches a C caller who did not.
    if (zq_cpu_supported() == 0) return ZQ_ERR_CPU;
    const cfg = config orelse return ZQ_ERR_INVALID;
    const slot = out orelse return ZQ_ERR_INVALID;
    // bits = 1 used to be accepted here and then reached `8 % 0` in the packing layout:
    // a crash at search time in ReleaseFast rather than an error at create.
    if (cfg.dim == 0 or cfg.dim > ZQ_MAX_DIM) return ZQ_ERR_INVALID;
    if (cfg.bits < 2 or cfg.bits > 6) return ZQ_ERR_INVALID;
    const metric = metricFrom(cfg.metric) orelse return ZQ_ERR_INVALID;

    const index = allocator.create(Index) catch return ZQ_ERR_ALLOC;
    errdefer allocator.destroy(index);

    index.inner = zq.flat.FlatIndex.init(allocator, .{
        .dim = cfg.dim,
        .bits = @intCast(cfg.bits),
        .metric = metric,
        .seed = cfg.seed,
        .residency = if (cfg.compact != 0) .compact else .expanded,
    }) catch |e| {
        allocator.destroy(index);
        return switch (e) {
            error.OutOfMemory => ZQ_ERR_ALLOC,
            else => ZQ_ERR_UNSUPPORTED,
        };
    };
    slot.* = index;
    return ZQ_OK;
}

export fn zq_index_free(index: ?*Index) void {
    const ix = index orelse return;
    ix.inner.deinit();
    allocator.destroy(ix);
}

export fn zq_index_calibrate(index: ?*Index, rows: ?[*]const f32, n: usize) c_int {
    const ix = index orelse return ZQ_ERR_INVALID;
    const data = rows orelse return ZQ_ERR_INVALID;
    if (n == 0) return ZQ_ERR_INVALID;
    const sample = rowsOf(data, n, ix.inner.dim()) orelse return ZQ_ERR_INVALID;
    ix.inner.calibrate(sample) catch |e| return switch (e) {
        error.OutOfMemory => ZQ_ERR_ALLOC,
        error.IndexNotEmpty => ZQ_ERR_STATE,
        error.EmptySample, error.NonFiniteInput => ZQ_ERR_INVALID,
        error.CalibrationNeedsScalarCorrection => ZQ_ERR_UNSUPPORTED,
    };
    return ZQ_OK;
}

/// Calibrate only if the sample's centroid norm says it will help. Either out pointer
/// may be null.
export fn zq_index_calibrate_auto(
    index: ?*Index,
    rows: ?[*]const f32,
    n: usize,
    centroid_norm: ?*f32,
    applied: ?*c_int,
) c_int {
    const ix = index orelse return ZQ_ERR_INVALID;
    const data = rows orelse return ZQ_ERR_INVALID;
    if (n == 0) return ZQ_ERR_INVALID;
    const sample = rowsOf(data, n, ix.inner.dim()) orelse return ZQ_ERR_INVALID;
    const got = ix.inner.calibrateAuto(sample) catch |e| return switch (e) {
        error.OutOfMemory => ZQ_ERR_ALLOC,
        error.IndexNotEmpty => ZQ_ERR_STATE,
        error.EmptySample, error.NonFiniteInput => ZQ_ERR_INVALID,
        error.CalibrationNeedsScalarCorrection => ZQ_ERR_UNSUPPORTED,
    };
    if (centroid_norm) |c| c.* = got.centroid_norm;
    if (applied) |a| a.* = @intFromBool(got.applied);
    return ZQ_OK;
}

export fn zq_index_add(index: ?*Index, rows: ?[*]const f32, n: usize) c_int {
    const ix = index orelse return ZQ_ERR_INVALID;
    const data = rows orelse return ZQ_ERR_INVALID;
    if (n == 0) return ZQ_OK;
    const vectors = rowsOf(data, n, ix.inner.dim()) orelse return ZQ_ERR_INVALID;
    ix.inner.addBatch(vectors) catch |e| return switch (e) {
        error.OutOfMemory => ZQ_ERR_ALLOC,
        error.NonFiniteInput => ZQ_ERR_INVALID,
    };
    return ZQ_OK;
}

/// `zq_index_add` with encoding spread over up to `threads` threads. Same result,
/// byte for byte; small batches run on the calling thread.
export fn zq_index_add_parallel(index: ?*Index, rows: ?[*]const f32, n: usize, threads: usize) c_int {
    const ix = index orelse return ZQ_ERR_INVALID;
    const data = rows orelse return ZQ_ERR_INVALID;
    if (threads > ZQ_MAX_THREADS) return ZQ_ERR_INVALID;
    if (n == 0) return ZQ_OK;
    const vectors = rowsOf(data, n, ix.inner.dim()) orelse return ZQ_ERR_INVALID;
    ix.inner.addBatchParallel(vectors, @max(threads, 1)) catch |e| return switch (e) {
        error.OutOfMemory => ZQ_ERR_ALLOC,
        error.NonFiniteInput => ZQ_ERR_INVALID,
    };
    return ZQ_OK;
}

/// Bytes `zq_index_serialize` writes for this index as it stands now.
export fn zq_index_serialized_size(index: ?*const Index) usize {
    const ix = index orelse return 0;
    return zq.persist.serializedSize(&ix.inner);
}

export fn zq_index_serialize(index: ?*const Index, buf: ?[*]u8, len: usize, written: ?*usize) c_int {
    const ix = index orelse return ZQ_ERR_INVALID;
    const out = buf orelse return ZQ_ERR_INVALID;
    const n = zq.persist.write(&ix.inner, out[0..len]) catch return ZQ_ERR_INVALID;
    if (written) |w| w.* = n;
    return ZQ_OK;
}

export fn zq_index_deserialize(buf: ?[*]const u8, len: usize, out: ?**Index) c_int {
    if (zq_cpu_supported() == 0) return ZQ_ERR_CPU;
    const data = buf orelse return ZQ_ERR_INVALID;
    const slot = out orelse return ZQ_ERR_INVALID;
    const index = allocator.create(Index) catch return ZQ_ERR_ALLOC;
    index.inner = zq.persist.read(allocator, data[0..len]) catch |e| {
        allocator.destroy(index);
        return switch (e) {
            error.OutOfMemory => ZQ_ERR_ALLOC,
            else => ZQ_ERR_FORMAT,
        };
    };
    slot.* = index;
    return ZQ_OK;
}

export fn zq_index_count(index: ?*const Index) usize {
    const ix = index orelse return 0;
    return ix.inner.count();
}

export fn zq_index_bytes_per_vector(index: ?*const Index) usize {
    const ix = index orelse return 0;
    return ix.inner.bytesPerVector();
}

export fn zq_index_dim(index: ?*const Index) u32 {
    const ix = index orelse return 0;
    return ix.inner.dim();
}

/// The configuration the index was created with, or was saved with if it was loaded.
/// Creating an index from it gives one that encodes and scores identically, apart
/// from calibration, which `zq_index_calibrated` reports.
export fn zq_index_config(index: ?*const Index, out: ?*ZqConfig) c_int {
    const ix = index orelse return ZQ_ERR_INVALID;
    const slot = out orelse return ZQ_ERR_INVALID;
    const p = ix.inner.params;
    // A file written through the Zig API can carry options zq_config has no field for
    // (a dense rotation, the QJL correction, the exact scan). A config that silently
    // dropped them would build a different index, so say so instead.
    const expressible: zq.flat.Params = .{
        .dim = p.dim,
        .bits = p.bits,
        .metric = p.metric,
        .seed = p.seed,
        .residency = p.residency,
    };
    if (!std.meta.eql(p, expressible)) return ZQ_ERR_UNSUPPORTED;
    slot.* = .{
        .dim = p.dim,
        .bits = p.bits,
        .metric = switch (p.metric) {
            .inner_product => ZQ_METRIC_INNER_PRODUCT,
            .cosine => ZQ_METRIC_COSINE,
            .l2 => ZQ_METRIC_L2,
        },
        .seed = p.seed,
        .compact = @intFromBool(p.residency == .compact),
    };
    return ZQ_OK;
}

export fn zq_index_calibrated(index: ?*const Index) c_int {
    const ix = index orelse return 0;
    return @intFromBool(ix.inner.calibrated);
}

/// `threads` of 0 or 1 searches on the calling thread. Greater than 1 spreads queries
/// across that many workers, which is only worthwhile for batches of at least that size.
export fn zq_searcher_create(
    index: ?*Index,
    batch: usize,
    k: usize,
    threads: usize,
    out: ?**Searcher,
) c_int {
    const ix = index orelse return ZQ_ERR_INVALID;
    const slot = out orelse return ZQ_ERR_INVALID;
    if (batch == 0 or k == 0) return ZQ_ERR_INVALID;
    if (batch > zq.flat.max_batch) return ZQ_ERR_UNSUPPORTED;
    if (threads > ZQ_MAX_THREADS) return ZQ_ERR_INVALID;
    // The searcher holds `threads * batch * k` results. Unchecked, a large k wrapped
    // that product to a small allocation and the search wrote past it.
    const workers = @max(threads, 1);
    const per_batch = std.math.mul(usize, batch, k) catch return ZQ_ERR_INVALID;
    _ = std.math.mul(usize, per_batch, workers) catch return ZQ_ERR_INVALID;

    const s = allocator.create(Searcher) catch return ZQ_ERR_ALLOC;
    errdefer allocator.destroy(s);

    s.batch = zq.flat.FlatIndex.BatchSearcher.init(allocator, ix.inner, batch, k) catch {
        allocator.destroy(s);
        return ZQ_ERR_ALLOC;
    };
    s.scorer = zq.flat.FlatIndex.Searcher.init(allocator, ix.inner, 1) catch {
        s.batch.deinit();
        allocator.destroy(s);
        return ZQ_ERR_ALLOC;
    };
    s.owner = ix;
    s.k = k;
    s.parallel = null;
    s.capacity = batch;
    if (threads > 1) {
        s.parallel = zq.flat.FlatIndex.ParallelSearcher.init(allocator, ix.inner, threads, batch, k) catch {
            s.scorer.deinit();
            s.batch.deinit();
            allocator.destroy(s);
            return ZQ_ERR_ALLOC;
        };
        s.capacity = threads * batch;
    }
    slot.* = s;
    return ZQ_OK;
}

export fn zq_searcher_free(searcher: ?*Searcher) void {
    const s = searcher orelse return;
    if (s.parallel) |*p| p.deinit();
    s.scorer.deinit();
    s.batch.deinit();
    allocator.destroy(s);
}

/// Largest `nq` a single `zq_search` call accepts for this searcher.
export fn zq_searcher_capacity(searcher: ?*const Searcher) usize {
    const s = searcher orelse return 0;
    return s.capacity;
}

/// Validate a search call and run it. Returns the searcher's results, `s.k` per query,
/// of which the first `@min(s.k, count)` in each row are real.
fn runSearch(ix: *Index, s: *Searcher, q: [*]const f32, nq: usize) error{ Invalid, Alloc }![]const zq.flat.Entry {
    if (s.owner != ix) return error.Invalid;
    if (nq > s.capacity) return error.Invalid;
    const batch = rowsOf(q, nq, ix.inner.dim()) orelse return error.Invalid;
    // A NaN query scores NaN against everything and the collector's order is then
    // meaningless; refuse it rather than return confident garbage.
    if (!zq.flat.allFinite(batch)) return error.Invalid;
    return if (s.parallel) |*p|
        ix.inner.searchBatchParallel(batch, p) catch return error.Alloc
    else
        ix.inner.searchBatch(batch, &s.batch);
}

fn statusOf(e: error{ Invalid, Alloc }) c_int {
    return switch (e) {
        error.Invalid => ZQ_ERR_INVALID,
        error.Alloc => ZQ_ERR_ALLOC,
    };
}

/// Write `k` results per row, padding past `found` with id UINT32_MAX and score -inf.
fn writeRow(row: []const zq.flat.Entry, k: usize, ids: [*]u32, scores: [*]f32) void {
    for (0..k) |j| {
        if (j < row.len) {
            ids[j] = row[j].id;
            scores[j] = row[j].score;
        } else {
            ids[j] = std.math.maxInt(u32);
            scores[j] = -std.math.inf(f32);
        }
    }
}

/// Writes `nq * k` ids and scores, query-major. Ids are indices in insertion order.
/// Fewer than `k` results exist only when the index holds fewer than `k` vectors; the
/// tail is then filled with id `UINT32_MAX` and score `-inf`.
export fn zq_search(
    index: ?*Index,
    searcher: ?*Searcher,
    queries: ?[*]const f32,
    nq: usize,
    out_ids: ?[*]u32,
    out_scores: ?[*]f32,
) c_int {
    const ix = index orelse return ZQ_ERR_INVALID;
    const s = searcher orelse return ZQ_ERR_INVALID;
    const q = queries orelse return ZQ_ERR_INVALID;
    const ids = out_ids orelse return ZQ_ERR_INVALID;
    const scores = out_scores orelse return ZQ_ERR_INVALID;
    if (nq == 0) return if (s.owner == ix) ZQ_OK else ZQ_ERR_INVALID;

    const results = runSearch(ix, s, q, nq) catch |e| return statusOf(e);
    const k = s.k;
    const found = @min(k, ix.inner.count());
    for (0..nq) |i| writeRow(results[i * k ..][0..found], k, ids + i * k, scores + i * k);
    return ZQ_OK;
}

/// Search, then rescore the candidates exactly against the caller's originals.
///
/// The searcher's `k` is the number of candidates fetched; `k` here, at most that, is
/// how many come back. `originals` holds `n_originals` row-major vectors indexed by id,
/// and must cover every id in the index.
export fn zq_search_rerank(
    index: ?*Index,
    searcher: ?*Searcher,
    queries: ?[*]const f32,
    nq: usize,
    originals: ?[*]const f32,
    n_originals: usize,
    k: usize,
    out_ids: ?[*]u32,
    out_scores: ?[*]f32,
) c_int {
    const ix = index orelse return ZQ_ERR_INVALID;
    const s = searcher orelse return ZQ_ERR_INVALID;
    const q = queries orelse return ZQ_ERR_INVALID;
    const orig = originals orelse return ZQ_ERR_INVALID;
    const ids = out_ids orelse return ZQ_ERR_INVALID;
    const scores = out_scores orelse return ZQ_ERR_INVALID;
    if (k == 0 or k > s.k) return ZQ_ERR_INVALID;
    const count = ix.inner.count();
    if (n_originals < count) return ZQ_ERR_INVALID;
    const d = ix.inner.dim();
    const corpus = rowsOf(orig, n_originals, d) orelse return ZQ_ERR_INVALID;
    if (nq == 0) return if (s.owner == ix) ZQ_OK else ZQ_ERR_INVALID;

    const results = runSearch(ix, s, q, nq) catch |e| return statusOf(e);
    const found = @min(s.k, count);
    for (0..nq) |i| {
        // The searcher's result buffer is scratch until the next search, so reranking
        // in place is safe.
        const row: []zq.flat.Entry = @constCast(results[i * s.k ..][0..found]);
        const best = zq.rerank.rerank(ix.inner.metric, q[i * d ..][0..d], corpus, d, row, k);
        writeRow(best, k, ids + i * k, scores + i * k);
    }
    return ZQ_OK;
}

/// Score specific stored vectors against one query, with the same estimate search
/// reports: `scores[j]` for `ids[j]`. For ranking a candidate set chosen elsewhere.
export fn zq_score_ids(
    index: ?*Index,
    searcher: ?*Searcher,
    query: ?[*]const f32,
    ids: ?[*]const u32,
    n: usize,
    out_scores: ?[*]f32,
) c_int {
    const ix = index orelse return ZQ_ERR_INVALID;
    const s = searcher orelse return ZQ_ERR_INVALID;
    const q = query orelse return ZQ_ERR_INVALID;
    const id_ptr = ids orelse return ZQ_ERR_INVALID;
    const scores = out_scores orelse return ZQ_ERR_INVALID;
    if (s.owner != ix) return ZQ_ERR_INVALID;
    if (n == 0) return ZQ_OK;
    const row = q[0..ix.inner.dim()];
    if (!zq.flat.allFinite(row)) return ZQ_ERR_INVALID;
    ix.inner.scoreIds(row, id_ptr[0..n], scores[0..n], &s.scorer) catch return ZQ_ERR_INVALID;
    return ZQ_OK;
}

// ── Codec: encode and decode without an index ───────────────────────────────────
//
// The index answers "which stored vectors best match this query". A KV cache asks
// something else: store these vectors compactly and give them back. Attention needs every
// score rather than the top few, and it needs them accurately in absolute terms, so the
// index's per-vector correction — which is fitted to preserve *ranking* — is the wrong
// tool and measured 7x worse than plain reconstruction (docs/notes.md).
//
// Codes are bit-packed, so `zq_codec_code_bytes` is the storage actually consumed. The
// unpacked form is one byte per coordinate and would silently cost twice the memory.

const Codec = struct {
    mse: zq.mse.Mse,
    workspace: zq.mse.Workspace,
    layout: zq.packing.Layout,
    scratch: []u8,
};

pub const ZqCodecConfig = extern struct {
    dim: u32,
    /// 1..6. Unlike the index this is the codebook width directly: there is no residual
    /// sketch to reserve a bit for, so even one bit leaves a codebook.
    bits: u8,
    seed: u64,
};

export fn zq_codec_create(config: ?*const ZqCodecConfig, out: ?**Codec) c_int {
    if (zq_cpu_supported() == 0) return ZQ_ERR_CPU;
    const cfg = config orelse return ZQ_ERR_INVALID;
    const slot = out orelse return ZQ_ERR_INVALID;
    if (cfg.dim == 0 or cfg.dim > ZQ_MAX_DIM) return ZQ_ERR_INVALID;
    if (cfg.bits < 1 or cfg.bits > 6) return ZQ_ERR_INVALID;

    const codec = allocator.create(Codec) catch return ZQ_ERR_ALLOC;
    codec.mse = zq.mse.Mse.init(allocator, .{
        .dim = cfg.dim,
        .bits = @intCast(cfg.bits),
        .seed = cfg.seed,
    }) catch {
        allocator.destroy(codec);
        return ZQ_ERR_ALLOC;
    };
    codec.workspace = zq.mse.Workspace.init(allocator, codec.mse) catch {
        codec.mse.deinit();
        allocator.destroy(codec);
        return ZQ_ERR_ALLOC;
    };
    codec.layout = zq.packing.Layout.init(codec.mse.padded, @intCast(cfg.bits));
    codec.scratch = allocator.alloc(u8, codec.mse.codeLen()) catch {
        codec.workspace.deinit();
        codec.mse.deinit();
        allocator.destroy(codec);
        return ZQ_ERR_ALLOC;
    };
    slot.* = codec;
    return ZQ_OK;
}

export fn zq_codec_free(codec: ?*Codec) void {
    const c = codec orelse return;
    allocator.free(c.scratch);
    c.workspace.deinit();
    c.mse.deinit();
    allocator.destroy(c);
}

/// Packed bytes of code per vector. Norms are stored separately, one float each.
export fn zq_codec_code_bytes(codec: ?*const Codec) usize {
    const c = codec orelse return 0;
    return c.layout.codeBytes();
}

export fn zq_codec_dim(codec: ?*const Codec) u32 {
    const c = codec orelse return 0;
    return c.mse.dim;
}

/// Encode `n` row-major vectors. `codes` holds `n * zq_codec_code_bytes()` bytes and
/// `norms` holds `n` floats; both are caller-owned and both are needed to decode.
export fn zq_codec_encode(
    codec: ?*Codec,
    rows: ?[*]const f32,
    n: usize,
    codes: ?[*]u8,
    norms: ?[*]f32,
) c_int {
    const c = codec orelse return ZQ_ERR_INVALID;
    const src = rows orelse return ZQ_ERR_INVALID;
    const dst = codes orelse return ZQ_ERR_INVALID;
    const nrm = norms orelse return ZQ_ERR_INVALID;
    if (n == 0) return ZQ_OK;

    const d = c.mse.dim;
    const stride = c.layout.codeBytes();
    const input = rowsOf(src, n, d) orelse return ZQ_ERR_INVALID;
    _ = std.math.mul(usize, n, stride) catch return ZQ_ERR_INVALID;
    if (!zq.flat.allFinite(input)) return ZQ_ERR_INVALID;
    for (0..n) |i| {
        nrm[i] = c.mse.encode(src[i * d ..][0..d], c.scratch, &c.workspace);
        c.layout.pack(c.scratch, dst[i * stride ..][0..stride]);
    }
    return ZQ_OK;
}

/// Decode `n` vectors into `rows`, which holds `n * dim` floats.
export fn zq_codec_decode(
    codec: ?*Codec,
    codes: ?[*]const u8,
    norms: ?[*]const f32,
    n: usize,
    rows: ?[*]f32,
) c_int {
    const c = codec orelse return ZQ_ERR_INVALID;
    const src = codes orelse return ZQ_ERR_INVALID;
    const nrm = norms orelse return ZQ_ERR_INVALID;
    const dst = rows orelse return ZQ_ERR_INVALID;
    if (n == 0) return ZQ_OK;

    const d = c.mse.dim;
    const stride = c.layout.codeBytes();
    _ = std.math.mul(usize, n, d) catch return ZQ_ERR_INVALID;
    _ = std.math.mul(usize, n, stride) catch return ZQ_ERR_INVALID;
    for (0..n) |i| {
        c.layout.unpack(src[i * stride ..][0..stride], c.scratch);
        c.mse.decode(c.scratch, nrm[i], dst[i * d ..][0..d], &c.workspace);
    }
    return ZQ_OK;
}
