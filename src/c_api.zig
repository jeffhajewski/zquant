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

pub const ZQ_METRIC_INNER_PRODUCT: c_int = 0;
pub const ZQ_METRIC_COSINE: c_int = 1;
pub const ZQ_METRIC_L2: c_int = 2;

pub const ZqConfig = extern struct {
    dim: u32,
    /// Total bit budget per coordinate, 2..6. The scalar codebook uses `bits - 1`.
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
    batch: zq.flat.FlatIndex.BatchSearcher,
    parallel: ?zq.flat.FlatIndex.ParallelSearcher,
    k: usize,
    capacity: usize,
};

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

export fn zq_status_string(status: c_int) [*:0]const u8 {
    return switch (status) {
        ZQ_OK => "ok",
        ZQ_ERR_ALLOC => "out of memory",
        ZQ_ERR_INVALID => "invalid argument",
        ZQ_ERR_STATE => "operation not valid in this state",
        ZQ_ERR_UNSUPPORTED => "unsupported configuration",
        else => "unknown status",
    };
}

export fn zq_index_create(config: ?*const ZqConfig, out: ?**Index) c_int {
    const cfg = config orelse return ZQ_ERR_INVALID;
    const slot = out orelse return ZQ_ERR_INVALID;
    if (cfg.dim == 0 or cfg.bits < 1 or cfg.bits > 6) return ZQ_ERR_INVALID;
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
    const d = ix.inner.dim();
    ix.inner.calibrate(data[0 .. n * d]) catch |e| return switch (e) {
        error.OutOfMemory => ZQ_ERR_ALLOC,
        error.IndexNotEmpty => ZQ_ERR_STATE,
        error.EmptySample => ZQ_ERR_INVALID,
    };
    return ZQ_OK;
}

export fn zq_index_add(index: ?*Index, rows: ?[*]const f32, n: usize) c_int {
    const ix = index orelse return ZQ_ERR_INVALID;
    const data = rows orelse return ZQ_ERR_INVALID;
    if (n == 0) return ZQ_OK;
    const d = ix.inner.dim();
    ix.inner.addBatch(data[0 .. n * d]) catch return ZQ_ERR_ALLOC;
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

    const s = allocator.create(Searcher) catch return ZQ_ERR_ALLOC;
    errdefer allocator.destroy(s);

    s.batch = zq.flat.FlatIndex.BatchSearcher.init(allocator, ix.inner, batch, k) catch {
        allocator.destroy(s);
        return ZQ_ERR_ALLOC;
    };
    s.k = k;
    s.parallel = null;
    s.capacity = batch;
    if (threads > 1) {
        s.parallel = zq.flat.FlatIndex.ParallelSearcher.init(allocator, ix.inner, threads, batch, k) catch {
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
    s.batch.deinit();
    allocator.destroy(s);
}

/// Largest `nq` a single `zq_search` call accepts for this searcher.
export fn zq_searcher_capacity(searcher: ?*const Searcher) usize {
    const s = searcher orelse return 0;
    return s.capacity;
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
    if (nq == 0) return ZQ_OK;
    if (nq > s.capacity) return ZQ_ERR_INVALID;

    const d = ix.inner.dim();
    const k = s.k;
    const found = @min(k, ix.inner.count());

    const results = if (s.parallel) |*p|
        ix.inner.searchBatchParallel(q[0 .. nq * d], p) catch return ZQ_ERR_ALLOC
    else
        ix.inner.searchBatch(q[0 .. nq * d], &s.batch);

    for (0..nq) |i| {
        const row = results[i * k ..];
        for (0..k) |j| {
            if (j < found) {
                ids[i * k + j] = row[j].id;
                scores[i * k + j] = row[j].score;
            } else {
                ids[i * k + j] = std.math.maxInt(u32);
                scores[i * k + j] = -std.math.inf(f32);
            }
        }
    }
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
    /// 2..6. Unlike the index this is the codebook width directly: there is no residual
    /// sketch to reserve a bit for.
    bits: u8,
    seed: u64,
};

export fn zq_codec_create(config: ?*const ZqCodecConfig, out: ?**Codec) c_int {
    const cfg = config orelse return ZQ_ERR_INVALID;
    const slot = out orelse return ZQ_ERR_INVALID;
    if (cfg.dim == 0 or cfg.bits < 1 or cfg.bits > 6) return ZQ_ERR_INVALID;

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
    for (0..n) |i| {
        c.layout.unpack(src[i * stride ..][0..stride], c.scratch);
        c.mse.decode(c.scratch, nrm[i], dst[i * d ..][0..d], &c.workspace);
    }
    return ZQ_OK;
}
