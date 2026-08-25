# zquant — Design & Implementation Plan

TurboQuant (Zandieh, Daliri, Hadian, Mirrokni — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026)
implemented in Zig, with first-class clients in Zig, Python, JS/TS, and Go.

Status: **P0 (reference core) complete**; P1 next. Zig toolchain pinned to **0.16.0**.

---

## 1. What the algorithm actually is

Two quantizers. Both are **data-oblivious** — no training, no codebook fitting, no data pass.
That property is the whole point: indexing time is essentially the cost of a rotation.

### 1.1 `TurboQuant_mse(b)`

```
setup:  Π  — random rotation in R^d
        c1..c_{2^b} — Lloyd-Max centroids for density f_X

Quant(x):    y   = Π · (x / ||x||)          ; store ||x|| separately
             idx_j = argmin_k |y_j - c_k|    ; b bits per coordinate
DeQuant(idx): ỹ_j = c_{idx_j}
             x̃   = ||x|| · Πᵀ · ỹ
```

The load-bearing fact: for `x` on the unit sphere, each coordinate of `Πx` is distributed as

```
f_X(t) = Γ(d/2) / (√π · Γ((d−1)/2)) · (1 − t²)^((d−3)/2),   t ∈ [−1, 1]
```

which converges to `N(0, 1/d)`, and distinct coordinates are near-independent in high `d`.
So the intractable d-dimensional VQ problem collapses to **one scalar quantizer applied d times**.

Centroids solve the continuous 1-D k-means problem
`C(f_X, b) = min Σ_i ∫ |t − c_i|² f_X(t) dt` over Voronoi cells. Solved **once, offline**, per `(b, d)`.

Guarantee: `D_mse ≤ (√3·π/2) · 4^−b`. The constant is 2.7207, and since the Shannon lower
bound for this problem is exactly `4^−b`, that constant *is* the "≈2.7 factor" the abstract
quotes. It also falls out of Panter-Dite independently: `(1/12)·(∫f^(1/3))³ = (1/12)·6√3·π`
for a unit Gaussian. (The PDF renders this as a stacked `3π / 2d` under a radical, which reads
naturally but wrongly as `√(3π)/2 = 1.535` — a value the true optimal quantizer violates at
every bit-width.)

### 1.2 `TurboQuant_prod(b)` — unbiased inner products

MSE-optimal quantizers are *biased* for inner products (at b=1 the bias is exactly `2/π`).
Fix: spend `b−1` bits on MSE, spend the last bit on a QJL sketch of the residual.

```
setup:  mse quantizer at bit-width b−1
        S ∈ R^{d×d}, S_ij ~ N(0,1)

Quant(x):   idx = Quant_mse(x)
            r   = x − DeQuant_mse(idx)
            qjl = sign(S · r)                 ; 1 bit per coordinate
            γ   = ||r||₂
            → (idx, qjl, γ)

DeQuant(idx, qjl, γ):
            x̃ = DeQuant_mse(idx) + (√(π/2) / d) · γ · Sᵀ · qjl
```

`E[⟨y, x̃⟩] = ⟨y, x⟩` exactly. The inner-product distortion follows from the QJL variance
bound as `D_prod = (π/2)·D_mse(b−1)·‖y‖²/d`, reproducing the paper's quoted
`{1.57, 0.56, 0.18, 0.047}/d` for b=1..4.

**We deviate here, and it is an improvement, not just a cheaper approximation.** We use an
*orthogonal* `S'` (a second RHT) rather than a Gaussian one. Two consequences:

1. *An exact constant instead of an asymptotic one.* Rows of a Haar-orthogonal matrix are uniform
   unit vectors, and for such a row `r` and unit `x`, `E[sign(rᵀx)·(rᵀy)] = c_m·⟨x,y⟩` **exactly** —
   decompose `y` along `x` plus a perpendicular part, whose contribution vanishes by symmetry. And
   `c_m = E|rᵀx|` is just the mean absolute value of the sphere-coordinate density, available in
   closed form from `density.moment`. The paper's `√(π/2)/√m` is the large-`m` limit of `1/(m·c_m)`.
   Using the exact value matters most at small `m` — i.e. the KV path.
2. *Roughly 2.7× lower distortion.* Orthonormal rows fix `Σᵢ(rᵢᵀq)² = ‖q‖²` exactly rather than
   letting it fluctuate, and the removed variance shows up directly in the estimator.

Measured, with the paper's own Gaussian construction implemented in the same harness on the same
vectors: Gaussian sketch gives `Var·m = 1.55` (matching the published 1.57, which validates the
harness); orthogonal gives `0.56`. Per bit-width our `D_prod·d` is `{0.567, 0.207, 0.068, 0.020}`
against the paper's `{1.57, 0.56, 0.18, 0.047}`. Tests assert both the regression values and the
strict inequality against the published ones.

The measured `0.566 ± 0.003` sits near `π/2 − 1 = 0.5708` — suggestive of an exact constant, but
1.5σ off, so it is not asserted as one.

**Bit accounting.** "b bits" for `prod` = `(b−1)` code bits + 1 sketch bit per coordinate,
plus two f16 scalars per vector (`‖x‖`, `γ`) — 32 bits amortized over d, i.e. noise at d≥256.

### 1.3 Two structural observations that drive the implementation

**(a) Stay in the rotated basis.** `r = Πᵀ(y − ỹ)`. Define `S' = S·Πᵀ`; by rotational invariance of
the Gaussian, `S'` is still i.i.d. Gaussian. So `qjl = sign(S' · u)` where `u = y − ỹ` is the
*rotated-domain* residual. And the estimator becomes, with `p = Π·q` computed **once per query**:

```
⟨q, x̃⟩  =  ⟨p, ỹ⟩  +  (√(π/2)/d) · γ · ⟨S'p, qjl⟩
```

Nothing on the database side is ever rotated back. Encode: one rotation per vector, at insert.
Search: one rotation + one sketch per *query*, then pure integer/FMA work over the corpus.

**(b) For b ≤ 4 the codebook is ≤16 scalars — one SIMD shuffle table.**
`vpshufb` (AVX2) / `tbl` (NEON) / `i8x16.swizzle` (WASM) dequantizes 32 nibbles in one instruction.
This is what makes a FastScan-style blocked layout the right core kernel (§4.2).

### 1.4 Deviation from the paper we intend to make (and must validate)

The paper generates `Π` by QR on a Gaussian matrix: `O(d²)` per vector, and `O(d²)` storage.
That is fine on a GPU with a batched GEMM; it is the wrong choice for a CPU library.

We use a **Randomized Hadamard Transform**, `Π = H·D₂·H·D₁·H·D₀` (3 rounds, `D_i` random ±1
diagonals, `H` normalized so it is orthogonal): `O(d log d)`, in-cache, branch-free, and specified
by `3d` sign bits instead of `d²` floats. Same for `S'` — a second RHT instance under a different
RNG purpose, which is sub-Gaussian rather than Gaussian.

*Implemented without the inter-round permutation this section originally called for.* FWHT already
makes every output coordinate depend on every input coordinate, so a permutation adds little that
another sign-flip round does not, and omitting it keeps `apply` allocation-free and thread-safe
(permuting in place otherwise needs scratch). Justified by measurement, not assertion: the RHT
reproduces the sphere coordinate density's exact fourth moment `3/(d(d+2))` — distinguishably not
the Gaussian `3/d²` — and matches the dense Haar reference to within 5%. If that ever regresses,
add rounds before adding permutations.

Rounds are 3 because 1 is provably not enough: a single `H·D` maps a standard basis vector to
`±1/√d` in *every* coordinate — identical magnitudes, the exact opposite of the Beta-distributed
spread the scalar quantizer is built for. Axis-aligned and one-hot-ish vectors do occur in real
corpora, so this is a real input, not a contrived one. There is a regression test for it.

This is standard practice (QuIP#, KIVI, ScaNN all do it) but it is *an approximation of the paper*,
so it is gated on a measured test: empirical `D_mse` / `D_prod` from the RHT path must match the
dense-QR path within the paper's stated per-bit-width constants (§7.2). The dense path stays in the
tree permanently as the reference oracle, not just as scaffolding.

**Non-power-of-two `d`.** Zero-padding `d=768 → 1024` wastes 33% of the bit budget. Instead
factor `d = m · 2^a` and use block-FWHT over `m` blocks of size `2^a`, with a random permutation
between rounds to mix across blocks (`768 = 3·256` works cleanly). Fall back to padding only when
`d` has an awkward factorization; the pad width is recorded in the header so the density parameter
uses the *padded* dimension.

---

## 2. Scope and non-goals

zquant has **two target workloads** sharing one core. They stress it differently, and the
difference is not cosmetic — see §8.

| | Vector search | KV cache |
|---|---|---|
| `d` | 200–3072 (embedding dim) | **64–128** (head_dim) |
| `n` per query | 10⁶–10⁹ corpus vectors | 10³–10⁵ cached tokens |
| Shape of work | throughput: one query vs. a huge static corpus | latency: append one vector, then scan a small growing one |
| Density model | `N(0, 1/d)` limit is fine | **exact Beta required** — d=128 is not the asymptotic regime |
| Bit widths | integer, 2–4 | fractional (2.5, 3.5) via outlier channel splitting |
| Numerics | f32 | f16 / bf16 |

**In scope (v1)**
- The two quantizers, exactly and correctly, with the paper's distortion numbers reproduced.
- A flat (exhaustive-scan) index with asymmetric distance computation and optional rerank.
- IP / cosine / L2 queries (L2 reduces to IP via `‖q−x‖² = ‖q‖² + ‖x‖² − 2⟨q,x⟩`).
- KV-cache path: streaming online encode, outlier channel splitting, attention kernels (§8).
- Stable C ABI, mmap-able on-disk format, Python + JS/TS clients.

**Deferred**
- IVF / graph (HNSW) indexes — layered on top, not entangled with the quantizer.
- GPU backends.
- Entropy coding of code indices. The paper measures ~5% gain at b=4 and explicitly declines it
  for speed; we agree.

**Explicit non-goal:** being a vector database. zquant is a quantizer plus a fast scan.
Storage, filtering, replication, and persistence policy belong to whatever embeds it.

---

## 3. Repository layout

```
build.zig                 # pinned Zig 0.15.2
build.zig.zon
src/
  root.zig                # public Zig API — the only thing downstream Zig code imports
  math/
    rng.zig               # Philox4x32 counter-based RNG: seekable, reproducible, no state to store
    rotation.zig          # RHT (block-FWHT + sign flips + permutation); dense-QR reference
    density.zig           # f_X for exact d; N(0,1/d) limit
    lloyd_max.zig         # continuous 1-D k-means solver (offline + comptime)
  quant/
    codebook.zig          # generated tables: centroids + midpoint thresholds, per (b, d-class)
    mse.zig               # TurboQuant_mse
    prod.zig              # TurboQuant_prod + QJL
    packing.zig           # bit packing 1/2/3/4b; blocked/interleaved layouts
  simd/
    encode.zig            # threshold-tree encoder
    scan.zig              # blocked ADC scan kernels (shuffle-LUT and int8-dot variants)
    sketch.zig            # sign-bit dot product
    topk.zig              # threshold-gated top-k
    attn.zig              # value-side accumulate kernel (KV path, §8.2)
    dispatch.zig          # runtime ISA detection → function-pointer table
  index/
    flat.zig
    rerank.zig
  kv/
    cache.zig             # streaming append, staging buffer → 32-vector blocks
    channels.zig          # outlier/regular channel split, fractional bit widths
  io/
    format.zig            # versioned, mmap-able, little-endian
  c_api.zig               # stable C ABI (opaque handles, error codes)
include/zquant.h
bindings/
  python/                 # abi3 extension built with the Zig toolchain
    zquant/kv/            # transformers-compatible Cache impl (§8.3)
  js/                     # node (N-API) + browser (wasm32 + simd128), one TS surface
  go/                     # cgo (deferred)
bench/                    # recall@k, QPS, encode throughput; faiss/PQ + RaBitQ baselines
tools/
  gen_codebooks.zig       # emits the comptime tables
  fetch_datasets.sh       # GloVe-200, DBpedia-OpenAI-1536/3072 (paper's datasets)
  calibrate_outliers.py   # offline per-(layer, head) outlier channel masks (§8.2)
tests/
  golden/                 # fixed-seed cross-language conformance vectors
docs/DESIGN.md            # this file
```

---

## 4. Performance architecture

### 4.1 Encode path

Per vector: normalize → RHT → threshold-encode → (prod only) residual, RHT-sketch, sign-pack.

The encoder is the one place a naive implementation loses badly. `argmin_k |y_j − c_k|` over sorted
centroids is equivalent to counting how many midpoint thresholds `y_j` exceeds. Two candidate
kernels, both branch-free over `@Vector(N, f32)`:

- **Linear compare-sum** — `idx = Σ_k (y > t_k)`. `2^b − 1` compare+add pairs. Best for b ≤ 2.
- **Threshold binary search** — `b` rounds of compare + select. `~3b` ops/element. Best for b ≥ 3.

Both are implemented; `dispatch.zig` picks per `b` from measured numbers, not from intuition.

### 4.2 Scan path — row-major, factorized LUT, integer dot product

> **Revised during P1 implementation.** This section originally specified a 32-vector
> dimension-major (FastScan) block. That was wrong for this quantizer — see below.

The corpus is stored **row-major**: each vector's codes packed contiguously at `b` bits per
coordinate, padded to a 16-byte stride.

**Why not FastScan.** In product quantization each subspace has its own *query-dependent* lookup
table `LUT_m[k] = ⟨q_m, centroid_{m,k}⟩`, and applying it needs every vector's code for subspace
`m` at once — which is exactly what a dimension-major block delivers. TurboQuant's codebook is
*scalar*, so the equivalent table **factorizes**:

```
LUT_j[k] = p_j · c[k]
```

`c[]` is a single 16-entry table at b≤4: query-independent, shared by every dimension, and
resident in one SIMD register for the whole scan. Only the scalar `p_j` depends on the query.
FastScan's entire justification evaporates, and the scan becomes a per-vector dot product with a
reduction along `j` — the exact shape `SDOT` (ARM) and VNNI (x86) exist to accelerate. This is
also why turbovec reports `SDOT`/`SMMLA`, which never fit a dimension-major layout.

**The kernel**, per vector at d=1024, b=4 (512 B of codes):

```
for chunk in 0..32:                       # 16 bytes = 32 nibbles = 32 dims
    codes = load16(v + chunk*16)
    lo    = codes & 0x0F                  # even dimensions
    hi    = codes >> 4                    # odd dimensions
    acc   = sdot(acc, tbl(c8, lo), q_even[chunk])
    acc   = sdot(acc, tbl(c8, hi), q_odd[chunk])
horizontal_add(acc)
```

6 ops per 32 dimensions ≈ **0.19 ops/dimension**, ~192 ops/vector. At ~4 IPC that is ~48
cycles/vector, i.e. **~37 GB/s of code traffic** — genuinely memory-bound, which is where we want
to sit. The dimension-major form needs an int8→f32 widen per dimension and costs ~3× more,
leaving the scan compute-bound.

Because byte `i` holds dimensions `2i` and `2i+1`, the query is de-interleaved into even and odd
streams once per query (not per vector) to line up with the nibble split.

**Precision variants:**
- **int8** — query uniformly quantized per-query to int8, centroids stored as int8. Fastest;
  adds a bounded, measurable query-side error. Opt-in, never a silent default.
- **exact f32** — four `tbl` lookups over the byte-planes of the 16 f32 centroids reconstruct
  exact f32 values from codes, at more ops but no approximation.

**Table lookup is per-architecture inline assembly** (`simd/shuffle.zig`), as this section
originally specified. Under Zig 0.15 it was portable source that LLVM pattern-matched; 0.16 removed
runtime vector indexing and the construct no longer compiles. Staging through an array compiles but
loses the instruction, so the assembly is back — with a scalar fallback for targets without one.

**Four bits is a hard ceiling on the vectorized path.** The shuffle instructions index a 16-byte
register, so the codebook must fit 16 levels. Wider codebooks fall back to the exact f32 scan.

**Bit-widths that straddle bytes use a bit-plane layout.** 3-bit codes cannot be shift-and-masked,
so they are stored as one plane per code bit in groups of 16 and reassembled by weight. Costs the
same `bits` per coordinate; roughly 1.8× slower than nibbles, against 25× for the scalar fallback
it replaced.

### 4.3 The QJL term

`⟨S'p, qjl⟩` with `qjl ∈ {±1}^d` stored as a bitmap. Sign application without branches:
expand 8 mask bits → 256-bit sign mask (shuffle + and + cmpeq), XOR into the f32 sign bit, accumulate.
~1 op per coordinate. Algebraic alternative `2·Σ_{bit=1} w_j − Σ w_j` precomputes `Σw` per query and
turns it into masked adds; benchmark both.

### 4.4 Top-k

Threshold-gated: keep the current k-th best in a broadcast register, compare the 32 block scores,
`movemask`, and only touch the scalar heap when the mask is nonzero. For k ≪ n the heap is
effectively never touched after the first few blocks.

### 4.5 Threading

Core library is **single-threaded and allocator-explicit**. Parallelism is an injected
`std.Thread.Pool` at the index layer, sharding blocks across workers with per-worker top-k merged at
the end. Bindings own pool lifetime. No hidden global state, no implicit thread spawning — this is
what makes the library safe to embed inside someone else's runtime (and it is why the C ABI takes an
explicit allocator and executor).

### 4.6 Determinism

`Π` and `S'` are never materialized or stored. A Philox4x32 counter-based RNG derives sign flips and
permutations from `(seed, round, index)` on demand, so:
- an index is fully described by `(seed, d, b, objective)` plus the codes;
- every binding and every platform reproduces byte-identical codes from the same seed;
- serialized indexes are portable without shipping a `d×d` matrix.

---

## 5. Public API surface

The Zig API is the source of truth; the C ABI is a mechanical projection of it; every other language
is a thin ergonomic layer over the C ABI. No language gets a feature the others cannot express.

### 5.1 Zig

```zig
const zq = @import("zquant");

const q = try zq.Quantizer.init(alloc, .{
    .dim       = 1536,
    .bits      = 4,
    .objective = .inner_product,   // or .mse
    .seed      = 0x5EED,
});
defer q.deinit();

var idx = try zq.FlatIndex.init(alloc, q, .{ .metric = .cosine });
defer idx.deinit();

try idx.addBatch(vectors);                       // []const f32, row-major, n*d
const hits = try idx.search(alloc, query, 10, .{ .rerank = 0 });

try idx.save("corpus.zq");
var loaded = try zq.FlatIndex.open(alloc, "corpus.zq");   // mmap, zero-copy
```

Standalone quantizer use (KV-cache-style, no index) is a first-class path:
`q.encodeInto(dst, x)` / `q.decodeInto(dst, codes)` / `q.dot(codes, query_state)`.

### 5.2 Python

Built as an **abi3 extension compiled by the Zig toolchain** — one source tree, cross-compiled wheels
for linux x86_64/aarch64, macOS universal2, and Windows, without a C compiler on the build host or a
matrix of per-version builds. NumPy in/out via the buffer protocol, zero-copy, GIL released around
every kernel.

```python
import numpy as np, zquant

q = zquant.Quantizer(dim=1536, bits=4, objective="ip", seed=0x5EED)

index = zquant.FlatIndex(q, metric="cosine")
index.add(X)                                  # (n, 1536) float32, zero-copy
scores, ids = index.search(Q, k=10)           # (m, 10) each

index.save("corpus.zq")
index = zquant.FlatIndex.open("corpus.zq")    # mmap

codes = q.encode(X)                           # bare quantizer, no index
X_hat = q.decode(codes)
```

### 5.3 JavaScript / TypeScript

One TS surface, two backends selected at load: **N-API** (prebuilt per platform, no node-gyp on
install) for Node/Bun/Deno, **wasm32 + simd128** for browsers. Zig targets both natively.
Zero-copy via `Float32Array` views over wasm memory or N-API external buffers.

```ts
import { Quantizer, FlatIndex } from "zquant";

const q = await Quantizer.create({ dim: 1536, bits: 4, objective: "ip", seed: 0x5EED });
const index = new FlatIndex(q, { metric: "cosine" });

index.add(vectors);                                  // Float32Array, n*dim
const { ids, scores } = index.search(query, 10);

const bytes = index.serialize();                     // Uint8Array
const restored = FlatIndex.deserialize(q, bytes);
```

Async only where it must be (`create` awaits wasm instantiation); everything else synchronous,
because the kernels are microseconds and a promise per search would dominate the cost.

### 5.4 Go (deferred to P6)

cgo over the same C ABI. `purego`/dlopen is evaluated as a cgo-free alternative once the ABI is frozen.

---

## 6. On-disk format

Versioned, little-endian, mmap-able, 64-byte aligned sections.

```
header:  magic "ZQNT" | format_version | dim | dim_padded | bits | objective
         metric | rotation_kind | rounds | codebook_id | seed | count | layout | crc32
sections: codes[]     (blocked, 32-vector groups, dimension-major)
          sketch[]    (prod only, 1 bit/dim, bitmap)
          gamma[]     (prod only, f16)
          norms[]     (f16)
          ids[]       (optional u64 external ids)
```

Rules: the header is self-describing enough that a reader can refuse mismatched builds rather than
producing silent garbage; `codebook_id` pins the exact centroid table so a future Lloyd–Max
refinement cannot corrupt existing indexes; `crc32` is over the header only (checking `n·d` bytes on
open would defeat the point of mmap).

---

## 7. Correctness strategy

The paper hands us an unusually good test oracle. We use all of it.

### 7.1 Exact-value tests
- Lloyd–Max output vs. the paper's published centroids: `b=1 → ±√(2/π)/√d`;
  `b=2 → ±0.453/√d, ±1.51/√d`.
- Rotation orthogonality: `‖Πx‖ = ‖x‖`, `ΠᵀΠ = I` to f32 epsilon; block-FWHT vs. naive `O(d²)` DFT-free reference.
- QJL unbiasedness: `E[⟨y, x̃⟩] = ⟨y, x⟩` inside a bootstrapped CI.

### 7.2 Distortion regression (gates the RHT deviation)
Measured against the paper's numbers, for both the dense-QR reference and the RHT path:
- `D_mse ≤ (√3·π/2)·4^−b = 2.7207·4^−b`, plus the finer per-bit-width values for b=1..4.
  Note `D(b)·4^b` approaches 2.7207 from below (1.45, 1.88, 2.21, 2.43, ...), so this is an
  asymptotic bound and a test asserting "≈4× per bit" at low b would be wrong: the measured
  ratios are 3.09, 3.40, 3.64, 3.79, climbing toward 4.
- `D_prod ≈ {1.57, 0.56, 0.18, 0.047}/d` for b=1..4.

If RHT misses these, the fast path is not shipped as the default — that decision is made on numbers.

### 7.3 Recall and throughput benchmarks
Paper's datasets: GloVe-200, DBpedia-OpenAI-3-large at d=1536 and d=3072, 100k corpus / 1k queries,
recall `1@k`. Tracked in CI as a numeric time series, so a "harmless" kernel refactor that costs
0.4% recall is visible.

**Baselines.**

| Baseline | Why | Bar |
|---|---|---|
| FAISS `IndexPQ` (LUT256) | The incumbent everyone compares to | Beat on recall (paper) and on QPS |
| RaBitQ | The other data-oblivious competitor in the paper | Beat on recall (paper) |
| **[turbovec](https://github.com/RyanCodrai/turbovec)** | Rust + PyO3 implementation of the *same* algorithm | See below |
| Exact f32 scan | Upper bound on recall, lower bound on speed | Quantify the gap we are trading away |

**turbovec is the benchmark that matters most**, because it isolates implementation quality from
algorithm choice: same paper, same quantizer, different language and different kernels. Any recall
difference against turbovec is *our bug* (or theirs), not an algorithmic tradeoff — which makes it a
far sharper correctness signal than FAISS. Its published numbers set the bar:

- **Search**, 100k vectors / 1k queries / k=64: 3.5× faster than FAISS PQ at 4-bit and 26% faster at
  2-bit on ARM (Axion); 3.4× / 20% on x86 (Sapphire Rapids).
- **Recall**: ~0.997–1.0 by k≤8 on d=1536/3072 with the inner-product variant.
- **Insert**: 6.3–19.7 µs single-vector; 4.6–16.3 µs/vector in 100-vector batches.
- **Compression**: 16× at 2-bit, 8× at 4-bit vs fp32.

Independently, turbovec is strong evidence that §4.2 is the right kernel design — it reports
ARM NEON `SDOT`/`SMMLA`, AVX-512 VNNI with `vpermb`, an AVX2 fallback, and a scalar path, which is
convergent with the int8 blocked-ADC plan arrived at here. (`vpermb` extends the shuffle-LUT trick
to 64-entry tables, i.e. b=5–6; worth adopting if we support bit-widths above 4.)

**Explicit rule for this comparison:** we run turbovec ourselves, on our hardware, in our harness —
we do not compare our measured numbers against their README's. Cross-machine benchmark comparison is
how libraries end up claiming wins that do not exist. The harness must be able to drive turbovec's
Python API (`TurboQuantIndex.add/search`) alongside ours in the same process and the same dataset load.

**Feature parity checklist** derived from turbovec's surface, to be answered deliberately rather
than by omission — each is either scheduled or explicitly declined:

| Capability | zquant plan |
|---|---|
| Online single-vector insert | P1 — data-oblivious encode makes this trivial; must not regress to a rebuild |
| `remove()` + compaction/`sync()` | P4 — tombstone + block compaction |
| Stable external IDs (`IdMapIndex`) | P2 — optional `ids[]` section already in the format (§6) |
| Filtered search (allowlist at query time) | P4 — bitmap predicate folded into the block scan, not a post-filter |
| `calibrate()` | Investigate in P1. TurboQuant is data-oblivious by construction, so a calibration step implies a data-dependent refinement outside the paper. Understand what it buys before deciding. |

### 7.4 Cross-language conformance
Fixed-seed golden vectors checked into `tests/golden/`: for a given `(seed, d, b, objective)` and a
fixed input matrix, the exact code bytes and the exact estimator outputs. Zig, Python, JS, and Go
each replay them. This is the only real defense against bindings drifting apart.

### 7.5 Fuzz / property
Encode-decode round-trip bounds, non-power-of-two dims, d < 64 (where the Beta density genuinely
differs from the Gaussian limit and the near-independence assumption weakens), zero vectors,
denormals, `n` not a multiple of the 32-vector block size.

---

## 8. KV-cache path

The paper's KV-cache results are not a footnote application of the ANN path — they use a different
dimension regime, fractional bit widths, and a data-dependent preprocessing step. Treating them as
"the same code with different parameters" is the main way this could go wrong.

### 8.1 What the paper actually does

- `d = 128` (head_dim), quantizing along the channel axis, per token, per head, per layer.
- **Outlier channel splitting** produces the fractional rates: channels are partitioned into an
  outlier set and a regular set, and *two independent TurboQuant instances* run over the disjoint
  subsets at different bit widths. Their 2.5-bit config is `(32 channels × 3b + 96 × 2b)/128 = 2.5`.
- Unlike KIVI and PolarQuant, which leave recently-generated tokens in fp16, **TurboQuant quantizes
  during streaming generation too**. There is no unquantized recent window to hide behind, which
  means the online encode path is on the critical path of every decode step.
- Reported: 3.5 bits/channel is quality-neutral vs. fp16; 2.5 bits/channel is marginally degraded.

### 8.2 Design consequences

**Exact Beta density is mandatory, not an optimization.** At `d=128` the coordinate density
`(1−t²)^((d−3)/2)` is measurably not `N(0,1/d)`, and the near-independence argument is weaker.
`density.zig` must solve Lloyd–Max against the true Beta for the KV bit widths, and §7.2 must be run
at `d=64` and `d=128` specifically — not just at embedding dimensions where everything looks fine.

**Keys and values want different quantizers.** Keys enter `q·Kᵀ`, an inner product → `TurboQuant_prod`
(unbiased IP is exactly what attention scores need). Values enter `Σₜ pₜ vₜ`, a weighted average →
`TurboQuant_mse` (minimizing reconstruction error is the right objective; there is no inner-product
bias to correct). This falls straight out of the two variants and should be the default, with an
override for experimentation.

**Two kernels, not one.** The key-side score computation is the §4.2 blocked scan with `n = seq_len`,
`d = 128` — reusable as-is. The value-side is new: `out += pₜ · dequant(vₜ)` accumulated over tokens,
which is still shuffle-LUT friendly (16-entry table, `vpshufb`) but accumulates into `d` lanes rather
than reducing to a scalar. `simd/attn.zig`.

**Layout and append.** Cache is `[layer][head][token][head_dim]`. The 32-vector block layout of §4.2
assumes a static corpus; a KV cache grows one token at a time. Plan: append into a small
row-major staging buffer, and transpose into a 32-vector block once full. Scan reads the sealed
blocks with the fast kernel and the partial tail with a scalar path. Encode latency at `d=128` is a
short RHT (7 butterfly stages, entirely in L1) plus a threshold pass — sub-microsecond, which is what
makes quantizing during streaming generation viable.

**Outlier channel calibration.** This is the one data-dependent piece in an otherwise data-oblivious
library, and it must be quarantined as such: a separate offline step producing a per-(layer, head)
channel mask + per-set bit widths, stored as a model-specific artifact, never inferred at runtime.
It is also almost certainly what turbovec's `calibrate()` is doing (§7.3) — worth confirming.

**RoPE ordering.** Keys are quantized post-RoPE. Stated explicitly because getting it backwards
produces a subtly-worse-but-not-obviously-broken model, the worst kind of bug.

**Numerics.** f16/bf16 end to end. Accumulate scores in f32 regardless; f16 accumulation over a long
context loses precision in a way that shows up as long-context degradation, i.e. exactly the thing
being measured.

### 8.3 Validation

Zig cannot run an LLM, so the KV path is validated through Python: a `transformers`-compatible
`Cache` implementation backed by zquant, run against the paper's own benchmarks.

- **LongBench-E**, Llama-3.1-8B-Instruct and Ministral-7B-Instruct, reproducing Table 1:
  full-cache average 50.06; TurboQuant @3.5b = 50.06; @2.5b = 49.44. We should land within noise of
  those, per-category, not just on the average — the average hides a lot.
- **Needle-in-a-haystack** at 4× compression: perfect retrieval across depths and lengths.
- Distortion tests from §7.2 re-run at `d=64` and `d=128`.

### 8.4 Honest limitation

KV-cache compression pays off most on GPU serving, where the bottleneck is HBM↔SRAM bandwidth, and
this is a CPU SIMD library. Realistic value: CPU and edge inference (llama.cpp-class runtimes), plus
a clean, testable reference implementation of the algorithm that a GPU kernel can be validated
against. We should not market a CPU library as an H100 serving win — the paper's 8× H100 number is
not a number this library can claim.

---

## 9. Roadmap

| Phase | Deliverable | Done when |
|---|---|---|
| **P0** ✅ | Scalar reference core: RNG, dense-QR + RHT rotations, Lloyd–Max, `mse`, `prod` | §7.1 and §7.2 pass on the reference path |
| **P1** | SIMD kernels, blocked layout, `FlatIndex`, bench harness | §7.3 matches/beats PQ + RaBitQ; recall within noise of turbovec; scan is memory-bound on the bench box |
| **P2** | C ABI, on-disk format, Python wheels | `pip install zquant` on 4 platforms; §7.4 passes for Python |
| **P3** | JS/TS: N-API + wasm-simd, one TS surface | `npm i zquant` works in Node and browser; §7.4 passes for JS |
| **P4** | KV track: streaming encode, outlier splitting, attention kernels, `transformers` Cache | §8.3 reproduces Table 1 within noise |
| **P5** | Threading, rerank, IVF partitioning, filtered search | Multi-core scaling measured; IVF recall/QPS curve published |
| **P6** | Go client | §7.4 passes for Go |

Sequencing rationale: P0 and P1 are where the value is and where the risk is. The C ABI is
deliberately *not* first — freezing an ABI before the kernels have settled locks in the wrong shape.
Python precedes JS because it is where the ANN benchmarking ecosystem lives, so P2 makes P1's
numbers independently checkable — and because the KV track (P4) is only testable from Python at all.

The two workloads share P0–P2 almost entirely; they diverge at the kernel and API layer. P4 is
sequenced after the JS client because it is the larger and less certain body of work (§8.4), and
because the exact-Beta and low-`d` work it depends on (§8.2) is validated back in P0.

---

## 10. Open risks

1. **RHT vs. true random rotation.** The near-independence-of-coordinates argument is proven for a
   Haar-random rotation. RHT is a good approximation but adversarial inputs (a vector aligned with a
   Hadamard row) can defeat a single round. Mitigation: 3 rounds with inter-round permutation, and
   §7.2 gates it. Reference path stays available at runtime.
2. **Low dimensions — now a first-order risk, not a corner case.** Guarantees degrade as `d` shrinks;
   near-independence is a high-`d` phenomenon. The KV path lives at `d=64..128`, squarely in the
   uncertain regime, so this is load-bearing rather than academic. Mitigations: exact Beta Lloyd–Max
   rather than the Gaussian limit, §7.2 run explicitly at `d=64/128`, and measuring the crossover so
   we can document a minimum recommended `d` instead of silently under-delivering.
3. **int8 scan precision.** The 4× fast path adds query-side error the paper does not analyze.
   It must be opt-in with a measured recall delta published per dataset, never a silent default.
4. **`vpshufb` availability.** The shuffle-LUT kernel assumes SSSE3+/NEON/simd128. Baseline scalar
   and SSE2 fallbacks are required for correctness, and dispatch must be tested on a machine where
   the wide path is disabled — untested fallbacks are how these libraries actually break.
5. **Outlier calibration is the one data-dependent step**, in a library whose entire pitch is being
   data-oblivious. It must stay quarantined offline and model-specific (§8.2). If it starts creeping
   into the runtime path, the zero-index-time property — the actual reason to pick TurboQuant — is gone.
6. **KV-cache value proposition is CPU-bound** (§8.4). Worth deciding early whether the KV track is
   aimed at edge/CPU inference or at being a validation reference for a future GPU kernel, because
   those imply different amounts of polish on the `transformers` integration.
7. **Wheel/prebuild logistics.** Cross-compiled Zig wheels are a real advantage but macOS
   universal2 + Windows MSVC ABI are where it gets fiddly; budget real time in P2.
