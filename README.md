# zquant

TurboQuant implemented in Zig.

An implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri,
Hadian, Mirrokni — ICLR 2026): a data-oblivious vector quantizer with near-optimal
distortion rate, for vector search and LLM KV-cache compression.

See [docs/DESIGN.md](docs/DESIGN.md) for the design and implementation plan, and
[docs/notes.md](docs/notes.md) for a per-commit engineering log of what was built, measured,
and got wrong along the way.

## Try it

```sh
zig build quickbench     # ~1 min, no downloads, no Python
```

Benchmarks are built ReleaseFast by default; `-Doptimize` does not affect them, and
`-Dbench-opt` is the knob if you want otherwise. The difference is not subtle —
`-Dbench-opt=Debug` measures 97 QPS where the default measures 10,450 — so a benchmark
that had quietly fallen back to Debug would be obvious rather than merely disappointing.

Generates its own corpora, computes exact ground truth, and prints recall against storage
and throughput. It runs two corpora that differ only in **centroid norm** — how far the
corpus of unit directions sits from the origin — because that is what decides whether the
per-coordinate calibration is worth anything: neutral at 0.005, **+11.6 points at bits=2**
at 0.648. Real embeddings span that range (nytimes-256 is 0.12, SIFT is 0.65).

For the competitive comparison against FAISS PQ, FAISS RaBitQ and turbovec, see
[docs/comparison.md](docs/comparison.md); it needs real corpora and their dependencies.

## Clients

| language | how it binds | tests |
|---|---|---|
| [Zig](src/root.zig) | native | 175 |
| [Python](clients/python) | ctypes over the C ABI | 9 |
| [TypeScript / Node](clients/typescript) | koffi over the C ABI | 8 |
| [Go](clients/go) | cgo over the C ABI | 9 |

```python
import zquant
index = zquant.Index(dim=256, bits=5)
index.add(vectors)
ids, scores = index.search(queries, k=10, threads=8)
```

All three sit on the same [C ABI](include/zquant.h), so a mistake in it surfaces in every
binding rather than hiding in one. Measured on 100k × 256 vectors at `bits=5`, an Apple M5:

| | build | 1 thread | 10 threads |
|---|---|---|---|
| Zig (native) | — | 5,492 | ~27,000 |
| Python | 225k vec/s | 5,492 | 28,927 |
| TypeScript | 227k vec/s | 5,539 | 27,951 |
| Go | 227k vec/s | 5,655 | 28,127 |

Every binding lands within noise of the library and of each other, so none of them costs
anything measurable. Storage is 132 B/vector — 13 MB against 102 MB as float32.

A flat (exhaustive) index and a standalone quantizer.

## KV-cache compression

Measured on real attention tensors — Q, K and V from SmolLM2-135M over 936 tokens
(`bench/py/dump_kv.py`, then `zig build kv_bench`). A KV cache is not a retrieval problem:
attention needs every score, because they pass through a softmax, so the metric is error in
the attention output against fp16, not recall.

| scheme | B/token/head | output error |
|---|---|---|
| fp16 | 256 | — |
| int8 per-row | 136 | 0.013 |
| int4 per-row | 72 | 0.228 |
| **zquant b=4** | **72** | **0.108** |
| zquant b=5 | 88 | 0.057 |
| zquant b=3 | 56 | 0.310 |

**About 2× lower output error than int4 at identical memory** — 0.108 against 0.228 on
layer 15, and 0.127 against 0.257 on layer 7.

Two things differ from the retrieval defaults, and both matter a great deal: **turn
calibration off** (these tensors have effective rank 6.4 of 64, and calibration costs
accuracy on low-rank data), and **reconstruct rather than estimate** — the index's
per-vector correction is fitted to preserve *ranking*, where attention needs accurate
absolute scores. Using the retrieval defaults instead measures 0.409, worse than int4.

```python
codec = zquant.Codec(dim=64, bits=4)
codes, norms = codec.encode(keys)      # keep both; codes are bit-packed
keys_back = codec.decode(codes, norms)
```

Note that `Codec` takes the codebook width directly, where `Index` reserves a bit for its
residual sketch — so `Codec(bits=4)` and `Index(bits=5)` pack the same number of bits per
coordinate. Measured on one 135M-parameter model over three layers; whether it holds at 7B
and long context is untested.

## Benchmark results

Every number below comes from a **single merged run per corpus**, with both systems on the
same data, the same retrieval depth (k=100) and the same core count. `bench/py/compare.py`
refuses to merge results measured under different conditions, because three earlier
comparisons in this project were wrong for exactly that reason.

Environment: **Apple M5** (4 performance + 6 efficiency cores), macOS 26.5, Zig 0.16.0,
aarch64, `-Doptimize=ReleaseFast`. Throughput is full-machine for every system — turbovec
and FAISS both use all cores inside `search()`.

### SIFT10K — 10,000 × 128, image descriptors

| B/vec | system | R@10 | QPS |
|---|---|---|---|
| 16 | **FAISS PQ** M=16,nbits=8 | **0.511** | 90,509 |
| 20 | zquant bits=2 +cal | 0.356 | 184,945 |
| 24 | FAISS RaBitQ qb=5 | 0.356 | 63,416 |
| 32 | FAISS PQ M=32,nbits=8 | 0.648 | 59,826 |
| 36 | **zquant** bits=3 +cal | **0.691** | **186,081** |
| 36 | turbovec bits=2 +cal | 0.682 | 100,022 |
| 52 | **zquant** bits=4 +cal | **0.832** | 131,320 |
| 64 | FAISS PQ M=64,nbits=8 | 0.844 | 32,692 |
| 68 | **zquant** bits=5 +cal | **0.907** | **194,024** |
| 68 | turbovec bits=4 +cal | 0.904 | 106,105 |

### nytimes-256 — 100,000 × 256, text embeddings

| B/vec | system | R@10 | QPS |
|---|---|---|---|
| 16 | **FAISS PQ** M=16,nbits=8 | **0.393** | 17,063 |
| 32 | **FAISS PQ** M=32,nbits=8 | **0.563** | 10,427 |
| 36 | zquant bits=2 +cal | 0.532 | 26,938 |
| 40 | FAISS RaBitQ qb=5 | 0.537 | 5,617 |
| 64 | **FAISS PQ** M=64,nbits=8 | **0.752** | 5,043 |
| 68 | turbovec bits=2 +cal | 0.743 | 21,691 |
| 68 | zquant bits=3 +cal | 0.737 | 26,565 |
| 100 | **zquant** bits=4 +cal | **0.857** | 13,040 |
| 132 | **zquant** bits=5 +cal | **0.916** | **27,013** |
| 132 | turbovec bits=4 +cal | 0.914 | 21,715 |

### Reading these honestly

**Against turbovec, recall is a tie and throughput is ours.** 0.907 against 0.904 at 68 B on
SIFT; 0.916 against 0.914 at 132 B on nytimes; 0.737 against 0.743 at 68 B on nytimes, where
*they* are ahead by 0.6 points. Storage is identical — their serialized form is the same size
as our compact one. Throughput is **1.2–1.8× ours** across every matched pair. Their resident
footprint is larger (270 B against our 132 B) because they dequantize to int8 in memory; our
`expanded` mode at 260 B is what corresponds to their in-memory form.

**Against FAISS PQ, the two systems win at opposite ends.** PQ has better recall per byte at
low storage — decisively on SIFT (0.511 against 0.356 at 16–20 B, a 15-point gap) and modestly
on nytimes (0.563 against 0.532 at 32–36 B). Its learned sub-vector codebooks represent
correlations that a per-coordinate scalar quantizer cannot express at ~1 bit per dimension;
that is structural and is [documented rather than treated as open](docs/comparison.md).
zquant wins the high-recall end on both corpora, and is **3–6× faster** throughout.

**What we would not claim.** These are two corpora at d=128 and d=256, on one machine. The
sub-25 B band belongs to PQ. Parallel throughput drifts with thermal state by up to 25% across
consecutive runs, so treat the QPS column as approximate; single-thread figures are stable to
about 5%. d=1536–3072 and corpora beyond 100k vectors are untested.

## Status

**P0 (reference core) and P1 (kernels, flat index, comparison) complete.** 175 tests passing.

Reproduced from the paper: Lloyd-Max levels (the classic Max table to six decimals),
`D_mse` of 0.36/0.117/0.03/0.009 for b=1..4 measured end to end, the 2/π inner-product
bias of an MSE-only estimator, and — as a harness check — the paper's own Gaussian
sketch at `Var·m = 1.55` against its published 1.57.

Two documented improvements over the paper's construction, both measured rather than
assumed: the distortion bound constant is `√3·π/2 = 2.7207` (not `√(3π)/2`), and an
orthogonal QJL sketch gives exact rather than asymptotic normalization plus ~2.7×
lower inner-product distortion. See [docs/DESIGN.md](docs/DESIGN.md) §1.2.

See [docs/comparison.md](docs/comparison.md) for the measured comparison against FAISS PQ,
FAISS RaBitQ, and turbovec. zquant currently **loses at every matched storage budget**;
the gap is localized to the estimator and the quantizer itself measures at theoretical
optimum.

## Building

Requires Zig 0.16.0.

```sh
zig build test
```

### macOS note

On this machine the Xcode 26 SDK ships a `libSystem.tbd` whose target list omits the
plain `arm64-macos` slice (it has `arm64e-macos` only), so linking against it fails on
Apple Silicon with a wall of `undefined symbol: _getcwd`-style errors. The Command Line
Tools SDK is unaffected. Either prefix builds:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools zig build test
```

or switch the active developer directory once, globally:

```sh
sudo xcode-select -s /Library/Developer/CommandLineTools
```
