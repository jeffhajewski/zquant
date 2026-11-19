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
corpus of unit directions sits from the origin — because that is what decides whether
calibration is worth anything: neutral at 0.005, **+13 points at bits=2** at 0.648. Real
embeddings span that range (nytimes-256 is 0.12, SIFT is 0.65, Cohere's is 0.85).

For the competitive comparison against FAISS PQ, FAISS RaBitQ and turbovec, see
[docs/comparison.md](docs/comparison.md); it needs real corpora and their dependencies.

## Clients

| language | how it binds | tests |
|---|---|---|
| [Zig](src/root.zig) | native | 198 |
| [Python](clients/python) | ctypes over the C ABI | 104 |
| [TypeScript / Node](clients/typescript) | koffi over the C ABI | 65 |
| [Go](clients/go) | cgo over the C ABI | 45 |

```python
import zquant
index = zquant.Index(dim=768, bits=5)
index.calibrate(sample, auto=True)    # only if the data says it will help
index.add(vectors, threads=8)
ids, scores = index.search(queries, k=10, threads=8)
ids, scores = index.search_rerank(queries, vectors, k=10)   # exact top-k from 4x candidates
index.save("vectors.zq")
```

A flat (exhaustive) index and a standalone quantizer (`Codec`), with the same API in every
binding:

- **Save and load.** Versioned, checksummed, the same bytes on every platform. A loaded
  index searches and accepts new vectors exactly as the saved one did.
- **Exact reranking.** Over-fetch candidates and rescore them against your full-precision
  vectors. On fashion-mnist this takes R@10 at bits=3 from 0.761 to 0.991, for about 5% of
  throughput.
- **Scores for ids you choose**, for ranking a candidate set from a filter or another
  index without scanning the corpus.
- **Automatic calibration.** It measures how far off-centre the data is and calibrates
  only where that has measured a gain. Calibrating fits the corpus centre and costs 2 B per
  vector; on data well off the origin it adds 9 to 22 points of R@10 at bits 2 and 3.
- **Parallel encoding.** It produces the same bytes as serial encoding.
- **Thread safety.** Many threads can search one index at once. Adding, closing and
  garbage collection are safe against searches in flight.

All three sit on the same [C ABI](include/zquant.h), so a mistake in it surfaces in every
binding rather than hiding in one. Each replays the same
[conformance fixture](tests/golden/conformance.json) and must reproduce its hashes byte
for byte. Measured on 100k × 256 vectors at `bits=5`, an Apple M5:

| | add, 1 thread | add, 10 threads | search, 1 thread | search, 10 threads |
|---|---|---|---|---|
| Python | 428k vec/s | 2.39M vec/s | 5,656 QPS | 29,345 QPS |
| TypeScript | 427k vec/s | 2.44M vec/s | 5,627 QPS | 29,214 QPS |
| Go | 436k vec/s | 2.43M vec/s | 5,580 QPS | 27,543 QPS |

The three bindings land within noise of each other, and of the library timed from C
(5,509 QPS on one thread). None of them costs anything measurable. Creating an index takes
about 50 ms. Storage is 132 B/vector: 13 MB, against 102 MB as float32.

## KV-cache compression

**At matched memory, zquant is not the best KV-cache quantizer measured here.** KIVI-style
4-bit quantization gives lower perplexity than zquant b=4 at the same 72 bytes on every
model tried. On Qwen2.5-0.5B zquant's keys fail outright. On the two SmolLM2 models zquant
does beat plain per-token int4. That was the baseline behind this section's earlier "about
2× lower error than int4" claim, and the baseline was too weak to support it.

### Perplexity

`bench/py/kv_ppl.py` runs a model over wikitext-2 (raw test split, consecutive 1024-token
windows). Every key and value is quantized and dequantized as it enters the cache, and keys
are post-RoPE, which is what a cache stores. B/token/head counts K and V together at
head_dim 64. That covers packed codes, every scale and zero point (f16), and zquant's norm
(f32).

| B/token/head | scheme | SmolLM2-135M | SmolLM2-360M | Qwen2.5-0.5B |
|---|---|---|---|---|
| 256 | fp16 (perplexity) | 17.37 | 12.07 | 13.42 |
| | *increase over fp16:* | | | |
| 72 | int4 per-row, symmetric (the old baseline) | +4.31 | +2.30 | +35.0 |
| 72 | int4 per-token, asymmetric, group 64 | +1.46 | +0.94 | +97.3 |
| 80 | int4 per-token, asymmetric, group 32 | +0.59 | +0.33 | +28.5 |
| 72 | **KIVI-4, group 64** | **+0.31** | **+0.16** | **+0.22** |
| 80 | KIVI-4, group 32 | +0.22 | +0.10 | +0.16 |
| 72 | zquant b=4 | +0.79 | +0.47 | +151 |
| 88 | zquant b=5 | +0.20 | +0.15 | +39.9 |
| 72 | zquant b=4, keys mean-centred | +0.67 | +0.43 | +5.25 |
| 88 | zquant b=5, keys mean-centred | +0.16 | +0.11 | +1.13 |
| 72 | KIVI-4 keys + zquant b=4 values | +0.30 | +0.16 | +0.23 |
| 40 | KIVI-2, group 64 | +30.4 | +15.5 | +41.1 |
| 40 | zquant b=2 | +31.1 | +22.2 | +1097 |
| 48 | KIVI-2, group 32 | +11.8 | +6.6 | +13.2 |
| 56 | zquant b=3 | +3.95 | +2.28 | +367 |

KIVI ([Liu et al., 2024](https://arxiv.org/abs/2402.02750)) quantizes keys per *channel*,
grouped over tokens, and values per token, grouped over channels. Both use a scale and a
zero point. SmolLM2-135M is 64 windows (65,536 tokens). The other two are 16 windows
(16,384 tokens), because the machine could not run longer.

What it shows:

- **At 72 B, KIVI-4's perplexity cost is 2.5–3× smaller than zquant b=4's** on both
  SmolLM2 models. zquant needs b=5, 22% more memory, to draw level with KIVI-4 at 72 B.
  Against KIVI-4 at 80 B it is level on 135M and behind on 360M.
- **The loss is in keys.** Pairing KIVI keys with zquant values matches all-KIVI. zquant
  reconstructs values as well as anything here.
- **Qwen2.5 breaks zquant's keys.** Its `k_proj` has a bias, so every key carries a large
  constant offset in some channels. zquant scales by the vector norm and rotates, which
  spreads that offset over every coordinate. Most of the code's range then goes to a
  constant. Subtracting each channel's mean before encoding (the "mean-centred" rows) takes
  b=4 from +151 to +5.25, and b=5 from +39.9 to +1.13. That is still far behind KIVI's
  +0.22. What remains is consistent with outlier channels that differ in *scale* as well
  as offset: KIVI gives each channel its own scale, and a data-oblivious rotation mixes
  them. Centring helps the SmolLM2 models a little too, but leaves KIVI ahead at 72 B.
  The `Codec` has no centring option today.
- **Nothing here is usable at 2 bits.** KIVI's own 2-bit results depend on an fp16 window
  of recent tokens, which is left out here (below).

### Attention-output error

`zig build kv_bench` measures the error in a single attention layer's output against
fp16, per layer. It uses Q, K and V dumped from SmolLM2-135M over 936 tokens
(`bench/py/dump_kv.py`), with the same schemes and byte accounting. This is a diagnostic,
not the verdict. It splits key error from value error, which perplexity cannot.

| B/token/head | scheme | layer 0 | layer 7 | layer 15 |
|---|---|---|---|---|
| 136 | int8 per-row | 0.010 | 0.016 | 0.013 |
| 72 | int4 per-row, symmetric | 0.200 | 0.242 | 0.233 |
| 72 | int4 per-token, asymmetric, group 64 | 0.152 | 0.152 | 0.169 |
| 72 | KIVI-4, group 64 | 0.107 | **0.091** | **0.093** |
| 72 | zquant b=4 | **0.092** | 0.124 | 0.106 |
| 88 | zquant b=5 | 0.044 | 0.062 | 0.055 |

The proxy is kinder to zquant than perplexity is: zquant wins one layer of three, where
perplexity has KIVI ahead on every model. On keys alone KIVI-4 is ahead on all three
layers: 0.024 / 0.074 / 0.056 against zquant's 0.052 / 0.108 / 0.084.

### Using it

Two things differ from the retrieval defaults, and both matter a great deal. **Turn
calibration off**: these tensors have effective rank 6.4 of 64, and calibration costs
accuracy on low-rank data. **Reconstruct rather than estimate**: the index's per-vector
correction is fitted to preserve *ranking*, and attention needs accurate absolute scores.
Using the retrieval defaults instead measures 0.40 at b=5 on layer 15, worse than int4.

```python
codec = zquant.Codec(dim=64, bits=4)
codes, norms = codec.encode(keys)      # keep both; codes are bit-packed
keys_back = codec.decode(codes, norms)
```

Note that `Codec` takes the codebook width directly, while `Index` reserves a bit for its
residual sketch. So `Codec(bits=4)` and `Index(bits=5)` pack the same number of bits per
coordinate.

Caveats:

- **KIVI is shown without its fp16 window.** Real KIVI keeps recent tokens in fp16 until
  a group fills. Every row here quantizes every token at the bytes shown, which is the
  less favourable setup for KIVI.
- **KIVI's key groups see a little of the future.** Each window is one forward pass, so a
  key's group statistics include up to 63 later keys. Real KIVI holds those tokens in fp16
  instead. This favours KIVI slightly, against the window it gives up. The mean-centred
  rows see the whole window's mean in the same way.
- **`kv_bench` pairs pre-RoPE queries with post-RoPE keys.** `dump_kv.py` hooks `q_proj`,
  which is before RoPE, so its scores are not the model's attention logits. That is one
  more reason the perplexity table is the one to trust.
- **Small models, one dataset.** 135M to 494M parameters, wikitext-2 only, context 1024,
  no downstream tasks. Larger models are known to have more extreme outlier channels,
  which is the regime where zquant did worst.

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
| 22 | zquant bits=2 +cal | 0.458 | 178,923 |
| 24 | FAISS RaBitQ qb=5 | 0.356 | 63,416 |
| 32 | FAISS PQ M=32,nbits=8 | 0.648 | 59,826 |
| 36 | turbovec bits=2 +cal | 0.682 | 100,022 |
| 38 | **zquant** bits=3 +cal | **0.702** | **179,340** |
| 54 | **zquant** bits=4 +cal | **0.837** | 129,551 |
| 64 | FAISS PQ M=64,nbits=8 | 0.844 | 32,692 |
| 68 | turbovec bits=4 +cal | 0.904 | 106,105 |
| 70 | **zquant** bits=5 +cal | **0.907** | **184,162** |

### nytimes-256 — 100,000 × 256, text embeddings

| B/vec | system | R@10 | QPS |
|---|---|---|---|
| 16 | **FAISS PQ** M=16,nbits=8 | **0.393** | 17,063 |
| 32 | **FAISS PQ** M=32,nbits=8 | **0.563** | 10,427 |
| 38 | zquant bits=2 +cal | 0.536 | 25,532 |
| 40 | FAISS RaBitQ qb=5 | 0.537 | 5,617 |
| 64 | **FAISS PQ** M=64,nbits=8 | **0.752** | 5,043 |
| 68 | turbovec bits=2 +cal | 0.743 | 21,691 |
| 70 | zquant bits=3 +cal | 0.744 | 28,308 |
| 102 | **zquant** bits=4 +cal | **0.858** | 12,980 |
| 132 | turbovec bits=4 +cal | 0.914 | 21,715 |
| 134 | **zquant** bits=5 +cal | **0.920** | **26,719** |

### Reading these honestly

**Against turbovec, recall is level or slightly ahead, and throughput is ours.** 0.907
against 0.904 on SIFT at 70 B and 68 B; 0.920 against 0.914 on nytimes at 134 B and 132 B;
0.744 against 0.743 on nytimes at 70 B and 68 B. Calibrated, our index is 2 B per vector
larger than theirs, for the bias the new estimator stores; uncalibrated, the sizes are
identical. Our throughput is **1.2–1.8× theirs** across every matched pair (1.74× and 1.79×
on SIFT, 1.23× and 1.31× on nytimes). Their resident footprint is larger (270 B against our
134 B) because they dequantize to int8 in memory; our `expanded` mode at 262 B is what
corresponds to their in-memory form.

**Against FAISS PQ, the two systems win at opposite ends.** PQ has better recall per byte at
the lowest budgets: on SIFT 0.511 at 16 B against our 0.458 at 22 B, and on nytimes 0.563 at
32 B against 0.536 at 38 B, and on nytimes PQ at 64 B (0.752) still edges our 70 B (0.744).
zquant wins from 38 B up on SIFT and from about 100 B up on nytimes, and is 1.5–6× faster
than PQ at every pair in these tables. The SIFT gap was 15 points until this release and
was long put down to PQ's learned codebooks. Most of it turned out to be how zquant assembled a score
from its codes: a new estimator closed two thirds of it
([docs/notes.md](docs/notes.md), "Partitioning and the low-bit gap").

**What we would not claim.** These are two corpora at d=128 and d=256, on one machine. At
those dimensions the lowest budgets still belong to PQ. Parallel throughput drifts with thermal
state by up to 25% across consecutive runs, so treat the QPS column as approximate;
single-thread figures are stable to about 5%.

### Real embeddings at d=768 and 1536

Here PQ is no longer the system to beat. On one thread, against FAISS's plain `IndexPQ` and
`IndexRaBitQ` (`bench/py/highdim.py`; turbovec was not rerun):

| corpus | bits/dim | zquant +cal | FAISS PQ | FAISS RaBitQ | zquant QPS | RaBitQ QPS |
|---|---|---|---|---|---|---|
| Cohere, 100k × 768 | ~1 | **0.750** | 0.582 | 0.738 | 1,817 | 510 |
| | ~2 | **0.868** | 0.753 | 0.861 | 1,900 | 434 |
| | ~4 | **0.960** | 0.917 | 0.952 | 2,016 | 417 |
| OpenAI ada-002, 50k × 1536 | ~1 | **0.791** | 0.669 | 0.786 | 1,868 | 838 |
| | ~2 | **0.894** | 0.828 | 0.890 | 1,935 | 616 |
| | ~4 | **0.967** | 0.944 | 0.962 | 2,072 | 568 |

PQ trails at every budget, by 12–17 points at 1 bit per dimension. zquant now edges RaBitQ
at every budget, by 0.4 to 1.2 points, while storing 2–14 B less per vector and searching
2.2–4.8× faster. Until this release RaBitQ led by 1.4–2.7 points at 1 and 2 bits; adopting
its treatment of the corpus centre closed that. [docs/comparison.md](docs/comparison.md) has
the full table.

### Larger corpora and higher dimensions

zquant alone, without the other systems (they were not rerun). nytimes is the full 290k
corpus with its published ground truth. The two higher-dimensional corpora are
**synthetic**: unit vectors with a decaying spectrum and a small common offset, with exact
ground truth, because the real corpora at d ≥ 768 in the previous section have only 50k–100k
vectors. `+rr4` is `search_rerank` with 4× candidates. "1 thread" is one core searching
queries in batches of 32, as the bindings do. Throughput is k=10, from C, on the M5. This
table predates the estimator change in this release and was not rerun; its recall is
uncalibrated, which on the 100k nytimes sample barely moved (0.730 at bits=3, against 0.732
before).

| corpus | bits | B/vec | R@10 | R@10 +rr4 | QPS, 1 thread | QPS, 10 threads | add, 10 threads |
|---|---|---|---|---|---|---|---|
| nytimes, 290k × 256 | 3 | 68 | 0.766 | 0.955 | 1,839 | 9,682 | 2.47M/s |
| | 5 | 132 | 0.918 | 0.985 | 1,953 | 10,758 | 2.58M/s |
| synthetic, 500k × 768 | 3 | 196 | 0.560 | 0.900 | 402 | 2,153 | 696k/s |
| | 5 | 388 | 0.866 | 1.000 | 423 | 2,342 | 708k/s |
| synthetic, 250k × 1536 | 3 | 388 | 0.629 | 0.948 | 409 | 2,242 | 357k/s |
| | 5 | 772 | 0.883 | 1.000 | 440 | 2,397 | 329k/s |

Search is a linear scan, so throughput falls with n·d as expected. At d=768 the index no
longer pads to 1024, so these sizes are 25% smaller than they were. bits=4 is left out
because it is the slow width (its 3-bit codes need bit-plane unpacking; see `Residency` in
[src/index/flat.zig](src/index/flat.zig)). It runs at about 40% of the speed of bits=3 and
5.

On x86_64 the only measurements are from a GitHub runner, an AMD Zen 3 with 4 vCPUs,
running quickbench (50k × 256, k=10). One thread at bits=5 measures 3,484 QPS with the
AVX2 kernels the release library chooses on such a CPU, and 1,992 with the SSSE3 ones it
falls back to without AVX2. [bench.yml](.github/workflows/bench.yml) reruns it.

## Status

**P0 (reference core) and P1 (kernels, flat index, comparison) complete.** 198 Zig tests
passing, plus 214 across the three bindings.

Reproduced from the paper: Lloyd-Max levels (the classic Max table to six decimals),
`D_mse` of 0.36/0.117/0.03/0.009 for b=1..4 measured end to end, the 2/π inner-product
bias of an MSE-only estimator, and — as a harness check — the paper's own Gaussian
sketch at `Var·m = 1.55` against its published 1.57.

Two documented improvements over the paper's construction, both measured rather than
assumed: the distortion bound constant is `√3·π/2 = 2.7207` (not `√(3π)/2`), and an
orthogonal QJL sketch gives exact rather than asymptotic normalization plus ~2.7×
lower inner-product distortion. See [docs/DESIGN.md](docs/DESIGN.md) §1.2.

See [docs/comparison.md](docs/comparison.md) for the measured comparison against FAISS PQ,
FAISS RaBitQ, and turbovec. The tables above summarize it. At d=128 and 256, FAISS PQ has
better recall per byte at the lowest budgets: below about 25 B on SIFT and below about
100 B on nytimes. Above those, zquant matches or beats turbovec, beats PQ and RaBitQ, and is
the fastest of the four. At d=768 and 1536, PQ trails at every budget, and zquant edges
FAISS RaBitQ at every budget while searching 2.2–4.8× faster.

## Building

Requires Zig 0.16.0.

```sh
zig build test
zig build lib            # the C library, header and pkg-config file, into zig-out/
```

### Prebuilt libraries and CPUs

The scan kernels are chosen at compile time, so a library built for one CPU uses that CPU's
instructions unconditionally. Releases are built for SSSE3 on x86_64 (`x86_64_v2`) and for
the dot-product extension on arm64. Search is 6–10× slower without them. Each release
also has a `-portable` build for the bare baseline, named `libzquant-portable`, for older
CPUs.

The x86_64 libraries also carry a copy of the kernels compiled for AVX2, and use it on a
CPU that has AVX2: one-thread search measured 1.34× (bits=4) to 1.75× (bits=5) faster
with it. `zq_build_kernels()` reports `wide=avx2` when that copy is in use. Setting
`ZQUANT_KERNELS=none` keeps the library to its own kernels.

The library checks at load time rather than crashing: `zq_cpu_supported()` is safe on any
CPU, and creating an index on an unsupported one returns `ZQ_ERR_CPU`. The Python and
TypeScript bindings fall back to `libzquant-portable` automatically when it sits beside
the main library. `zq_build_cpu()` and `zq_build_kernels()` say what a given build
contains. `zig build lib` with no options builds for the machine you're on.

Releases also carry a Python wheel per platform and one npm tarball for every
platform, holding the same libraries as the archives; the
[Python](clients/python/README.md#install) and
[TypeScript](clients/typescript/README.md#install) READMEs say how to install them.
Linux builds need glibc 2.17 or newer, macOS builds macOS 13.

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
