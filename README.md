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
zig build quickbench -Doptimize=ReleaseFast     # ~1 min, no downloads, no Python
```

Generates its own corpora, computes exact ground truth, and prints recall against storage
and throughput. It runs two corpora that differ only in **centroid norm** — how far the
corpus of unit directions sits from the origin — because that is what decides whether the
per-coordinate calibration is worth anything: neutral at 0.005, **+11.6 points at bits=2**
at 0.648. Real embeddings span that range (nytimes-256 is 0.12, SIFT is 0.65).

For the competitive comparison against FAISS PQ, FAISS RaBitQ and turbovec, see
[docs/comparison.md](docs/comparison.md); it needs real corpora and their dependencies.

## What you can use it for today

A flat (exhaustive) index and a standalone quantizer, **callable from Zig only** — there is
no C ABI and no Python/JS/Go bindings yet, so none of the results below are reachable from
those languages. KV-cache compression is in scope for the algorithm and is a stated goal,
but is not yet benchmarked.

## Status

**P0 (reference core) and P1 (kernels, flat index, comparison) complete.** 175 tests passing.

Measured against turbovec on identical corpora, retrieval depth and core count, in a single
merged run each (the harness refuses to merge results measured under different conditions):

| corpus | | B/vec | R@10 | QPS |
|---|---|---|---|---|
| SIFT10K | **zquant** bits=5 +cal | 68 | **0.907** | **193,386** |
| | turbovec bits=4 +cal | 68 | 0.904 | 108,715 |
| nytimes-256 | **zquant** bits=5 +cal | 132 | **0.916** | **27,013** |
| | turbovec bits=4 +cal | 132 | 0.914 | ~22,000 |

FAISS PQ leads below 25 B/vector (0.511 against 0.357 at 16–20 B); that is conceded with
evidence rather than open — see [docs/comparison.md](docs/comparison.md).

Measured on an **Apple M5 (4 performance + 6 efficiency cores), macOS 26.5, Zig 0.16.0,
aarch64**. Several findings are specific to that hardware — notably that `SMMLA` and `SDOT`
have identical int8 MAC throughput there, so the wider matrix instruction buys nothing.
Parallel figures drift with thermal state by as much as 25% across consecutive runs;
single-thread figures are stable to about 5%.

Implemented:

| | |
|---|---|
| `math/rng` | Philox4x32-10, counter-based, pinned by Random123 KAT vectors |
| `math/quadrature` | Composite Gauss-Legendre, nodes solved at comptime |
| `math/density` | Exact sphere-coordinate density and its N(0,1/d) limit |
| `math/lloyd_max` | Continuous 1-D k-means, with guarded Aitken acceleration |
| `math/rotation` | 3-round randomized Hadamard; dense Haar reference |
| `quant/codebook` | Lloyd-Max solution narrowed to f32 |
| `quant/mse` | TurboQuant_mse (Algorithm 1) |
| `quant/prod` | TurboQuant_prod (Algorithm 2) with QJL residual sketch |

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
