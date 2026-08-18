# zquant

TurboQuant implemented in Zig.

An implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri,
Hadian, Mirrokni — ICLR 2026): a data-oblivious vector quantizer with near-optimal
distortion rate, for vector search and LLM KV-cache compression.

See [docs/DESIGN.md](docs/DESIGN.md) for the design and implementation plan.

## Status

**P0 (reference core) complete.** 80 tests passing.

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

Next: P1 — SIMD kernels, blocked layout, flat index, benchmarks.

## Building

Requires Zig 0.15.2.

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
