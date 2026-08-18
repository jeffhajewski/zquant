# P0 Engineering Notes

A working log of the reference-core implementation, sectioned by commit. Written to be
re-read cold: each section records not just what landed but *why*, what was measured,
what was wrong on the first attempt, and what is still open.

Reference: Zandieh, Daliri, Hadian, Mirrokni, *TurboQuant: Online Vector Quantization
with Near-optimal Distortion Rate*, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874),
ICLR 2026.

**Environment.** Zig 0.15.2, Apple Silicon, Darwin 25.5. All builds need
`DEVELOPER_DIR=/Library/Developer/CommandLineTools` — see [2b38e4f](#2b38e4f).

**Final state.** 8 commits, ~3,120 lines, 80 tests. 36s Debug / 6s ReleaseFast.

---

## Headline findings

Three things worth remembering even if nothing else here is:

1. **The distortion-bound constant in the paper's PDF is easy to misread, and I did.**
   It is `√3·π/2 = 2.7207`, not `√(3π)/2 = 1.535`. See [8ff805c](#8ff805c).

2. **An orthogonal QJL sketch is better than the paper's Gaussian one** — exact rather than
   asymptotic normalization, and ~2.7× lower inner-product distortion. Verified by
   re-implementing the paper's construction in the same harness before claiming anything.
   See [71b1e1b](#71b1e1b). *Status of the two halves differs:* unbiasedness is **proven**
   (exact at finite m, by a symmetry argument); the variance improvement is **empirical
   only** — the paper's proofs assume Gaussian S and do not transfer. And lower estimator
   variance on random unit vectors is not yet evidence of better recall on real
   embeddings; that is a P1 question for the turbovec comparison.

3. **Lloyd's convergence is governed by its linear rate, not by the tolerance.** Loosening
   tolerance 6 orders of magnitude cut iterations 33% while costing 4 orders of accuracy.
   See [8ff805c](#8ff805c).

A recurring pattern: **every one of the substantive findings came from a test disagreeing
with me, not from reading the paper more carefully.** Four tests failed across the session;
three exposed real errors (a wrong constant, a wrong statistical method, an aliasing bug)
and one exposed an unjustified assumption (that "≈4× per bit" holds at low b — it does not,
the ratios climb 3.09 → 3.94 toward 4). None were flaky.

---

## <a name="0c2582f"></a>`0c2582f` — docs: add design and implementation plan

562 lines. The plan itself; see `docs/DESIGN.md`.

**Sources.** The blog post is too thin to implement from. I pulled the PDF and extracted
Algorithms 1 and 2 verbatim (`pdftotext -layout`). The blog omits the residual-norm scalar
γ, the exact Beta density, all constants, and the fact that KV-cache work happens at
d = 128 with fractional bit rates.

**Two structural facts that drove everything downstream**, neither stated in the paper:

- *Stay in the rotated basis.* `S' = S·Πᵀ` has the same distribution as `S` (rotational
  invariance of the Gaussian), so the residual can be formed as `u = y − ỹ` and the
  estimator becomes `⟨q,x̃⟩ = ‖x‖·[⟨p,ỹ⟩ + γ·scale·⟨S'p,qjl⟩]` with `p = Π·q` computed once
  per *query*. The corpus is never rotated back. This is the difference between a rotation
  per comparison and a rotation per query.
- *At b ≤ 4 the codebook is ≤ 16 scalars* — one `vpshufb`/`tbl` register. Dequantization
  becomes a single shuffle per 32 codes, which is what makes a FastScan-style 32-vector
  blocked layout the right kernel shape.

**turbovec** ([RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec)) was added as
the primary baseline mid-session at Jeff's request. It is the same algorithm in Rust, which
makes it a sharper correctness signal than FAISS: any recall gap is a bug on one side, not
a tradeoff. It independently reports NEON `SDOT`/`SMMLA` + AVX-512 VNNI `vpermb`, convergent
with the int8 blocked-ADC design arrived at here — good evidence the kernel plan is right.

**Scope decisions (Jeff's).** Quantizer + flat index for v1; IVF/graph deferred. KV cache
is a *real* target, not just vector search — which promoted low-dimension behaviour from a
corner case to a first-order risk, since head_dim is 64–128.

---

## <a name="2b38e4f"></a>`2b38e4f` — build: scaffold Zig 0.15.2 project

**The macOS toolchain trap, documented because it will recur.** `zig build` failed with a
wall of `undefined symbol: _getcwd`-style errors. Root cause: the Xcode 26 SDK's
`libSystem.tbd` lists targets

```
[ x86_64-macos, x86_64-maccatalyst, arm64e-macos, arm64e-maccatalyst ]
```

with **no plain `arm64-macos` slice**, so the linker matches nothing on Apple Silicon. The
Command Line Tools SDK still carries `arm64-macos`.

Two dead ends worth not repeating:
- `SDKROOT=...` — ignored by Zig (the cache hash didn't even change).
- `--sysroot ...` — reaches the build's *children* but not the build runner itself, which
  fails to link first. Works for bare `zig test`, not for `zig build`.

Working fix: `DEVELOPER_DIR=/Library/Developer/CommandLineTools zig build test`. Permanent
alternative is `sudo xcode-select -s /Library/Developer/CommandLineTools`, which is a
global machine change and therefore Jeff's call, not mine.

Also: `build.zig.zon` needs a `fingerprint`; Zig prints the correct value in the error.
Zig 0.15's `addTest` takes `root_module: *Module`, not `root_source_file`.

---

## <a name="55ab689"></a>`55ab689` — math: Philox4x32-10 counter-based RNG

322 lines, 11 tests.

**Why counter-based.** An index must be reproducible from `(seed, purpose, index)` alone.
That is what lets a serialized index be described by its seed rather than by a materialized
d×d matrix, and what will let Python/JS/Go regenerate byte-identical codes. A stateful PRNG
cannot do this.

**Domain separation is an explicit enum** (`Purpose`), not an ad-hoc offset. Π and S' must
be independent under the same seed; a counter collision there would silently correlate two
things the algorithm assumes independent — a bug that would show up only as slightly worse
recall, which is the worst way to find it.

**Validation.** Three Random123 KAT vectors. Worth noting how this was validated: the
implementation came from the round structure, the expected vectors came from my
recollection of the reference suite, and they agreed. Two independent derivations matching
is much stronger evidence than either alone. They pin round structure, constants, and the
key-bump schedule together.

**Cross-platform determinism.** Sign bits come from integer words only, so everything
affecting stored codes is bit-exact everywhere. Gaussians use Marsaglia polar rather than
Box-Muller specifically to avoid `sin`/`cos`, whose last-bit results vary across libm
implementations — and they feed only the dense reference rotation, which is a test oracle
and never serialized.

---

## <a name="8ff805c"></a>`8ff805c` — math: densities, quadrature, Lloyd-Max solver

981 lines, 26 tests. The largest and most eventful commit.

### The corrected constant

**This is the single most important thing in these notes.** The paper's PDF renders the
Panter-Dite bound as a stacked fraction `3π / 2d` under a radical. That reads naturally as
`√(3π)/2 = 1.535`. It is actually `√3·π/(2d)`, i.e. `D_mse ≤ √3·π/2 · 4^−b = 2.7207·4^−b`.

Caught because the test asserting the wrong form failed against the *true optimal*
quantizer — at b=2 the optimum is 0.1175 but the bad bound said 0.0959. An optimal
quantizer violating its own bound means the bound is transcribed wrong.

Three independent confirmations:
- Panter-Dite from scratch: `(1/12)·(∫f^(1/3))³ = (1/12)·6√3·π = 2.7207` for a unit Gaussian.
- The abstract says TurboQuant is within "a small constant (≈ 2.7) factor" of the Shannon
  lower bound, and that bound is exactly `4^−b`. So 2.7207 *is* that constant.
- `D(b)·4^b` climbs 1.45, 1.88, 2.21, 2.43, 2.56, 2.64, 2.68 — approaching 2.7207 from
  below, as an asymptotic bound should.

Corrected in `docs/DESIGN.md` §1.1 and §7.2 as well as in code.

### Convergence: tolerance was the wrong lever

Measured accuracy vs. tolerance at b=8:

| tolerance | iterations | max centroid rel. err |
|---|---|---|
| 1e-8 | 33,665 | 4.35e-3 |
| 1e-10 | 47,431 | 2.24e-4 |
| 1e-12 | 50,538 | 0 (ref) |
| 1e-14 | 50,538 | 0 |

Loosening 6 orders of magnitude buys 33% fewer iterations and costs 4 orders of accuracy.
That is the signature of a linear convergence rate very close to 1: the iteration count is
set by the slow mode, not the stopping threshold. **Tuning the tolerance was never going
to work.**

Fixes applied, in order of payoff:
1. *Stop on distortion, not centroid movement.* The outermost levels drift through the
   tails for thousands of iterations after the distortion has stopped changing in the 14th
   digit. b=7: 37,409 → 15,553.
2. *Guarded Aitken Δ² extrapolation.* Extrapolates the geometric error tail; accepted only
   if it lowers the distortion, so it can never converge worse than plain Lloyd. Roughly
   halves again — b=8: 50,538 → 26,922, b=6: 4,597 → 2,185.

Still slow at high b. **Deliberately stopped there**: b ≤ 4 is the shipping target (that's
the shuffle-LUT constraint) and converges in <300 iterations. If high bit-widths ever
matter, the right fix is Max's shooting method — bisect on the lowest centroid and
propagate the recurrence — which converges in ~50 bisection steps regardless of level
count. Noted in `Options.max_iterations`.

### A quadrature performance bug

`mass()` was calling `quadrature.default.init()` — a Newton solve for 16 Gauss-Legendre
nodes — **on every call, inside Lloyd's inner loop**. Hoisting to container scope makes it
comptime-evaluated once. This is why the suite briefly took >2 minutes.

### Other decisions

- **Gauss-Legendre nodes are computed, not tabulated.** A table is a long list of digits
  with no way to distinguish a typo from a correct entry; a computed rule can be checked
  against polynomials it must integrate exactly (degree 31 for 16 points).
- **Closed-form moments, numerical mass.** `∫t·f` has an elementary closed form for both
  densities and is what Lloyd divides by cell mass — quadrature error there would *bias*
  every centroid, not merely blur it. `∫f` for the sphere density is a regularized
  incomplete beta with no elementary form, and std has no `erf`, so it goes through
  quadrature with panels sized to the density's own scale.
- **`lgamma`, not `gamma`.** Γ(768) overflows f64 long before the dimensions in play.
- **Both densities, deliberately.** `f_X → N(0,1/d)` is fine at embedding dimensions but
  measurably wrong at d = 64–128. There is a test asserting the divergence at KV
  dimensions, so the §8.2 requirement rests on a measurement that will fail loudly if it
  ever stops being true.

### Validated against

Classic Max table to six decimals — `0.797885`; `0.452780/1.510418`;
`0.245094/0.756005/1.343909/2.151946`; and 8 values at b=4. Distortions
`0.363380/0.117482/0.034548/0.009501` against the paper's `0.36/0.117/0.03/0.009`.

**Test-tolerance lesson:** the first version compared against the published table at 5e-5.
The table is quoted to 4 decimals, so half an ulp of the quote *is* 5e-5 — the test was
measuring my transcription, not the solver. Relaxed to 1e-4.

---

## <a name="6711abe"></a>`6711abe` — math: randomized Hadamard and dense Haar rotations

517 lines, 14 tests.

**Dropped the inter-round permutation the design called for.** FWHT already makes every
output coordinate depend on every input coordinate, so a permutation adds little that
another sign-flip round does not — and omitting it keeps `apply` allocation-free and
thread-safe, since permuting in place otherwise needs scratch or cycle-following.

Justified by measurement rather than assertion: the 3-round RHT reproduces the sphere
coordinate density's **exact fourth moment `3/(d(d+2))`** — distinguishably *not* the
Gaussian `3/d²` — and matches the dense Haar reference within 5%. If that regresses, add
rounds before adding permutations.

**Why 3 rounds, with a concrete adversarial case.** A single `H·D` maps a standard basis
vector `e₀` to `±1/√d` in *every* coordinate: identical magnitudes, zero spread — the exact
opposite of the Beta-distributed spread the scalar quantizer is built for. Axis-aligned and
one-hot-ish vectors occur in real corpora, so this is a real input, not contrived. There is
a regression test pinning both the degenerate single-round behaviour and its absence at
three.

**Dense reference uses modified, not classical, Gram-Schmidt.** Classical loses
orthogonality badly at these dimensions, and this oracle's own error must stay far below
the effect it is used to measure.

**Padding.** Non-power-of-two `d` is zero-padded to the next power of two. This is *sound*
— a Haar rotation maps any unit vector to a uniform point on `S^(padded−1)`, so the density
parameter is the padded dimension — but it *wastes bits*: 768 → 1024 is 33% overhead. The
block-FWHT scheme in DESIGN.md §1.4 (factor `d = m·2^a`, e.g. `768 = 3·256`) is a P1 item.

---

## <a name="c134aad"></a>`c134aad` — quant: codebook and TurboQuant_mse

567 lines, 15 tests. Algorithm 1, end to end.

**Measured distortion reproduces the paper through the full pipeline** — rotation,
codebook, encode, decode — not just the solver in isolation: 0.36/0.117/0.03/0.009 for
b=1..4 over random unit vectors. This is the DESIGN.md §7.2 gate, and the Hadamard and
dense paths agree within 6% at b=2 and b=4.

**f32 narrowing subtlety.** Thresholds are rebuilt from the *narrowed* f32 centroids rather
than narrowed independently from f64. Rounding them separately can place a threshold off
the true f32 midpoint, so threshold-counting would disagree with a direct argmin for inputs
landing in the gap. There is a test sweeping ~14k values per bit-width for exactly that
disagreement.

**The aliasing bug.** First version passed `ws.rotated[0..dim]` as source and `ws.rotated`
as destination to `Rotation.apply` — exactly overlapping. Caught by `@memcpy`'s own
overlap assertion (signal 6), which was luck; the dense path would have produced silently
wrong numbers. Fixed with two workspace buffers, and `Rotation` now asserts disjointness
explicitly so this class of mistake fails loudly rather than quietly.

**Workspace design.** Encode/decode take a caller-owned `Workspace`, so they allocate
nothing and can run concurrently against one quantizer. This is the pattern P1 needs
anyway for per-thread scratch.

---

## <a name="71b1e1b"></a>`71b1e1b` — quant: TurboQuant_prod with QJL residual sketch

756 lines, 13 tests. Algorithm 2, and the most interesting result of the session.

### The deviation: an orthogonal sketch

The paper draws `S` with i.i.d. N(0,1) entries and normalizes by `√(π/2)/d`. We use an
orthogonal `S'` (a second `Rotation`), for the same O(d log d) / nothing-to-store reasons
as Π. That turns out to be better in **two independent ways**.

**1. Exact normalization instead of asymptotic.** Rows of a Haar-orthogonal matrix are
uniform unit vectors, and for such a row `r` and unit `x`:

```
E[ sign(rᵀx) · (rᵀy) ] = c_m · ⟨x, y⟩,   c_m = E|rᵀx|
```

*exactly* — decompose `y` along `x` plus a perpendicular part, whose contribution vanishes
by symmetry. And `c_m` is precisely the mean absolute value of the sphere-coordinate
density, `2·∫₀¹ t·f(t)dt`, which `density.moment` already gives in closed form. The paper's
`√(π/2)/√m` is the large-`m` limit of `1/(m·c_m)`. At m=64–128 the two differ by 0.1–5%,
so this matters exactly where the KV path lives.

**2. ~2.7× lower distortion.** Orthonormal rows fix `Σᵢ(rᵢᵀq)² = ‖q‖²` *exactly* rather
than letting it fluctuate; the removed variance appears directly in the estimator.

### How the improvement claim was verified

"We beat the paper" is the kind of claim that is usually a bug, so it was not accepted from
our own numbers. The paper's Gaussian construction was implemented in the same harness and
run on the same vectors:

| sketch | `Var·m` at b=1 | slope (unbiasedness) |
|---|---|---|
| Gaussian (paper's) | **1.551** | 1.039 |
| Orthogonal (ours) | **0.556** | 0.995 |
| *paper's published figure* | *1.57* | — |

The Gaussian arm reproducing the published 1.57 to within 1% is what makes the orthogonal
number trustworthy. Per bit-width, `D_prod·d`:

| b | paper | ours | ratio |
|---|---|---|---|
| 1 | 1.57 | 0.567 | 2.77 |
| 2 | 0.56 | 0.207 | 2.71 |
| 3 | 0.18 | 0.068 | 2.64 |
| 4 | 0.047 | 0.020 | 2.41 |

Tests pin *our* measured values as regressions and separately assert `ours < published`,
which is the claim that actually matters.

**Unresolved and mildly tantalizing:** the b=1 value measures `0.5658 ± 0.0033` at 60k
trials, against `π/2 − 1 = 0.57080`. That is 1.5σ — suggestive of an exact constant but not
conclusive, so it is *not* asserted as one. Worth a proper derivation sometime: if
`Var·m = π/2 − 1` exactly for an orthogonal sketch, that is a clean small result. Note the
Gaussian case is exactly `π/2`, so a "−1" is plausibly the removed row-norm variance.

### A small arithmetic error in the paper

Verified after the fact, while checking that `D_prod(b) = (π/2)·D_mse(b−1)` held. The
paper's quoted `D_prod` values propagate its own *rounded* `D_mse` values rather than the
exact ones:

| b | paper `D_prod` | correct `(π/2)·D_mse(b−1)` | error |
|---|---|---|---|
| 1 | 1.57 | 1.571 | 0.1% |
| 2 | 0.56 | 0.571 | 1.9% |
| 3 | 0.18 | 0.184 | 2.5% |
| 4 | **0.047** | **0.054** | **13.4%** |

Root cause is upstream: `D_mse(3)` is quoted as `0.03` when it is `0.034548` — one
significant figure, 13% low. Then `(π/2)·0.03 = 0.0471`, which is exactly the quoted 0.047.

Consequence: none that matters. These are illustrative "finer-grained values", not the
theorem; the bound `√3·π/2 · 4^−b` holds throughout. Recorded only so that a future
comparison against `0.047` is not mistaken for a bug on our side.

**Not to be confused with the constant in [8ff805c](#8ff805c).** That one was *my* error —
`pdftotext` mangled `\frac{\sqrt{3}\pi}{2d}` into a stacked fraction under a displaced
radical, and I reconstructed it wrong. The paper is correct there.

### A statistics bug in my own test

The test for "MSE-only estimation is biased by 2/π" first measured **0.194** against an
expected 0.6366. The implementation was fine; the *test* was wrong. It averaged per-trial
ratios `estimate/truth`, and for random unit vectors in dimension d the true inner product
is about ±1/√d ≈ ±0.044 — a near-zero denominator, so the mean of ratios is numerically
worthless. Replaced with a least-squares slope `Σ(est·truth)/Σ(truth²)`, which reads
0.6366 as predicted. All bias measurements now use the slope.

**Worth internalizing:** this failure looked exactly like an implementation bug and was not
one. The tell was that the *other* unbiasedness test (which used a different statistic) was
passing simultaneously.

### Other notes

- **b=1 means sketch-only.** `Codebook` supports 0 bits (a single level at the density
  mean), so `prod` at b=1 spends its whole budget on the sketch. Not a degenerate guard —
  a real configuration, and one of the paper's four quoted values.
- **Query path is linear in q**, with no hidden normalization, so unnormalized queries give
  unnormalized inner products. There is a test.
- **A dead line survived into the first version** of `decode` — an `applyInverse` into
  scratch that was never read. Harmless, which is why the round-trip test still passed.
  Removed. Tests passing is not evidence that code is dead-free.

---

## P1 (in progress)

### <a name="becbc6a"></a>`becbc6a` → revised by `HEAD` — code layout

**Landed a dimension-major FastScan block, then reversed it while writing the kernel.** Worth
recording as a reasoning error rather than quietly rewriting history.

The blocked layout was carried over from product quantization without checking whether its
premise held. It does not. FastScan is dimension-major because PQ's per-subspace LUT is
*query-dependent* (`LUT_m[k] = ⟨q_m, centroid_{m,k}⟩`), so applying it needs all vectors' codes
for one subspace together. TurboQuant's codebook is scalar, so the table **factorizes** into
`LUT_j[k] = p_j · c[k]` — a single 16-entry, query-independent table that lives in one register
for the entire scan, times a scalar.

Consequence: the scan is a per-vector dot product along dimensions, which is what `SDOT`/VNNI
accelerate, and row-major storage is both faster and simpler (no transpose at insert).

| layout | ops/dim at d=1024,b=4 | effective |
|---|---|---|
| row-major + SDOT | 0.19 | ~37 GB/s — memory-bound ✅ |
| dimension-major + f32 widen | 0.56 | ~3× more — compute-bound ❌ |

This also retroactively explains turbovec's reported `SDOT`/`SMMLA`, which never fit a
dimension-major layout and should have been a clue at planning time.

**Lesson:** "adopt the known-good layout from the adjacent problem" skipped checking whether the
property that motivated it (query-dependent per-subspace LUTs) actually holds here. It doesn't,
and a scalar codebook is a *simpler* problem than PQ, not merely a different one.

### <a name="e5fbc1a"></a>`e5fbc1a` — vectorized threshold encoder

Measured, 4M coordinates, ReleaseFast: **1.50 Gcoord/s at b=4 (11×)**, **6.87 Gcoord/s at b=2
(36×)**. The multiples flatter it — the scalar baseline is a branchy early-break loop written for
clarity, so much of the gap is misprediction. Absolute rates are the number to track.

Two things dropped from the plan, both for the same underlying reason:
- **The threshold binary search** (DESIGN.md §4.1, ~3b ops/element) does not vectorize: each round
  probes `thresholds[idx]` at a *per-lane* runtime index — a gather. No portable spelling, and
  slow where it exists. Linear compare-sum uses only broadcasts.
- Full unrolling stops at b=4; b=8 is 255 broadcast constants and 255 compares per specialization,
  which pushed compile time past two minutes.

Tests cover b=1..5, not 1..8 — a b=8 Lloyd solve is ~27k iterations (see [8ff805c](#8ff805c)) and
b=5 already exercises both kernels. First version of these tests didn't think about that and made
the suite take minutes.

---

## Open items for P1

Ordered by how much I would want to resolve them before building on top:

1. **Non-power-of-two padding wastes bits.** d=768 → 1024 is 33% overhead. Block-FWHT
   (`d = m·2^a`) per DESIGN.md §1.4. Affects real embedding dimensions (768, 1536, 3072 are
   fine; 200, 300, 768 are not all clean).
2. **Bit-packing is not implemented.** Codes are one byte per coordinate. `quant/packing.zig`
   plus the 32-vector blocked, dimension-major layout is the actual P1 deliverable.
3. **Codebooks are solved at runtime.** Fine at these sizes but wants comptime tables,
   particularly since the KV path needs per-(layer, head) configurations.
4. **Derive the orthogonal sketch's variance** rather than measuring it, both to justify
   the ~2.7× claim properly and to settle the `π/2 − 1` conjecture (above).
5. **turbovec has not been run yet.** The plan commits to running it in our own harness
   rather than comparing against its README. Also: find out what its `calibrate()` does —
   TurboQuant is data-oblivious by construction, so a calibration step implies something
   outside the paper, possibly the outlier-channel split that the KV path needs anyway.
6. **Low-dimension behaviour is untested below d=64.** The KV path lives at 64–128 and the
   near-independence argument weakens as d shrinks. DESIGN.md §10 risk 2.
7. **Lloyd's solver is slow above b=6** (Max's shooting method, above). Only matters if we
   ship bit-widths past the shuffle-LUT range.

## Things deliberately not done

Recording these so they are not mistaken for oversights:

- Inter-round permutation in the RHT — measured as unnecessary ([6711abe](#6711abe)).
- Entropy coding of code indices — the paper measures ~5% at b=4 and declines it for speed;
  we agree.
- SIMD anywhere. P0 is the correctness oracle that P1's kernels get checked against, so it
  is written for obviousness, not speed.
