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

### <a name="scan"></a>`HEAD` — scan kernel, and a 14× one-line fix

The kernel itself is unremarkable: mask nibbles, two table lookups, `smlal` into i16, widen to
i32, reduce once per vector. What is worth recording is how nearly it shipped at a fifth of its
speed.

**LLVM lowers the portable lookup form to one instruction — until you inline it.**
`inline for (0..16) |i| out[i] = table[idx[i]]` compiles to exactly `tbl` (aarch64), `pshufb`
(SSSE3), or `vpshufb` (AVX2) when the result is *returned*. Inlined into this kernel, where the
result immediately feeds a widening multiply, LLVM folds each `extractelement` into the consumer's
`sext` and never forms the vector-build pattern — emitting **32 scalar `umov`/`bfxil` pairs per
chunk instead of one `tbl`**.

Fix is an empty `asm` with a vector-register constraint (`"+w"` on ARM, `"+x"` on x86). It emits no
instruction and blocks the fold.

| | before | after |
|---|---|---|
| throughput | 2.87M vec/s | **39.8M vec/s** |
| code bandwidth | 1.5 GB/s | **20.4 GB/s** |
| ns/vector (d=1024) | 349 | **25.1** |
| vs unquantized f32 brute force | 1.2× | **17.7×** |

**How it was caught, and how it nearly wasn't.** Every test passed both before and after — the
kernel was always *correct*. What flagged it was the benchmark showing the int8 quantized scan
barely beating an f32 brute-force loop, which is structurally implausible: 8× less memory traffic
should not buy 1.2×. Disassembly then showed no `tbl` at all.

**This is a fragile trick** — it depends on LLVM's fold behaviour and could regress on a compiler
update, silently, at ~4× cost. `bench/scan_bench.zig` is the guard; there is no unit test that can
see it. Worth re-checking on every Zig upgrade.

**Remaining headroom.** 20.4 GB/s against a predicted 37. The gap is four `saddw` widenings per
chunk, forced by widening i16→i32 every chunk. Products reach 127·127 = 16129 and two already sit
at 32258, just inside i16's 32767, so no more can be batched at full range. Shrinking the centroid
range to ±32 would allow four chunks per widen at some precision cost — measure before taking it.

### <a name="tests"></a>`HEAD` — integration and property tests

The unit tests each verify a module against its own oracle. Nothing verified that the modules agreed
with *each other*, and nothing verified the property the library exists to provide: that top-k
retrieval returns the right neighbours. Distortion bounds are a proxy for recall; recall is the
thing itself.

Added `tests/pipeline.zig` (seams + recall) and `tests/invariants.zig` (property sweeps +
adversarial inputs), wired as separate build targets so they exercise the public API rather than
reaching into internals. 117 tests total.

**Found a real bug immediately.** The dimension sweep crashed at `dim < 3`: `padded` came out below
3, and `Density.sphereCoord` asserts `dim ≥ 3` because `(1−t²)^((d−3)/2)` is unbounded below that.
Failure mode was an assert three modules from the call site. Fixed by flooring `padded` at 4 —
zero-padding a 1-D vector into R⁴ is well defined and costs three wasted codes, which is strictly
better than crashing. Nothing in 101 prior tests touched `dim < 4`.

**Measured recall (10k corpus, 200 queries, `prod` estimator):**

| d | bits | data | 1@1 | 1@10 | 1@100 |
|---|---|---|---|---|---|
| 256 | 2 | uniform | 0.285 | 0.805 | 0.985 |
| 256 | 3 | uniform | 0.525 | 0.975 | 1.000 |
| 256 | 4 | uniform | 0.685 | 1.000 | 1.000 |
| 256 | 4 | clustered | 0.410 | 0.940 | 1.000 |
| 1024 | 4 | clustered | 0.515 | 0.960 | 1.000 |
| 128 | 4 | clustered | 0.460 | 0.920 | 1.000 |

Recall is monotone in both bits and k in every configuration, which is good evidence the pipeline is
correct and that clustered data is genuinely harder rather than buggy.

**Do not read these as a comparison against the paper.** The paper's numbers are on real embeddings
(DBpedia/OpenAI, GloVe); "clustered" here is synthetic and deliberately adversarial — 32 tight
clusters produce many near-ties. Comparable numbers require the real datasets, which is still an
open P1 item.

**Worth noting for the index design:** 1@1 on clustered data is only 0.41–0.52 even at b=4. Top-1
retrieval will need a rerank stage; 1@10 ≥ 0.92 means a small candidate set suffices, which is
exactly what rerank assumes.

Test thresholds are set from measurement with ~3 standard errors of margin, and the measured value
is written next to each so a future tightening does not have to re-derive it.

### <a name="sketch"></a>`HEAD` — QJL sign-dot kernel, and a bit-width mismatch

The scan kernel computed only `⟨p, ỹ⟩`. The full `prod` estimator is
`‖x‖·[⟨p,ỹ⟩ + γ·scale·⟨S'p, qjl⟩]`, so the fast path was returning exactly the MSE-only estimate
that `prod` exists to correct — the one measured biased by 2/π. Wiring an index to it would have
silently reintroduced that bias while every test still passed. `simd/sketch.zig` closes it: bit
expansion by broadcast + selector + compare (3 ops, and `@shuffle` with a comptime mask is portable
where a runtime-indexed lookup is not), then `@select` and i16 accumulation.

**Discovered while doing it: `prod` at total width `b` uses `b−1` MSE bits, so the headline 4-bit
configuration produces 3-bit codes — and the scan kernel only handles 4-bit nibbles.**

| prod b | mse bits | bits/coord | canVectorize |
|---|---|---|---|
| 2 | 1 | 2.00 | ✗ |
| 3 | 2 | 3.00 | ✗ |
| 4 | 3 | 4.00 | ✗ |
| 5 | 4 | 5.00 | ✓ |

Only `b=5` vectorized. The obvious fix — pad 3-bit codes into nibbles — is **strictly dominated**,
and measurement says so plainly (clustered, d=1024):

| prod b | bytes/vec | 1@10 |
|---|---|---|
| 3 | 384 | 0.690 |
| 4 | 512 | 0.960 |
| 5 | 640 | 0.995 |

Padding `b=4` costs 640 B for the same 0.960, where `b=5` already gives 0.995 at that size. So
padding is off the table. But `b=4` is genuinely on the Pareto frontier (8× compression, 0.96), so
it earns a real 3-bit kernel — deferred to its own commit rather than bolted on here.

Next kernel work should extend `scan` to the byte-divisor widths {1, 2, 4}, which are a uniform
shift-and-mask and cover prod b ∈ {2, 3, 5}, leaving only b=4 on the scalar path.

**Test-tolerance note, twice now.** Two sketch tests failed at a 0.05 absolute tolerance. Not bugs:
int8 error over a `dim`-term signed sum grows as √(dim/12)·step, which is ~0.13 at d=256, so 0.05 was
never achievable. Replaced with a derived `quantizationTolerance(dim, scale)`. Same mistake as the
Lloyd-Max table tolerance in [8ff805c](#8ff805c) — when a tolerance is a guess rather than a bound,
it tests the guess.

### <a name="flat"></a>`HEAD` — FlatIndex, top-k, and the cost of the 3-bit gap

`index/topk.zig` and `index/flat.zig` compose the quantizer, packing, and both kernels into
something callable. 143 tests, all passing first run.

**The design's SIMD top-k gate is gone.** §4.4 specified comparing 32 block scores against a
broadcast threshold and `movemask`-ing survivors — a shape that only existed because
dimension-major blocks produced 32 scores at once. Row-major produces one at a time, so the gate is
a scalar compare. No loss: the win was never the compare, it was avoiding the heap, and a bounded
min-heap with a single-compare reject does that either way.

**Measured the b=4 fallback, and it overturns the earlier decision to defer a 3-bit kernel**
(d=1024, n=100k, k=10):

| bits | B/vec | vectorized | QPS | µs/query | 1@10 (uniform) |
|---|---|---|---|---|---|
| 2 | 256 | yes | 189.8 | 5,267 | 0.700 |
| 3 | 384 | yes | 192.9 | 5,185 | 0.980 |
| **4** | **512** | **no** | **7.8** | **127,776** | **1.000** |
| 5 | 640 | yes | 197.1 | 5,075 | 1.000 |

**25× slower**, not a mild degradation — the scalar path makes b=4 unusable. Earlier I judged the
3-bit kernel deferrable because padding-to-nibbles was dominated by b=5. That reasoning was about
*storage*, and it was fine as far as it went; it simply did not price the fallback. With clustered
recall at b=4 = 0.960 against b=3 = 0.690, b=4 is the best realistic-data operating point at 8×
compression, and it currently cannot be used.

Sketch of the fix: 3-bit codes cannot be extracted by shift-and-mask because they straddle bytes,
but a bit-plane layout (3 planes of 16 bits per 16 codes) can reuse `sketch.zig`'s bit expansion —
roughly 17 ops per 16 codes against b=4's ~6, so ~3× slower per code rather than 25×. Not parity,
but usable.

Also worth noting: index QPS of ~190 at n=100k is 19M vec/s, against the raw scan bench's 40M. The
index adds the sketch term, which roughly doubles per-vector work. Consistent, and a useful check
that nothing unexpected is being paid.

### <a name="bitplane"></a>`HEAD` — 3-bit kernel via bit-planes

Closes the 25× gap at `bits=4`.

**Layout.** 3-bit codes straddle bytes, so shift-and-mask cannot reach them. Instead they are
stored as *bit-planes*: for each group of 16 codes, three 2-byte planes holding bit 0, bit 1, and
bit 2. Unpacking is then three bit expansions (the same primitive the QJL sketch already needed,
now factored into `simd/bitmask.zig`) recombined by weight. Costs `bits` per coordinate exactly —
grouping by 16 rather than rounding to bytes means no waste at any dimension that is a multiple of
16, which every padded dimension is.

**Result** (d=1024, n=100k, k=10):

| bits | B/vec | before | after | 1@10 |
|---|---|---|---|---|
| 3 | 384 | 192.9 | 193.1 | 0.980 |
| **4** | **512** | **7.8** | **110.1** | **1.000** |
| 5 | 640 | 197.1 | 198.1 | 1.000 |

**14×**, and better than the ~3×-slower-per-code estimate predicted. b=4 is now 1.8× behind b=5
rather than 25×, which makes 8× compression at 0.96 clustered recall a usable operating point
rather than a theoretical one. Encode cost rose (24.6 µs/vector against ~16) because the bit-plane
writer is a scalar bit loop; encoding is not the hot path, and it is vectorizable if it ever
matters.

**A real bug the sweep caught.** The first dispatch routed 5-, 6-, and 7-bit codes into
`scorePlanes` too. But `tbl`/`pshufb` index a **16-byte** register, so the shuffle table holds 16
levels — **four bits is a hard ceiling on the vectorized path**, independent of layout. Codes above
15 indexed past the end of the table. This is a property of the instruction, not of the packing,
and it now has a name (`max_table_bits`) and a test rather than being implicit in "b ≤ 4 works".

Practical effect: prod b ∈ {2,3,4,5} vectorize; b ≥ 6 uses the exact scan. No loss worth chasing,
since b=5 already reaches 0.995 recall.

**On reversing the earlier deferral.** The measurement that changed the call was QPS, not recall or
storage — the earlier analysis compared *bytes* and correctly concluded padding was dominated, but
never asked what the fallback cost. A dominance argument over one axis says nothing about the axis
you did not measure.

### <a name="sift"></a>`HEAD` — first real-data benchmark (SIFT10K), and a buffer overrun

Everything before this used synthetic corpora whose difficulty I chose. ANN_SIFT10K is 10,000 base
vectors at 128d with 100 queries and published top-100 exact L2 ground truth.
`tools/fetch_datasets.sh` retrieves it; `data/` is gitignored.

| bits | total B | ratio | median rank | p90 | worst | 1@10 | R@10 | QPS |
|---|---|---|---|---|---|---|---|---|
| 2 | 36 | 14.2× | 2 | 31 | **92** | 0.75 | 0.421 | 11939 |
| 3 | 52 | 9.8× | 1 | 14 | **81** | 0.88 | 0.571 | 12159 |
| 4 | 68 | 7.5× | 0 | 3 | **9** | 1.00 | 0.740 | 7665 |
| 5 | 84 | 6.1× | 0 | 2 | 8 | 1.00 | 0.839 | 12449 |
| 6 | 100 | 5.1× | 0 | 1 | 3 | 1.00 | 0.902 | 299 |

#### Reporting `1@k` was hiding the margin

The first version of this table reported `1@10` and `1@100`, both saturated at 1.00, and I read that
as a strong result — going as far as recommending a rerank stage sized at 100 candidates on the
strength of it. Jeff pushed back that 1.00 looked like a testing error. The harness turned out to be
correct, but the reading was wrong in a way that matters more:

- b=2's worst query lands at rank **92 of 100**. `1@100 = 1.00` is true by eight places.
- b=4's worst lands at rank **9**. `1@10 = 1.00` is true by *exactly one position*.

**A saturated metric carries no information about margin.** Both numbers are one unlucky query from
reading 0.99, and a rerank depth chosen from them would have been fitted to a cliff edge. The table
now reports the rank distribution, which is what a candidate count actually has to be sized against.

`bench/sift_verify.zig` is the harness validation, kept permanently:

| check | result |
|---|---|
| ground truth vs. brute force | 100/100 agree — parsing and id mapping correct |
| shuffled-corpus control | 1@100 = 0.00 — the metric is not trivially satisfied |
| dist(NN)/dist(100th) | 0.617 mean, 0.176 worst — SIFT's NN is genuinely well separated |

That last row is the real explanation: `1@100` saturates because SIFT10K's nearest neighbour is
38% closer than its 100th on average. That is a property of the corpus, not evidence about the
quantizer.

#### The buffer overrun this uncovered

Chasing an `OutOfMemory` at b=6 found a genuine memory-safety bug. `Searcher.init` built the int8
shuffle table unconditionally, but `Table.init` asserts `centroids.len <= 16` — the hard
`max_table_bits` limit. At b=6 the MSE stage has 5 bits, so **32 centroids were written into a
16-entry array**. Debug catches it as an assert; ReleaseFast compiles the check out and corrupts
adjacent memory, which surfaced as a spurious allocation failure and then a crash.

Missed because the vectorization test stopped at b=5 and the wider-bit test only checked storage
sizing without ever constructing a `Searcher`. There is now a test that actually searches at every
width from 2 to 7.

Two lessons worth keeping. **Benchmarks run in ReleaseFast, so a bug they trigger appears as
nonsense rather than as a panic** — `-Dbench-opt=Debug` now exists for exactly this, and each bench
has its own step so one can be run alone. And **an assert only protects the builds that keep it**:
this one was correct, documented, and useless in the configuration that mattered.

#### What these numbers are not

Still no PQ, RaBitQ, or turbovec baseline run here, so nothing above supports a parity claim. And
SIFT10K at 10k vectors is much easier than SIFT1M.

Also corrected here: `bytesPerVector` excluded the per-vector `norm` and `gamma`, which were f32 —
under 2% of a d=1024 vector but 11% of a d=128 one, the KV regime. They are f16 now as DESIGN.md
always specified, and the accessor includes them. b=4 is 7.5×, not the 8× reported earlier.

### <a name="zig016"></a>`HEAD` — port to Zig 0.16

The toolchain on this machine moved to 0.16.0 and 0.15.2 was removed, so this was forced rather
than chosen. Two breaking changes mattered.

**Runtime vector indexing is gone**, and it was the basis of the byte-table lookup:

```zig
inline for (0..16) |i| out[i] = table[idx[i]];  // idx[i] is runtime → error in 0.16
```

Five sites, but only one in production code (`scan.lookup`); the rest were tests indexing a vector
with a loop variable, fixed by coercing to an array first.

The obvious port — stage the vector through an array, where runtime indexing is still legal —
compiles and **loses the instruction**: it spills to the stack and issues scalar loads. Measured
previously at ~14× slower. So `simd/shuffle.zig` now holds per-architecture inline assembly (`tbl`
on aarch64, `pshufb` on SSSE3) with a scalar fallback.

**This vindicates the original design and refutes a simplification I made.** DESIGN.md §4.2
specified exactly this dispatch layer; during P1 I deleted it on the grounds that LLVM
pattern-matched the portable form, which was true and did not survive one minor release. The
[scan-kernel note](#scan) called the barrier trick "fragile, compiler-version-dependent" and
predicted a silent 4× regression on upgrade. The prediction was right about fragility and wrong
about the failure mode: it broke as a **compile error**, which is the good outcome. A silent
slowdown would have needed the benchmark to catch it.

**Other 0.16 removals hit only benchmarks:** `std.heap.GeneralPurposeAllocator` (→ `smp_allocator`),
`std.time.Timer` (→ a small `bench/timer.zig` over the C monotonic clock), and `std.fs.cwd()` (file
I/O now needs an `Io` instance, constructed via `std.Io.Threaded`).

**Verification that the port preserved behaviour**, which matters more than the port compiling:

- 155/155 tests pass in Debug *and* ReleaseFast.
- **Golden vectors pass unchanged** — encodings are bit-identical across a compiler major version.
  This is the first real payoff from pinning them by hash rather than recomputing.
- Disassembly shows the hot loop unchanged: `tbl.16b` ×2, `smull`/`smlal`, `saddw`.
- SIFT10K reproduces exactly: same ranks, same recall at every bit-width.
- Index QPS within noise: 189/183/110/196 against 190/193/110/197.

The raw scan bench reads 18.4 GB/s against 20.9 before, but the f32 brute-force baseline in the same
run dropped from 9.2 to 5.5 GB/s, so the machine is slower right now rather than the kernel having
regressed — the *ratio* improved from 17.7× to 26.6×. Worth remembering that absolute throughput
numbers across sessions are not comparable; ratios measured in the same run are.

### <a name="batch"></a>`HEAD` — batched search, and a claim of mine that was wrong

Added `searchBatch`, which scores several queries in one pass over the corpus so a vector's codes
are fetched once and scored `n` times. Verified exact agreement with the single-query path across
all three metrics.

**It gives 1.01×.** Not 5×, not 2× — nothing.

| corpus | single | batched | gain |
|---|---|---|---|
| 26 MB | 190.0 | 188.9 | 0.99× |
| 52 MB | 196.6 | 197.3 | 1.00× |
| 103 MB | 210.5 | 212.4 | 1.01× |

**That falsifies "the scan is memory-bound", which I had claimed repeatedly** — in DESIGN.md §4.2,
in the layout-revision commit, and in the scan-kernel notes. It came from dividing throughput by
corpus size, getting ~21 GB/s, and calling that memory-bound. **A bandwidth figure is not evidence
of a bandwidth bound.** The test is whether reducing the traffic helps, and it does not: at 103 MB,
far beyond any cache here, batching 32 queries per pass changes nothing.

So the kernel is **compute-bound**, and the levers are fewer ops per vector or more cores — not
better locality. Threading should now scale close to linearly, which it would not if memory were
the constraint.

**A second wrong prediction, worth separating.** I expected batching to explain turbovec's 5.2×
batched-versus-single advantage. It cannot, since batching does nothing for us. That gap is almost
certainly Python FFI call overhead amortized across 1000 calls — not a kernel property. The
comparison that matters is single-query against single-query: **6,072 QPS for turbovec against
1,381 for us**, a real 4.4× that batching will not close.

`searchBatch` is kept anyway: it is a natural API, it is correct, and it is the right shape to
parallelize over. But it was built on a mistaken premise and did not deliver what it was built for.

### <a name="threads"></a>`HEAD` — query-parallel search

Threads own disjoint **queries**, not disjoint corpus shards. Each thread therefore reads the whole
corpus — which would be the wrong design if the scan were memory-bound, and is the right one now
that [batching showed it is not](#batch). No shared mutable state, no top-k merge, no false
sharing, and results are bit-identical to the serial path.

| corpus | 1 thread | 10 threads | speedup |
|---|---|---|---|
| 26 MB | 192 | 1118 | 5.8× |
| 52 MB | 199 | 1198 | 6.0× |
| 103 MB | 215 | 1246 | 5.8× |

**5.8–6.3×, not 10×.** Apple Silicon mixes performance and efficiency cores, so ten "cores" are not
ten equal ones, and the speedup tracks the performance-core count. Worth noting the other end too:
at 103 MB and 1246 QPS the corpus traffic is ~128 GB/s, which is near this machine's memory
bandwidth — so the scan is compute-bound on one core and approaches a bandwidth wall on ten. Both
statements are true and neither generalizes to the other.

Threads are spawned per call, which is only viable because a batch is long enough to amortize it:
~160 ms of work against ~30 µs of spawn cost per thread. Spawning per *query* would not pay, and
`std.Thread.Pool` no longer exists in 0.16 (concurrency moved behind `Io.Group`). Migrating to that
is the idiomatic path if per-call spawning ever becomes the bottleneck.

**Where throughput now stands** (nytimes-256, 132 B/vector):

| | QPS | note |
|---|---|---|
| zquant, 1 thread | 1,467 | |
| turbovec, 1 query at a time | 6,072 | **4.1× ahead per core** |
| zquant, 10 threads | 7,886 | beats turbovec's single-query rate |
| turbovec, batched | 31,700 | almost certainly also threaded |

Threading closes the gap against turbovec's *single-query* rate but not against its batched one. The
honest reading is that we remain **roughly 4× behind per core**, and that is now the clearest
remaining deficit — recall is competitive, per-core speed is not.

### <a name="sdot"></a>`HEAD` — SDOT, and a per-core comparison that was wrong

**Profiling first, since the last two guesses about this kernel were both wrong.** The scan was
running at **IPC 4.0** — essentially peak. It was not executing slowly, it was executing too many
instructions: accumulating through i16 costs `smlal`, `smlal2` and two `saddw` per 16 products,
where `SDOT` does the same work in one and accumulates straight into i32, with no overflow ceiling
to pair chunks around.

LLVM will not emit `SDOT` from the natural formulation — the 4-way grouping lowers to scalar `umov`
extraction — so `simd/dot.zig` is inline assembly, for the same reason `simd/shuffle.zig` is.

**Two attempts, and the first was a regression:**

| | scan | note |
|---|---|---|
| i16 accumulate (before) | 35.9M vec/s | |
| SDOT, one accumulator | 43.0M | +20%, but IPC fell 4.0 → 2.8 |
| SDOT, runtime-indexed accumulators | **24.7M** | **−42%** |
| SDOT, comptime-indexed accumulators | **67.9M** | **+89%** |

Fewer instructions exposed a **dependency chain**: every `sdot` reads the accumulator it writes, so
one register serialized the loop at multi-cycle latency. Independent accumulators fix it — but only
if the slot is a *comptime* index. Selecting it with `(ch * groups + k) % accumulators` spilled the
accumulators to the stack and cost 42%. One accumulator per bit-field group (comptime `k`) keeps
them in registers.

End to end: index throughput at d=1024 rose 214.7 → 325.6 QPS single-thread, and on nytimes
1,467 → 2,024.

### The per-core comparison was wrong in our favour

I had reported turbovec as **4.1× ahead per core**, from its 6,072 QPS "one query at a time" against
our 1,467. That assumed one query at a time meant one thread. It does not:

| | QPS | cores busy | QPS/core |
|---|---|---|---|
| turbovec, one-at-a-time | 5,672 | 7.6 | 746 |
| turbovec, batched | 38,052 | 9.4 | 4,048 |
| zquant, 1 thread | 2,024 | 1.0 | **2,024** |

Measured with `process_time` against `perf_counter`. So we are **~2× behind per core**, not 4×, and
the absolute gap (38,052 against our 11,790) is as much a *scaling* problem as an efficiency one —
they saturate 9.4 of 10 cores where our 10 threads deliver 5.8× effective.

Note also that turbovec's per-core rate is 5.4× worse one-at-a-time than batched (746 against
4,048), which is the Python FFI overhead showing up as busy cores rather than useful work. Comparing
against the one-at-a-time figure flattered us in the other direction. **Cross-language throughput
comparisons need core accounting, not just wall time.**

### <a name="scaling"></a>`HEAD` — thread scaling, and the storage comparison was measuring the wrong thing

**Thread scaling is not the problem.** A sweep separates our implementation from the machine:

| threads | QPS | speedup | efficiency |
|---|---|---|---|
| 1 | 323 | 0.99× | — |
| 2 | 630 | 1.93× | 96% |
| **4** | **1249** | **3.83×** | **96%** |
| 6 | 1531 | 4.69× | 78% |
| 10 | 1929 | 5.92× | 59% |

96% efficiency across all four cores, then flattening exactly as threads spill onto the six
efficiency cores (`hw.perflevel0/1` = 4P + 6E). The 5.8× I called a scaling deficit is close to this
machine's ceiling. My hypothesis — static partitioning causing load imbalance on heterogeneous
cores — was wrong, and the sweep cost less than implementing the work-stealing fix would have.

### The storage comparison was measuring the wrong number

turbovec's throughput implied ~3.8 billion vector-scans/s, roughly 100% of theoretical `sdot` peak
— meaning it issues almost nothing *but* `sdot`, with no unpacking. Two checks explain how:

- **Not pruning.** QPS falls 1.87× per corpus doubling, so it is a genuine full scan.
- **Storing dequantized bytes in memory.** At `bit_width=4`, d=256:

| | per vector | bits/coord |
|---|---|---|
| serialized (`to_bytes`) | 135 B | 4.22 |
| **resident (RSS delta)** | **270 B** | **8.44** |

**I had been comparing their serialized size against our in-memory size.** At equal residency the
picture inverts: zquant holds **132 B/vector at R@10 0.917**, turbovec **270 B at 0.914**. Half the
memory for marginally better recall.

Their speed advantage is therefore a deliberate **2× space-for-time trade** — dequantized int8 needs
no `tbl`, so the kernel is pure `sdot` — not a better kernel. That is a legitimate design choice and
an option we could offer too: an expanded in-memory layout for callers who want throughput over
footprint. It is not evidence that our scan is 3× worse.

**Three framings of this comparison were wrong in a row**, each corrected by measurement: "4× behind
per core" (assumed their single-query path was single-threaded — it uses 7.6 cores), then "2× behind
per core" (per-core is ill-defined on 4P+6E), and now the storage baseline itself. The honest
end-to-end statement is the one that needs no per-core inference: **on a full machine, turbovec
reaches 38,052 QPS at 270 B/vector and zquant 11,790 at 132 B/vector.** Different points on the
space/time curve, not a like-for-like deficit.

### <a name="expanded"></a>`HEAD` — expanded residency: a measured negative

Built the space/time trade turbovec makes — dequantize to int8 at insert so the scan needs no table
lookup — expecting roughly 2× throughput for roughly 2× memory. Verified bit-identical to the
compact path first, so the timings mean something.

**It buys nothing.** nytimes, d=256, QPS single thread:

| bits | compact | expanded | |
|---|---|---|---|
| 2 | 1848 @ 36 B | 1881 @ 260 B | no gain, 7× memory |
| 3 | 1977 @ 68 B | 1904 @ 260 B | *slower* |
| 4 | 668 @ 100 B | 1911 @ 260 B | 2.9× — the bit-plane path |
| 5 | **1921 @ 132 B** | 1883 @ 260 B | no gain, 2× memory |

**"Unpacking is free" — corrected.** That was the first conclusion, drawn from the end-to-end table
above, and it was overgeneralized. Jeff pushed back that "free" sounded suspicious. Timing the
kernels in isolation (`bench/kernel_bench.zig`), with no index overhead in the path, splits the
answer by packing:

| codebook bits | packing | packed | expanded | ratio |
|---|---|---|---|---|
| 2 | sequential | 3.82 ns | 3.37 ns | 1.13× |
| **3** | **bit-plane** | **13.76 ns** | **3.44 ns** | **4.01×** |
| 4 | sequential | 3.28 ns | 3.38 ns | 0.97× |

For **sequential** packing unpacking really is free — at 4 bits the packed path is marginally
*ahead*, since the `tbl` issues alongside the `sdot`s rather than competing with them. For
**bit-plane** packing it costs **4×**: three bit expansions plus a weighted recombination per 16
codes is real work, not hidden work.

The end-to-end measurement was not wrong — the index's slow `bits=4` configuration is exactly the
bit-plane path, and the two agree. What was wrong was generalizing one number across both layouts.

**Method note.** An end-to-end benchmark can only bound the sum of its parts. Attributing a null
result to one part requires isolating that part, and doing so here reversed the conclusion for half
the configurations.

### <a name="noise"></a>Benchmark noise, and a retracted number

Following up on this produced a worse problem. I claimed index per-vector overhead was "~34%",
derived by comparing a kernel timing against an index timing **from a different run**. Checking the
run-to-run spread on the same binary:

| run | packed | expanded | ratio |
|---|---|---|---|
| 1 | 4.60 | 4.02 | 1.14× |
| 2 | 5.61 | 3.67 | **1.53×** |
| 3 | 4.76 | 4.71 | **1.01×** |
| 4 | 6.04 | 4.05 | 1.49× |

**The same ratio measured anywhere from 1.01× to 1.53× across four identical runs.** The "34%
overhead" was a cross-run subtraction and is **retracted** — it is not distinguishable from noise.
The single-sample "0.97×" and "1.13×" ratios in the table above were also inside that spread.

`bench/kernel_bench.zig` now reports the **minimum of seven trials** with the spread alongside.
Minimum is the right estimator here because noise only ever adds time — scheduling, frequency,
eviction — so the fastest observation is the least contaminated. With that:

| codebook bits | packing | packed | expanded | ratio |
|---|---|---|---|---|
| 2 | sequential | 6.64 (+49%) | 7.28 (+88%) | 0.91× |
| **3** | **bit-plane** | **15.06** | **3.47** | **4.34×** |
| 4 | sequential | 4.34 (+25%) | 4.36 (+33%) | 1.00× |

The conclusion survives — sequential unpacking is free, bit-plane costs ~4× — because that gap is
far outside the spread. Effects smaller than about 1.3× on this machine need min-of-N to see at
all, and several numbers earlier in this log were single samples that did not clear that bar.

The option is kept, documented as not recommended, because being able to re-run the comparison is
worth more than the API surface costs.

**A bug this caught, twice over.** Building the int8 table in `FlatIndex.init` overran a 16-entry
array for `bits > 5` — the identical mistake to the `Searcher.init` overrun from earlier. The
regression test written for *that* one caught *this* one immediately, which is the clearest argument
so far for writing the test rather than just the fix.

### <a name="tqplus"></a>`HEAD` — reading turbovec's source: what we got wrong, and an unresolved integration

turbovec is MIT-licensed with public source, so studying it is legitimate. Doing so answered the
open question about their calibration directly.

**Their `tqplus_shift` / `tqplus_scale` is per-coordinate shift *and* scale — the same transform I
tried twice and measured worse. The difference is entirely in how the pair is chosen.**

I fitted **mean and standard deviation**. They fit by **quantile anchoring**: map each coordinate's
empirical `p`-quantiles onto the codebook's outermost centroids, where `p = P(|x| ≤ c_outer)` under
the canonical marginal — ~0.93 at 2 bits, ~0.996 at 4. Their reasoning, which is exactly the failure
I hit:

> Values past the outermost centroid all collapse into one bucket with unbounded error, so the right
> anchor is the point where the codebook stops.

Matching σ says nothing about where the tails land relative to `c_outer`. They record a case where a
fixed anchor scaled 4-bit data 2× too far and dropped R@10 from 0.4835 to 0.1439.

**Implemented it, and it half-worked.** The fit is demonstrably right — on SIFT it **halves
reconstruction error**, ‖y−ŷ‖² going 0.00899 → 0.00436 — and the fitted parameters are sensible
(anchor probability 0.9967, scales 1.05–2.11, and the coordinates really are skewed: one had
quantiles at −2.42σ and +1.44σ).

**But recall got worse**: 0.869 → 0.811 at 68 B. Isolated as far as it went:

- Not the int8 query path — an exact f32 scan shows the same gap (0.871 → 0.811).
- Partly the α rescale, which is actively harmful here (0.736 → 0.812 when disabled) even after
  correcting its weighting to match the `1/scale` weights the estimator actually applies.
- Not the fit — reconstruction is better, measured directly.

So: **better reconstruction, worse retrieval.** That is the signature of MSE-optimal not being
MIPS-optimal — the error that matters for inner products is weighted differently from the error that
matters for reconstruction — but I could not close the gap between that observation and turbovec's
+8 points at 2 bits within a reasonable budget.

**Then found the actual bug.** The paradox — better reconstruction, worse retrieval — was not a
deep property of MIPS. A stale post-encode block was still running unconditionally and
**overwriting** the α that `encodeCalibrated` returns, treating it as a residual norm and applying
the uncalibrated formula. With `gamma = α ≈ 1` that reduces to `(1 + ‖ŷ‖² − 1)/2/‖ŷ‖² = 0.5`
exactly, which is the clean factor of 2 the diagnostic showed: stored α ≈ 0.50 against a correct
0.99–1.01.

It presented as "calibration hurts recall" because the halving carried a small per-vector wobble
(0.207–0.229) that reordered near-ties, rather than as a constant factor that would have been
harmless. Three hypotheses were tested and discarded first — int8 query quantization (exact f32
shows the same gap), the α weighting (real, but not this), and quantile noise from a small sample
(1024 rows and 10,000 rows agree to 0.001).

**What found it:** comparing the index's score against an explicit `⟨p, ŷ⟩` computed from the
reconstruction by hand. That is now a test. It would have caught this in seconds; every hypothesis
above was reasoning about the *symptom* instead.

**Where it lands after the fix.** The estimator now matches an independent reconstruction (index
0.836, manual 0.838, truth 0.834), and calibration is correct rather than harmful — but it is close
to **neutral**: +0.9 points at bits=3 on nytimes, ~0 at bits=5 on both corpora, slightly negative at
the lowest bit-widths on SIFT. turbovec gets +8 at bits=2 on SIFT from the same transform, so
something is still missing, and it is no longer the fit, the estimator, or the sample.

**What is worth keeping from this:**

1. The premise "we are missing a lot" was not what the measurements said, and reading their source
   confirmed the specific gap rather than a broad one: one calibration technique, on one axis.
2. Quantile anchoring is the right idea and provably improves reconstruction here. The unresolved
   part is how it composes with our `prod` estimator, which differs from theirs.
3. A better reconstruction that retrieves worse is a real result and points at anisotropic /
   score-aware quantization (ScaNN's insight) as the missing piece, not at a coding error.

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


## 04bf5e4..HEAD — the calibration correction had to be fit in the basis it is applied in

Quantile anchoring landed correct but near-neutral. The remaining error was in how α
composed with it, and it took reading turbovec's scoring site to see it.

Their kernel comment (`search.rs:588`) is the tell: *"fa already holds bias + Σ
scale*partial — only vec_scales left"*. The per-vector correction multiplies the **whole**
expression, shift bias included. Ours multiplied only half of it:

```
ours:   norm · (−⟨p,shift⟩ + α·⟨p,u⟩)          u = c/scale
theirs: norm · α · (⟨p,u⟩ − ⟨p,shift⟩) = norm · α · ⟨p, x̂⟩
```

And ours fit α in the shifted basis, `⟨y+shift, u⟩/‖u‖²`, while applying it to a sum that
is really about `x̂ = u − shift`. Both sides of that fit contain the shift, which dominates,
so α came out ≈1 whatever the codes did — the correction had quietly become a no-op.

**Neither half is the fix.** Measured separately on SIFT at bits=3:

| variant | R@10 |
|---|---|
| baseline (old basis, old scope) | 0.582 |
| new basis only | 0.600 |
| new scope only | **0.463** |
| both | **0.691** |

Scope alone is *worse than doing nothing*. This is one change, not two: α is a least-squares
coefficient, and it is only meaningful when fit against the same vector it multiplies.
Mixing the bases produces a coefficient that is wrong for the sum it scales.

**Result on SIFT** (anisotropic, where calibration has work to do): bits=2 0.285→0.357,
bits=3 0.582→**0.691**, bits=4 0.773→0.832, bits=5 0.869→**0.907**. That is the +8-point
class of gain turbovec gets, and it puts us ahead of them at both matched byte widths —
0.691 vs 0.6825 at 36 B, 0.907 vs 0.9036 at 68 B.

**nytimes stays neutral** (+0.7 at bits=3, ~0 elsewhere), and that is the expected shape
rather than a disappointment: nytimes is near-isotropic after rotation, so there is little
per-coordinate mis-fit for the shift/scale to remove. Calibration pays where the data is
anisotropic.

**On the earlier mechanism claim.** I had written that α was "a near-tautology ≈1" and
expected the fix to make it a real shrinkage factor. Measured, α is 0.98–1.00 both before
and after — the magnitude barely moves. The gain comes from α being the *right* coefficient
for the sum it scales, not from it becoming small. I only found that by isolating the two
changes; the recall win alone would have let the wrong story stand.

**Test.** `calibrated alpha is the least-squares fit in the pre-shift basis` asserts α against
its definition. The existing score-vs-reconstruction test cannot catch this: under the wrong
basis the score stays *proportional* to ⟨p,x̂⟩, just with a wrong constant, so it passes with
a ratio near 1. Verified the new test fails on the old basis before keeping it.


## 8824e43..HEAD — the scan was load-issue bound, and batching had never exploited it

The scan sat at a flat ~48–65 G dim/s single-threaded regardless of dimension, dataset, or
residency. That flatness is the tell: a memory-bound kernel would move with corpus size, and
a compute-bound one would move with bit width. A kernel pinned on *load issue* does neither.

The one-query kernel spends two loads per `SDOT` — the code chunk and the query chunk — which
is exactly one `SDOT` per cycle on the load ports.

Batching had been measured at 1.00× and written off. The reason it did nothing is that the
batch loop called the one-query kernel once per query, re-walking the vector and reloading
both operands each time. It amortized cache lines, which were never the constraint.

**Three restructurings, each measured:**

| kernel | loads per SDOT | d=784, G dim/s |
|---|---|---|
| one query | 2.0 | 60.7 |
| query group, Q=8 | 1.125 | 148.2 |
| tiled, V=4 × Q=4 | 0.625 | **206.8** |

Tiling the stored vectors is what breaks below one load per SDOT — the query chunks stay in
registers and are reused across V vectors. The query-only form cannot reach that at any Q.

**Register budget picks the tile, not taste.** V·Q accumulators + Q query chunks + V code
chunks must fit in 32 vector registers. 4×8 needs 44 and collapses to 153 G dim/s — the
spill. 4×4 needs 24, is fastest at both d=256 and d=784, and is the most stable candidate
(1% spread against 9% for 2×8).

**A load that isn't in the source.** Hoisting the query base pointers out of the loop was
worth as much as the restructuring: reading `queries[q].data` inside the loop makes the
slice header itself a load — two per query per chunk, worse than the kernel it replaces.
Batched went 1196 → 1652 QPS on that one change.

**Compact benefits more than expanded.** Its shift/mask/table-lookup is query-independent,
so the group amortizes the whole unpack rather than just a load. Keeping one accumulator per
(vector, query) instead of per (query, bit-field group) is what lets Q stay at 4 for every
bit width. Integer addition is associative, so that fold is bit-identical — every
multi-query and tiled kernel is tested for *exact* equality against its per-query
counterpart, not a tolerance, since they issue the same SDOTs and an approximate bound would
hide precisely the reordering bugs worth catching.

**Where it lands.** SIFT10K at 68 B: 203,542 QPS against turbovec's 87,522, at R@10 0.907
against 0.904 — we now win that corpus on recall, memory, and throughput at once.
nytimes-256 at 132 B: ~21,100 parallel QPS at R@10 0.916, against turbovec's 38,052 at 270 B
and 0.914. Half their memory, equal recall, still 1.8× behind on throughput.

**Measurement caveat, stated because it matters.** The parallel numbers drift with thermal
state: three consecutive runs of the identical configuration gave 21,096 → 18,950 → 15,525.
Single-thread batched throughput is stable to about ±5% and is the honest number to compare;
the parallel figures above are first-run (coldest) and should be read as an upper bound.
The corollary is that the remaining nytimes gap is partly parallel efficiency — 3,845 batched
single-thread against ~21,100 across 10 threads is 5.5×, not the ~8× the core count suggests.


## Parallel scaling: the anomaly was the benchmark, not the scheduler

The thread sweep showed ten threads running *slower* than eight — 3,269 against 3,739 QPS
at d=1024, reproducible across runs. A work-conserving split cannot do that, so the reading
was that static equal shares were stranding work on the four performance / six efficiency
core split, with every thread waiting on the slowest.

**That was wrong, and implementing the fix is what showed it.** Replacing the static split
with threads claiming chunks from an atomic cursor made things *worse* by about 15% at every
thread count, including one thread — which contends with nothing. The cause is that each
`searchBatch` call is a full pass over the corpus, so chunks smaller than the batch multiply
the per-vector overhead. Measured back-to-back against a stashed baseline, because the
machine drifts enough (740 → 635 at one thread across fifteen minutes) that comparing to an
earlier run would have credited the drift to the change.

The real cause was the benchmark. It swept over the 100 ground-truth queries, so at ten
threads each thread got a batch of ten while at eight it got twelve or thirteen. The
per-vector cost amortizes over the batch, so raising the thread count *shrank* the unit of
work. The sweep was measuring batch efficiency and calling it scaling.

Sweeping over a replicated 640-query set, so every thread gets a full 32-query batch at
every thread count:

| threads | QPS | speedup | efficiency |
|---|---|---|---|
| 1 | 744 | 1.00× | 100% |
| 2 | 1,443 | 1.94× | 97% |
| 4 | 2,897 | 3.89× | 97% |
| 8 | 4,214 | 5.66× | 71% |
| 10 | 4,314 | 5.80× | 58% |

Monotonic, and 97% across the four performance cores. The efficiency column falls only
because the last six cores are efficiency cores; 5.80× on 4P+6E is close to the machine's
real capacity.

**Correction.** I had listed parallel efficiency as part of the remaining gap against
turbovec, on the strength of "3,845 single-thread against ~21,100 on 10 threads is 5.5×,
short of the core count." The core count was the wrong denominator — six of those cores are
worth about a third of a performance core each. Scaling is close to maxed, and the remaining
throughput gap is per-core kernel work, not threading.


## Closing on the per-core gap, and two things it is not

Hoisting the metric switch out of the batch inner loop was worth ~20%. `score()` was
switching on the metric for every (vector, query) pair even though it is fixed for the whole
index and is the *identity* under inner product, so the switch, the branch, and the
query-norm load were all overhead on the common path. It joins the correction and residency
tests in one index-wide predicate. nytimes-256 bits=5 batched: 4,096 → 4,906 compact,
4,428 → 5,257 expanded.

That leaves the index at ~134 G dim/s against the isolated kernel's ~181 at d=256 — 74% of
kernel peak reaching the index. Two hypotheses for the missing 26%, both tested, both wrong:

**Not memory bandwidth.** The isolated kernel benchmark used a 5 MB corpus while the real
index is 26 MB, so the obvious reading was that the index is bandwidth-bound where the
benchmark is not. Re-running the kernel at n=100,000 (25.6 MB at d=256) gives **180.7 G
dim/s**, against 175 at n=20,000 — no drop at all. The scan streams codes once per pass with
no reuse, and at ~0.6 loads per SDOT it simply does not demand enough bandwidth to saturate.

**Not the epilogue's loads.** Per pair the epilogue does three loads — the scan does about
ten — so loading `shift_terms[i]` and `thresholds[i]` once per tile instead of once per pair
looked like a clear win. Inverting the loop nesting to query-major measured **4% slower**
(5,257 → 5,032). The saving was real but it made the `mse_block` read strided across four
cache lines, and the premise was weak to begin with: vector-major reads all three arrays as
*contiguous* L1 streams, which cost almost nothing. Reverted.

So the remaining gap is neither bandwidth nor epilogue addressing. Ruling those out is worth
more than the 4% the second experiment cost, since both were the obvious next places to
spend effort.

**Standing on nytimes-256** (100k × 256): bits=5 compact 132 B at R@10 0.916, 4,885 batched
and ~25,300 parallel; expanded 260 B at 5,257 batched and ~28,400 parallel. turbovec is
38,052 at 270 B and 0.914 — so 1.34× behind at matched memory, from 4.2× when this started,
with equal recall and, in the compact configuration, half the memory.


## The sub-25 B gap is the codes, and three cheaper explanations are not

FAISS PQ leads below 25 B/vector on SIFT — 0.511 at 16 B against our 0.357 at 20 B. Three
candidate causes, each cheap to test, each eliminated:

| hypothesis | test | result |
|---|---|---|
| int8 query quantization or the estimator | exact f32 scan of the same codes | **0.356 vs 0.357** — identical |
| the rotation leaves structure unmixed | 4, 6, 8 RHT rounds instead of 3 | +0.6 to +1.2 points, saturates by 6 |
| too many dimensions, too few bits each | d/2 at 2 bits, d/4 at 2–4 bits, same bytes | **0.232 and 0.071** — far worse |

The first is the important one. An exact f32 scan over the same codes scores the same to
within noise, so nothing in the query path or the estimator is losing anything: **the codes
themselves are the limit.** The paper's QJL sketch correction is also confirmed worse here —
0.262 at 36 B against the scalar correction's 0.357 at 20 B.

The mixing test is worth noting because calibration gains +7 points on SIFT and ~0 on
nytimes, which reads as "the rotation is failing to mix SIFT". More rounds recovers about a
point of that, so the rotation is fine and the calibration gain comes from somewhere else.

The third result is the most useful for design. A random rotation makes coordinates
exchangeable, so keeping the first m is a Johnson–Lindenstrauss projection — it trades signal
for bits on the coordinates that survive. At equal bytes that trade loses badly in both
directions tested. For MIPS at ~1 bit per dimension, **dimension count dominates
per-coordinate precision**, and there is no budget to buy precision with.

What remains is a learned or vector quantizer. Per-coordinate scalar quantization against a
fixed density cannot represent inter-coordinate structure at all, and that is exactly what
PQ's learned sub-vector codebooks buy at this bit rate. It is a real feature, not a tuning
change.


## Per-call thread spawn: measured, not worth removing

`searchBatchParallel` spawns its workers per call, which looks like an obvious thing to
replace with a persistent pool. Measured: spawn plus join for ten threads is **113 µs**,
against a parallel call of ~9.7 ms at current throughput — **1.2%**. A pool would buy that
back in exchange for a condition-variable handshake on every call and the deadlock surface
that comes with it. Not taken.

This also retires the last item on the throughput list. The batch loop now reaches 89% of
isolated kernel throughput (86–91% by residency, measured by ablation), spawn accounts for
1.2%, and parallel scaling is 97% across the performance cores. There is no large structural
overhead left to remove — further gains would come from the scan kernel itself, which is
already within a few percent of one SDOT per cycle at 0.625 loads per SDOT.


## The turbovec throughput gap was mostly our benchmark, and my 1.15× was not apples-to-apples

`bench/py/baselines.py` times turbovec as `ix.search(query, RETRIEVE)` — **all 1000 queries in
one call, retrieving 100**. zquant's arm was timed at **32 queries per thread, retrieving 10**.
Two asymmetries, pulling in opposite directions:

| configuration | expanded, 260 B | vs turbovec 38,052 |
|---|---|---|
| 32 per thread, k=10 (as previously reported) | 32,982 | 1.15× |
| 64 per thread, k=10 | 34,000 | 1.12× |
| 100 per thread, k=10 (their batch shape) | 36,875 | 1.03× |
| **100 per thread, k=100 (their batch shape *and* depth)** | **30,141** | **1.26×** |

**The 1.15× I reported was wrong** — it compared our k=10 against their k=100 and flattered us.
Matched properly the gap is 1.26×, not 1.15×.

Two separate findings fall out:

**Batch shape was worth 12%.** Per-vector cost amortizes over the batch, and chunking at 32
queries against their 1000 gave away 32,982 → 36,875. That was a benchmark artifact, not an
implementation difference; the harness now uses the same batch shape and the same retrieval
depth as the baselines it compares against.

**Retrieval depth costs us 18%**, and that one is real: 36,875 at k=10 against 30,141 at
k=100, with nothing else changed. The arithmetic says it is not the sift work — at k=100 the
expected accepts per query are ~k·ln(n/k) ≈ 690 against ~92 at k=10, and even at log₂(100)
swaps each that is well under 1% of the scan. What it plausibly is instead is working set:
100 heaps × 100 entries × 8 B is 80 KB per thread against 8 KB at k=10, so every accept
walks a heap that no longer sits in L1. That is a hypothesis, not a measurement, and the
last three hypotheses in this file were wrong — it needs testing before it is worth acting on.


## Why we win one corpus and lose the other: fixed cost against scan rate

Being 1.7× faster on SIFT10K and 1.26× slower on nytimes-256 looks incoherent until the
numbers are normalized. Raw QPS is not comparable across corpora — nytimes asks 20× more
work per query (100k×256 against 10k×128). Normalized to the scan work actually done:

| corpus | zquant G dim/s | turbovec G dim/s |
|---|---|---|
| SIFT10K | 190.6 | 112.0 |
| nytimes-256 | 771.6 | 974.1 |

*Both* systems are far less efficient on the small corpus, because per-query fixed costs —
rotation, query preparation, top-k setup — amortize over the corpus. turbovec degrades more
(8.7× against our 4.0×), which is the whole story.

Fitting `time_per_query = fixed + work / rate` to the two corpora:

| system | fixed µs/query | peak G dim/s |
|---|---|---|
| zquant | 5.3 | 919 |
| turbovec | 10.6 | 1637 |

**turbovec's steady-state scan is ~1.8× faster; our per-query fixed cost is ~2× lower.** So
there is a crossover, predicted at ~43,500 vectors at d=256.

**This was validated rather than asserted.** Two points fit a two-parameter model by
construction, so the model was used to predict a third: at n=25,000 zquant should lead. Built
a 25,000-vector nytimes subset with its own ground truth and ran both systems. At matched
storage (132 B) and matched recall (0.908 against 0.909), zquant 56,411 QPS against turbovec
55,907 — even, where at n=100,000 turbovec leads by 1.26×. Direction confirmed.

The absolute predictions were ~25% optimistic for both systems (81k predicted against 59k
measured for us; 69k against 56k for them), so the model captures the *shape* and not the
magnitude. The fitted peak rates in particular should not be quoted: 1637 G dim/s machine-wide
would need ~4.3 SDOT per cycle per core, which is at the edge of what the hardware can issue,
so that number is probably absorbing scaling effects the two-point fit cannot separate.

Two things follow. Our fixed-cost advantage is real and matters for small-to-medium corpora
and for latency-sensitive use. And the honest statement of the remaining gap is not "we are
26% slower" but "our steady-state scan rate is behind, and it only shows above roughly 40,000
vectors at d=256."

Also corrects the SIFT headline: 3.8× was measured at k=10 against turbovec's k=100. Matched
at k=100 it is **1.70×**, and retrieval depth costs us 55% on that corpus against 18% on
nytimes — the same top-k weakness, amplified where the scan is small enough not to hide it.


## "Better SIMD" — measured, and there is nothing to switch to

turbovec's NEON scan uses `SMMLA` (ARMv8.6 I8MM) alongside an `SDOT` path, and this machine
reports `FEAT_I8MM: 1`. SMMLA does a 2×8 by 8×2 int8 matrix product into a 2×2 int32
accumulator — **32 MACs per instruction against SDOT's 16** — which looked like exactly the
missing factor.

It is not. Raw issue throughput, 16 independent accumulators, register-resident operands, no
memory traffic:

| instruction | issue rate | MACs/instr | MAC throughput |
|---|---|---|---|
| `SDOT` | 16.18 G/s | 16 | **258.8 G/s** |
| `SMMLA` | 7.98 G/s | 32 | **255.4 G/s** |

SMMLA issues at *exactly half* SDOT's rate, which cancels its doubled MACs to within 1%. On
this core the two are interchangeable and there is no faster int8 MAC available. A tiled
SMMLA kernel over a blocked layout measured **0.78×** against the current SDOT kernel, which
is consistent: same MAC ceiling, worse load structure (SMMLA's operands are 8-byte rows, so a
contiguous layout needs two loads where SDOT needs one).

**This also retires the "their scan is 1.8× faster" claim.** That came from the two-point fit,
which put turbovec at 1637 G dim/s machine-wide. Peak is now known to be ~259 G MAC/s per
core, so 1637 would need ~6.3 cores sustaining 100% of MAC issue while also issuing loads —
impossible on 4P+6E. The fit was absorbing scaling it could not separate, exactly as suspected.

**What the ceiling actually says.** Peak is ~259 G pair-dims/s per core. The isolated tiled
kernel reaches 196 (76%), and the index reaches ~156 (60%). Working backwards from the
matched full-nytimes comparison, turbovec's end-to-end rate is consistent with ~76% of peak —
about what our *kernel* already achieves in isolation.

So the gap is not instruction selection and not the kernel. It is that our index sustains 60%
of peak where our own kernel sustains 76%, and the largest identified piece of that is
collection at depth: retrieval at k=100 costs 18% on nytimes and 55% on SIFT. That is the
lever, and it is not a SIMD problem.
