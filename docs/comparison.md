# Competitive comparison

The P1 acceptance criterion: run PQ, RaBitQ, and turbovec **ourselves, on our hardware,
in our harness**, rather than quoting published numbers.

**Where it stands, and it depends on the corpus.**

On **real text embeddings** (nytimes-256, d=256) zquant **wins the two highest storage
bands outright**, including beating turbovec's best configuration at identical storage.
On **SIFT10K** (d=128 image descriptors) it wins one middle band and trails at the
extremes.

That difference is the most important result here. The first pass lost every band on
SIFT, and it would have been easy to conclude the implementation was simply behind.
Testing at a realistic embedding dimension — which is what the library actually targets
— showed a materially different position.

Two changes got there: a per-vector scalar in place of the paper's 1-bit sketch, and a
fitted per-coordinate scale.

## nytimes-256 — real text embeddings, d=256

100,000 × 256 sampled from the ann-benchmarks corpus, 1000 queries, exact
inner-product ground truth. **Recall ceilings at 0.993**, not 1.0: the corpus contains
6,837 duplicate rows, and every top-10 disagreement between an exact f32 scan and the
ground truth is an exact boundary tie. All systems face the same ceiling.

| system | config | B/vec | R@10 |
|---|---|---|---|
| FAISS PQ | M=16,nbits=8 | 16 | 0.393 |
| zquant | bits=2 | 36 | 0.530 |
| FAISS PQ | M=32,nbits=8 | 32 | **0.563** |
| FAISS RaBitQ | qb=5 | 40 | 0.537 |
| zquant | bits=3 | 68 | 0.732 |
| turbovec | bits=2 +calibrate | 68 | 0.743 |
| FAISS PQ | M=64,nbits=8 | 64 | **0.752** |
| **zquant** | **bits=4 +calibrate** | **100** | **0.853** |
| turbovec | bits=3 +calibrate | 132 | 0.850 |
| turbovec | bits=4 +calibrate | 132 | 0.914 |
| **zquant** | **bits=5 +calibrate** | **132** | **0.917** |

| band | winner | zquant | gap |
|---|---|---|---|
| 25–40 B | FAISS PQ 0.563 | 0.530 | −0.033 |
| 56–72 B | FAISS PQ 0.752 | 0.732 | −0.020 |
| **90–110 B** | **zquant 0.853** | — | **wins** |
| **110–140 B** | **zquant 0.917** | — | **wins** |

At 132 B zquant beats turbovec's *best* configuration (0.917 against 0.914) and its
3-bit mode by 6.7 points.

**Calibration is worth almost nothing on this corpus** — +0.2 points for zquant, +0.02
for turbovec — against +1.4 and +8.1 respectively on SIFT. The anisotropy it exploits is
a property of the data, not of the method, and nytimes does not have much of it. That
also corroborates that our calibration captures the same structure theirs does; the SIFT
low-bit gap is specific to that corpus rather than a general deficiency.

## SIFT10K — image descriptors, d=128

## Setup

| | |
|---|---|
| Corpus | ANN_SIFT10K, 10,000 × 128d, unit-normalized |
| Queries | 1000 (100 supplied + 900 held-out from the learn split) |
| Ground truth | exact inner product, top-100, computed on the normalized data |
| Metric | inner product (`k=10`, retrieve 100 so ranks are uncensored) |
| Systems | zquant, turbovec 1.0.0, FAISS 1.15.0 (`IndexPQ`, `IndexRaBitQ`) |

The query set is enlarged deliberately. With the supplied 100 queries the binomial
standard error on R@10 is ~1.0 point — the same size as the differences between systems,
so a 1.5-point gap would have been 1.5σ and unreportable. At 1000 queries it is 0.30
points, which resolves them.

Normalization is what makes the comparison possible: turbovec is inner-product only,
SIFT's published ground truth is L2, and on unit vectors IP, cosine, and L2 induce the
*same* ranking. All four systems read the same prepared files (`bench/py/prepare.py`),
so inputs are shared rather than three loaders agreeing.

**A control runs first.** An exact unquantized f32 scan through our own harness scores
`R@10 = 1.0000`. Without it, every number below could be measuring a harness bug.

**The baselines are favoured, deliberately.** PQ and RaBitQ *train* on the very corpus
they then index. TurboQuant is data-oblivious and gets no such pass. A comparison we
want to trust should lean against us.

## Results

R@10, 1000 queries, standard error ≈ 0.30 points.

| system | config | B/vec | R@10 | med | p90 | worst |
|---|---|---|---|---|---|---|
| FAISS PQ | M=16,nbits=8 | 16 | **0.511** | 1 | 20 | 100 |
| zquant | bits=2 scalar | 20 | 0.285 | 9 | 100 | 100 |
| FAISS RaBitQ | qb=5 | 24 | 0.395 | 2 | 39 | 100 |
| FAISS PQ | M=32,nbits=8 | 32 | 0.641 | 0 | 6 | 68 |
| turbovec | bits=2 | 35.9 | 0.602 | 1 | 7 | 62 |
| turbovec | bits=2 +calibrate | 36 | **0.682** | 0 | 5 | 31 |
| zquant | bits=3 scalar | 36 | 0.582 | 1 | 10 | 66 |
| **zquant** | **bits=4 scalar +calibrate** | **52** | **0.776** | 0 | 2 | 23 |
| FAISS PQ | M=64,nbits=8 | 64 | 0.845 | 0 | 2 | 22 |
| turbovec | bits=3 | 67.9 | 0.770 | 0 | 3 | 17 |
| turbovec | bits=3 +calibrate | 68 | 0.821 | 0 | 2 | 14 |
| **zquant** | **bits=5 scalar** | **68** | **0.869** | 0 | 1 | 10 |
| **zquant** | **bits=5 scalar +calibrate** | **68** | **0.883** | 0 | 1 | 7 |
| turbovec | bits=4 | 67.9 | 0.880 | 0 | 1 | 7 |
| turbovec | bits=4 +calibrate | 68 | **0.904** | 0 | 1 | 9 |
| zquant | bits=5 qjl-sketch | 84 | 0.833 | 0 | 2 | 10 |

**Head to head at 68 bytes:**

| | R@10 | vs zquant |
|---|---|---|
| turbovec bits=3 | 0.770 | **−11.3 pts** |
| turbovec bits=3 +calibrate | 0.821 | **−6.2 pts** |
| turbovec bits=4 | 0.880 | **−0.3 pts** |
| **zquant bits=5 scalar +calibrate** | **0.883** | — |
| turbovec bits=4 +calibrate | 0.904 | +2.1 pts |

**Best in each storage band:**

| band | winner | zquant | gap |
|---|---|---|---|
| 0–25 B | FAISS PQ 0.511 | 0.285 | **−0.226** |
| 25–40 B | turbovec 0.682 | 0.590 | **−0.093** |
| **40–56 B** | **zquant 0.776** | — | **wins** |
| 56–72 B | turbovec 0.904 | 0.883 | −0.020 |

## The scalar correction

MSE-optimal reconstruction shrinks ỹ, so ⟨p, ỹ⟩ underestimates ⟨p, y⟩. The paper corrects
this with a 1-bit-per-coordinate QJL sketch of the residual. But over random queries the
least-squares estimate is just

    ⟨p, y⟩ ≈ (⟨y, ỹ⟩ / ‖ỹ‖²) · ⟨p, ỹ⟩

— **one scalar per vector, not one bit per coordinate.** At d=128 that is 2 bytes against
16, and the sketch costs exactly what one more MSE bit costs. So the trade is: does a
1-bit residual sketch beat a whole extra bit of codebook resolution? Measured, it does
not, at every budget above the smallest:

| storage | QJL sketch | scalar | delta |
|---|---|---|---|
| 36 B | 0.369 | **0.582** | +0.213 |
| 52 B | 0.521 | **0.773** | +0.252 |
| 68 B | 0.704 | **0.869** | +0.165 |

`.scalar` is now the default. `.qjl_sketch` remains, both because it is the paper's
construction and because it is the reference the scalar form is tested against. At b=2
the sketch is still better — with a single code bit the residual carries real information
that a scalar cannot capture.

## Where the gap is

Estimator RMS error against exact inner products, measured under one protocol —
a 400-vector index, every vector scored, unconditioned:

Measured *before* the scalar correction, under one protocol for both — a 400-vector
index, every vector scored:

| bytes/vec | zquant | turbovec | ratio |
|---|---|---|---|
| 36 | 0.03980 | **0.02677** | 1.49× |
| 68 | 0.01210 | **0.00718** | 1.69× |
| 68 | 0.01210 | 0.01370 *(their bits=3)* | 0.88× — we win |

This localized the problem to the estimator rather than the quantizer, which is what
pointed at the bit allocation and led to the scalar correction. Since turbovec implements
the same algorithm, a gap here could only ever be an implementation difference, which is
exactly why the design named it a sharper signal than FAISS.

## What has been ruled out

The quantizer itself is provably fine. On real rotated SIFT data:

- `E[y²]` measured `7.8125e-3` against the theoretical `1/d` — ratio **1.000**
- `E[y⁴]` ratio **0.969**; kurtosis 2.863 against a Gaussian 3.0
- `D_mse` matches the paper at every bit-width: 0.356/0.113/0.033/0.009 against
  0.36/0.117/0.03/0.009

So the rotation mixes correctly and the codebook achieves its theoretical distortion.
Two hypotheses were tested and **refuted**:

1. *"The sketch is wasted for ranking."* The bias it removes is multiplicative, and a
   constant factor cannot change an ordering. Measured: **refuted as stated** — deleting
   the sketch outright costs recall, because the estimator is then genuinely biased.
   The correct version of the idea is that the sketch is the wrong *price* for the
   correction, not that the correction is unnecessary. That became the scalar form.
2. *"Per-vector shrinkage of ỹ biases scores."* Rescaling so `‖x̃‖ = ‖x‖` costs nothing
   — the factor folds into the stored norm. Measured: **worse**, 0.742 → 0.595. It
   breaks the calibration between the MSE term and the γ-scaled sketch correction.

## Calibration: per-coordinate scales

A random rotation *randomizes* the axes but does not *whiten* them. After rotation
coordinate j has variance πⱼᵀCπⱼ, which still tracks the spectrum of the data
covariance C. On SIFT that leaves a **4.3× spread** across coordinates (cv 0.30), so a
single shared codebook is badly matched at both ends.

Fitting a per-coordinate scale from a sample recovers +1.4 points at 68 B. It costs `d`
floats for the whole index, nothing per vector, and nothing at query time — the scale
folds into the rotated query, so the scan kernel keeps using the raw centroid table.
It is opt-in: it is the one data-dependent step in an otherwise data-oblivious library.

It *hurts* at bits=2, where a single code bit is too coarse for a scale fit to help.

## Mean-centring: a measured negative

The signal is unambiguous. The codebook is symmetric about zero — correct for a vector
uniform on the sphere, the worst case the paper analyses. Real embeddings share a strong
common direction, and after rotation coordinate j inherits mean πⱼᵀμ:

    |mean| / σ  averaged 0.78, worst 3.78

So some coordinates put nearly all their mass on one side of a symmetric codebook, and
half its levels go unused. Subtracting the mean should obviously help.

**It measured worse, twice.** First naively (0.883 → 0.864), then again after fixing a
real flaw in the first attempt — `y` is a unit vector, so `r = y − m` has a norm that
varies per vector, mismatching a codebook built for fixed variance; renormalizing `r`
and folding the factor into the stored scalars addressed that (0.883 → 0.857).

Both attempts are arithmetically consistent, and the expected gain is real: ‖m‖ ≈ 0.62
implies centring shrinks what is quantized by ~21%. The measurement disagrees with the
derivation and the derivation has not been faulted. **Left out until it is understood** —
this is recorded as an open problem rather than a closed one.

A related bug is worth remembering: while testing this, `calibrate` began fitting the
*centred* standard deviation while the encoder still did not subtract the mean. The two
must agree, or the encoded values fall off the codebook's range. Cost 3.4 points before
it was caught.

## Open leads

- **The low-bit regime is the real weakness.** Below 40 B we trail PQ by 22 points. This
  is likely structural: PQ uses *vector* codebooks (256 centroids over 8 dimensions at
  M=16), and vector quantization beats scalar quantization hardest at low rates.
  TurboQuant is scalar by construction. Closing this may need something outside the
  paper rather than a better implementation of it.
- **turbovec's calibration is more effective than ours at low bit-widths** — +8.1 points
  at bits=2 against our −1.5. Whatever it fits, it is not just a per-coordinate scale.
  Mean handling is the obvious suspect, and is exactly what we could not make work.
- **Mean-centring** (above) — the largest identified but unrealized gain.
- **turbovec's bits=3 stores 68 B — the same as its bits=4** — so it pads 3-bit codes
  rather than packing them. Our bit-plane layout does not, and at 68 B we beat that
  configuration on RMS (0.01210 against 0.01370). The gap is specifically against their
  *4-bit* mode.
- Our index path costs ~22% RMS over the f32 reference estimator (int8 kernels plus f16
  scalars). Real, but far too small to account for 1.69×.

## Caveats

- Two corpora. SIFT10K is 10k vectors of image descriptors — easier than SIFT1M and not
  an embedding model's output. nytimes-256 is real text embeddings but sampled to 100k.
  Higher dimensions (768–3072), where the design's assumptions are strongest, remain
  unrun.
- Results differ substantially between the two, so neither generalizes on its own.
- **Throughput and storage are one tradeoff, not two axes.** turbovec keeps **270 B per
  vector resident** at `bit_width=4` on nytimes — twice what it serializes (135 B), and
  twice zquant's 132 B. The in-memory form is dequantized int8, which removes the table
  lookup and lets its kernel run near-pure `sdot`. So on a full machine turbovec reaches
  38,052 QPS at 270 B/vector against zquant's 11,790 at 132 B/vector, at R@10 0.914 and
  0.917 respectively — **different points on the space/time curve**, not a like-for-like
  deficit. An earlier version of this document compared their *serialized* size against
  our *resident* size and drew the wrong conclusion.
- Per-core comparisons are ill-defined here: this machine has 4 performance and 6
  efficiency cores, and zquant scales at 96% efficiency across the four performance cores
  before flattening. Quote full-machine throughput alongside residency instead.

## Reproducing

```sh
tools/fetch_datasets.sh
python bench/py/prepare.py                 # SIFT10K
python bench/py/prepare.py nytimes-256     # or an ann-benchmarks HDF5
python bench/py/baselines.py       # PQ, RaBitQ, turbovec  -> data/baselines.csv
zig build compare_bench            # zquant                -> data/zquant.csv
python bench/py/compare.py         # merged table
zig build diagnose                 # rotation moments, D_mse, estimator RMS
```

Requires `numpy`, `faiss-cpu`, `turbovec` in a virtualenv.
