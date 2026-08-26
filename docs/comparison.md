# Competitive comparison — SIFT10K

The P1 acceptance criterion: run PQ, RaBitQ, and turbovec **ourselves, on our hardware,
in our harness**, rather than quoting published numbers.

**Result after one round of work: zquant beats turbovec's 3-bit mode by 9.9 points
uncalibrated and 4.8 calibrated, at identical storage**, and wins the 40–56 B band
outright. It still trails turbovec's 4-bit mode by 3.4 points and FAISS PQ at the
smallest budgets.

The first pass lost everywhere. What closed the gap was replacing the paper's 1-bit
QJL sketch with a per-vector scalar — see [The scalar correction](#the-scalar-correction).

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
| **zquant** | **bits=4 scalar** | **52** | **0.773** | 0 | 3 | 18 |
| FAISS PQ | M=64,nbits=8 | 64 | 0.845 | 0 | 2 | 22 |
| turbovec | bits=3 | 67.9 | 0.770 | 0 | 3 | 17 |
| turbovec | bits=3 +calibrate | 68 | 0.821 | 0 | 2 | 14 |
| **zquant** | **bits=5 scalar** | **68** | **0.869** | 0 | 1 | 10 |
| turbovec | bits=4 | 67.9 | 0.880 | 0 | 1 | 7 |
| turbovec | bits=4 +calibrate | 68 | **0.904** | 0 | 1 | 9 |
| zquant | bits=5 qjl-sketch | 84 | 0.833 | 0 | 2 | 10 |

**Head to head at 68 bytes:**

| | R@10 | vs zquant |
|---|---|---|
| turbovec bits=3 | 0.770 | **−9.9 pts** (33σ) |
| turbovec bits=3 +calibrate | 0.821 | **−4.8 pts** (16σ) |
| **zquant bits=5 scalar** | **0.869** | — |
| turbovec bits=4 | 0.880 | +1.1 pts |
| turbovec bits=4 +calibrate | 0.904 | +3.4 pts |

**Best in each storage band:**

| band | winner | zquant | gap |
|---|---|---|---|
| 0–25 B | FAISS PQ 0.511 | 0.285 | −0.226 |
| 25–40 B | turbovec 0.682 | 0.582 | −0.101 |
| **40–56 B** | **zquant 0.773** | — | **wins** |
| 56–72 B | turbovec 0.904 | 0.869 | −0.034 |

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

## Open leads

- **turbovec's `calibrate()` is a per-coordinate "TQ+" calibration** fitted from a
  ~1024-row sample, worth +2.4 points at bits=4 and +5.1 at bits=3. This is the
  data-dependent step DESIGN.md speculated about, and it is the largest single lever we
  have not pulled. It would plausibly close the remaining 1.1 points to their bits=4.
- **The smallest budgets are still weak.** Below 40 B we trail both PQ and turbovec
  badly. At one or two code bits the reconstruction is too coarse for a scalar rescale
  to recover, and this is where the QJL sketch still earns its cost.
- **turbovec's bits=3 stores 68 B — the same as its bits=4** — so it pads 3-bit codes
  rather than packing them. Our bit-plane layout does not, and at 68 B we beat that
  configuration on RMS (0.01210 against 0.01370). The gap is specifically against their
  *4-bit* mode.
- Our index path costs ~22% RMS over the f32 reference estimator (int8 kernels plus f16
  scalars). Real, but far too small to account for 1.69×.

## Caveats

- One corpus, 10k vectors. SIFT10K is easier than SIFT1M and is not an embedding model's
  output. GloVe and DBpedia-OpenAI remain unrun.
- QPS is not comparable across systems here: turbovec and FAISS batch all 100 queries in
  one call, zquant loops one at a time.

## Reproducing

```sh
tools/fetch_datasets.sh
python bench/py/prepare.py         # normalize + exact IP ground truth
python bench/py/baselines.py       # PQ, RaBitQ, turbovec  -> data/baselines.csv
zig build compare_bench            # zquant                -> data/zquant.csv
python bench/py/compare.py         # merged table
zig build diagnose                 # rotation moments, D_mse, estimator RMS
```

Requires `numpy`, `faiss-cpu`, `turbovec` in a virtualenv.
