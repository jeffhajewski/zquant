# Competitive comparison — SIFT10K

The P1 acceptance criterion: run PQ, RaBitQ, and turbovec **ourselves, on our hardware,
in our harness**, rather than quoting published numbers.

**Result: zquant loses at every matched storage budget.** The gap is real, reproducible,
and localized to the estimator. This document records it and what has been ruled out.

## Setup

| | |
|---|---|
| Corpus | ANN_SIFT10K, 10,000 × 128d, unit-normalized |
| Queries | 100 |
| Ground truth | exact inner product, top-100, computed on the normalized data |
| Metric | inner product (`k=10`, retrieve 100 so ranks are uncensored) |
| Systems | zquant, turbovec 1.0.0, FAISS 1.15.0 (`IndexPQ`, `IndexRaBitQ`) |

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

| system | config | B/vec | R@10 | med | p90 | worst | QPS |
|---|---|---|---|---|---|---|---|
| FAISS PQ | M=16,nbits=8 | 16 | 0.555 | 1 | 15 | 55 | 21128 |
| zquant | bits=2 no-sketch | 20 | 0.458 | 1 | 27 | 100 | 16059 |
| FAISS RaBitQ | qb=5 | 24 | 0.432 | 1 | 26 | 100 | 59409 |
| FAISS PQ | M=32,nbits=8 | 32 | **0.675** | 0 | 4 | 15 | 63198 |
| turbovec | bits=2 | 35.9 | 0.656 | 0 | 4 | 38 | 61915 |
| turbovec | bits=2 +calibrate | 36 | **0.724** | 0 | 2 | 21 | 82404 |
| zquant | bits=3 no-sketch | 36 | 0.459 | 2 | 34 | 100 | 16717 |
| zquant | bits=2 | 36 | 0.419 | 2 | 33 | 95 | 9799 |
| zquant | bits=4 no-sketch | 52 | 0.588 | 0 | 9 | 83 | 9270 |
| zquant | bits=3 | 52 | 0.570 | 1 | 14 | 80 | 10165 |
| FAISS PQ | M=64,nbits=8 | 64 | 0.860 | 0 | 1 | 8 | 33646 |
| turbovec | bits=4 | 67.9 | 0.896 | 0 | 1 | 3 | 73017 |
| turbovec | bits=4 +calibrate | 68 | **0.907** | 0 | 1 | 4 | 78966 |
| turbovec | bits=3 | 67.9 | 0.814 | 0 | 2 | 8 | 80488 |
| zquant | bits=5 no-sketch | 68 | 0.754 | 0 | 4 | 17 | 17021 |
| zquant | bits=4 | 68 | 0.742 | 0 | 3 | 9 | 6841 |
| zquant | bits=5 | 84 | 0.837 | 0 | 2 | 8 | 10385 |

**Deficit against the best in each band:**

| band | winner | zquant best | gap |
|---|---|---|---|
| 0–25 B | FAISS PQ 0.555 | 0.458 | **−0.097** |
| 25–40 B | turbovec 0.724 | 0.459 | **−0.265** |
| 56–72 B | turbovec 0.907 | 0.754 | **−0.153** |

## Where the gap is

Estimator RMS error against exact inner products, measured under one protocol —
a 400-vector index, every vector scored, unconditioned:

| bytes/vec | zquant | turbovec | ratio |
|---|---|---|---|
| 36 | 0.03980 | **0.02677** | 1.49× |
| 68 | 0.01210 | **0.00718** | 1.69× |
| 68 | 0.01210 | 0.01370 *(their bits=3)* | 0.88× — we win |
| 84 | 0.00640 | — | — |

turbovec is roughly **0.7–0.8 bits more efficient**. Since it implements the same
algorithm, this is an implementation gap, not an algorithmic tradeoff — which is exactly
why the design named it as a sharper correctness signal than FAISS.

## What has been ruled out

The quantizer itself is provably fine. On real rotated SIFT data:

- `E[y²]` measured `7.8125e-3` against the theoretical `1/d` — ratio **1.000**
- `E[y⁴]` ratio **0.969**; kurtosis 2.863 against a Gaussian 3.0
- `D_mse` matches the paper at every bit-width: 0.356/0.113/0.033/0.009 against
  0.36/0.117/0.03/0.009

So the rotation mixes correctly and the codebook achieves its theoretical distortion.
Two hypotheses were tested and **refuted**:

1. *"The sketch is wasted for ranking."* The bias it removes is multiplicative, and a
   constant factor cannot change an ordering. Measured: the sketch **helps** at b≥3
   (0.742 against 0.588 at 68 B). Only at b=2 is it a bad trade.
2. *"Per-vector shrinkage of ỹ biases scores."* Rescaling so `‖x̃‖ = ‖x‖` costs nothing
   — the factor folds into the stored norm. Measured: **worse**, 0.742 → 0.595. It
   breaks the calibration between the MSE term and the γ-scaled sketch correction.

## Open leads

- **turbovec's `calibrate()` is a per-coordinate "TQ+" calibration** fitted from a
  ~1024-row sample, worth +2.5 to +7 points of R@10. This is the data-dependent step
  DESIGN.md speculated about. But it does not explain the gap: *uncalibrated* turbovec
  bits=4 already scores 0.896 against our 0.754.
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
