# Competitive comparison

The P1 acceptance criterion: run PQ, RaBitQ, and turbovec **ourselves, on our hardware,
in our harness**, rather than quoting published numbers.

## Where it stands

- **Embeddings at d=768 and 1536** (Cohere, OpenAI). zquant finds the most true neighbours
  at every budget from 1 to 4 bits per dimension but one, by about a point over FAISS's
  RaBitQ and by 2 to 17 over PQ. On speed it trails FAISS's fast-scan RaBitQ at 1 bit per
  dimension, which is 1.5 to 1.7 times faster there, and leads at 2 and 4 bits, by 1.2 to
  1.9 times.
- **Partitioned (IVF)**, against FAISS's IVF-RaBitQ fast-scan on the same lists: from
  about 6% of the lists probed, zquant finds 0.5 to 1.5 points more at 1.4 to 2.9 times
  the speed. At the fewest probes FAISS is faster.
- **d=128 and 256** (SIFT, nytimes). PQ still has the best recall per byte at the lowest
  budgets, below about 25 B on SIFT and 100 B on nytimes. Above them zquant matches or
  passes turbovec and passes PQ and RaBitQ.
- **Not compared:** graph indexes such as HNSW, FAISS on x86, and turbovec since the
  estimator changed. Everything here ran on one Apple M5.

The rest of this page records how that position was reached, measurement by measurement.

## How it got here

The corpus difference is the most important result here. The first pass lost every band
on SIFT, and it would have been easy to conclude the implementation was simply behind.
Testing at a realistic embedding dimension — which is what the library actually targets
— showed a materially different position, and fixing the SIFT case later confirmed the
gap had been a specific defect rather than a structural deficit.

Three changes got there: a per-vector scalar in place of the paper's 1-bit sketch, a
fitted per-coordinate shift and scale anchored on the codebook's outermost centroids,
and — the one that mattered most on SIFT — fitting the per-vector correction in the same
basis the estimator applies it in.

**A fourth change replaced the second.** Calibration now fits the corpus centre and
computes its share of every score exactly, and the per-vector scalar makes the estimate
exact along the stored vector rather than fitting it by least squares. Both come from
RaBitQ's estimator. The next section has the current numbers; the sections after it record
how the comparison developed, and their zquant rows predate this change unless they say
otherwise.

## Current position

zquant rows are from this release; the other systems' rows are from the latest matched runs
(`data/baselines.csv` for SIFT), which nothing here changed. "+cal" now costs 2 B per vector more than
before, for the bias the estimator stores.

| corpus | B/vec | system | R@10 |
|---|---|---|---|
| SIFT10K, d=128 | 16 | **FAISS PQ** M=16 | **0.511** |
| | 22 | zquant bits=2 +cal | 0.458 (was 0.357 at 20 B) |
| | 24 | FAISS RaBitQ qb=5 | 0.356 |
| | 36 | turbovec bits=2 +cal | 0.682 |
| | 38 | **zquant** bits=3 +cal | **0.702** (was 0.691) |
| | 54 | **zquant** bits=4 +cal | **0.837** (was 0.832) |
| | 64 | FAISS PQ M=64 | 0.845 |
| | 70 | **zquant** bits=5 +cal | **0.907** (was 0.907) |
| nytimes-256 | 32 | **FAISS PQ** M=32 | **0.563** |
| | 38 | zquant bits=2 +cal | 0.536 (was 0.532) |
| | 64 | **FAISS PQ** M=64 | **0.752** |
| | 68 | turbovec bits=2 +cal | 0.743 |
| | 70 | zquant bits=3 +cal | 0.744 (was 0.737) |
| | 102 | **zquant** bits=4 +cal | **0.858** (was 0.857) |
| | 134 | **zquant** bits=5 +cal | **0.920** (was 0.916) |

On SIFT the gap to PQ at the bottom fell from 15 points to 5, and zquant passes RaBitQ there
by 10. nytimes barely moves, since its centroid sits near the origin (0.12) and centring has
little to remove. At d=768 and 1536 zquant now matches or passes FAISS RaBitQ at every
matched budget (see "Real embeddings at d=768 and 1536" below).

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
| zquant | bits=2 scalar +calibrate | 20 | 0.357 | 5 | 58 | 100 |
| FAISS RaBitQ | qb=5 | 24 | 0.395 | 2 | 39 | 100 |
| FAISS PQ | M=32,nbits=8 | 32 | 0.641 | 0 | 6 | 68 |
| turbovec | bits=2 | 35.9 | 0.602 | 1 | 7 | 62 |
| turbovec | bits=2 +calibrate | 36 | **0.682** | 0 | 5 | 31 |
| **zquant** | **bits=3 scalar +calibrate** | **36** | **0.691** | 0 | 5 | 55 |
| **zquant** | **bits=4 scalar +calibrate** | **52** | **0.832** | 0 | 2 | 8 |
| FAISS PQ | M=64,nbits=8 | 64 | 0.845 | 0 | 2 | 22 |
| turbovec | bits=3 | 67.9 | 0.770 | 0 | 3 | 17 |
| turbovec | bits=3 +calibrate | 68 | 0.821 | 0 | 2 | 14 |
| **zquant** | **bits=5 scalar** | **68** | **0.869** | 0 | 1 | 10 |
| **zquant** | **bits=5 scalar +calibrate** | **68** | **0.907** | 0 | 1 | 4 |
| turbovec | bits=4 | 67.9 | 0.880 | 0 | 1 | 7 |
| turbovec | bits=4 +calibrate | 68 | **0.904** | 0 | 1 | 9 |
| zquant | bits=5 qjl-sketch | 84 | 0.833 | 0 | 2 | 10 |

**Throughput at 68 bytes, SIFT10K**, 10 threads:

| | B/vec | R@10 | QPS |
|---|---|---|---|
| **zquant bits=5 +calibrate** | **68** | **0.907** | **193,386** |
| turbovec bits=4 +calibrate | 68 | 0.904 | 108,715 |

Both rows above come from a single merged run at the same corpus, retrieval depth and
core count — `compare.py` now refuses to print a table whose two halves were measured
under different conditions, so a figure here cannot be assembled from separate runs.

zquant wins this corpus on all three axes at once — better recall at identical storage, and
**1.78× the throughput**.

On nytimes-256 it wins recall and memory but still trails on throughput:

| | B/vec | R@10 | QPS (10 threads) |
|---|---|---|---|
| **zquant** bits=5 compact +cal | **132** | **0.916** | **27,013** |
| turbovec bits=4 +cal | 132 | 0.914 | 21,700–23,800 |

Ahead on both, at identical serialized storage. turbovec still holds **270 B resident**
against our 132 B, because it dequantizes to int8 in memory; that is a real memory advantage
but it is a *resident* one, not a storage one, and this document previously conflated the
two.

The split is corpus size: SIFT10K is 680 KB and cache-resident, nytimes-256 is 13 MB compact.
The remaining gap is per-core kernel work — the index reaches 134 G dim/s against the
isolated kernel's 181, and neither memory bandwidth nor epilogue addressing accounts for the
difference (both tested; see `docs/notes.md`). It is **not** parallel efficiency: scaling is
97% across the four performance cores, and 5.80× on this 4P+6E machine is close to its real
capacity.

Parallel figures drift with thermal state — three consecutive runs of one configuration gave
21,096, 18,950, and 15,525 QPS. Those quoted are first-run and should be read as upper
bounds; single-thread batched throughput is stable to about 5%.

**Head to head at 68 bytes:**

| | R@10 | vs zquant |
|---|---|---|
| turbovec bits=3 | 0.770 | **−13.7 pts** |
| turbovec bits=3 +calibrate | 0.821 | **−8.6 pts** |
| turbovec bits=4 | 0.880 | **−2.7 pts** |
| turbovec bits=4 +calibrate | 0.904 | **−0.3 pts** |
| **zquant bits=5 scalar +calibrate** | **0.907** | — |

**Best in each storage band:**

| band | winner | zquant | gap |
|---|---|---|---|
| 0–25 B | FAISS PQ 0.511 | 0.357 | **−0.154** |
| **25–40 B** | **zquant 0.691** | — | **wins** (turbovec 0.682) |
| **40–56 B** | **zquant 0.832** | — | **wins** |
| **56–72 B** | **zquant 0.907** | — | **wins** (turbovec 0.904, a tie) |

Calibration is what moved three of these bands. Fitting the per-vector correction α in the
same basis it is applied in — against `x̂ = c/scale − shift`, not against `c/scale` — is
worth +10.9 points at bits=3 and +3.8 at bits=5. The two halves of that change do not
decompose: applying α to the whole estimator while still fitting it in the shifted basis
scores 0.463, *below* the 0.582 baseline. See `docs/notes.md`.

### The sub-25 B band is conceded, and why

FAISS PQ leads below 25 B — 0.511 at 16 B against 0.357 at 20 B — and that is a deliberate
concession rather than an open item. Three cheaper explanations were tested and eliminated:

- **The estimator and the int8 query path are not losing anything.** An exact f32 scan over
  the same codes scores 0.356 against 0.357.
- **The rotation is not under-mixing.** Raising the RHT from 3 rounds to 4, 6, and 8 gains
  0.6–1.2 points and saturates.
- **The bit budget is not misallocated across dimensions.** Spending the same bytes on d/2 at
  2 bits scores 0.232, and d/4 scores 0.071. At ~1 bit per dimension, dimension count
  dominates per-coordinate precision.

So the codes themselves are the limit, and closing it needs a learned vector quantizer —
PQ's sub-vector codebooks represent inter-coordinate structure that per-coordinate scalar
quantization cannot express at any tuning. That is a second algorithm, not a fix.

It is conceded because of where the band sits. 20 B at d=128 is ~1.25 bits per dimension; at
the d=768–3072 this library targets, 25 B is ~0.2 bits per dimension — a rate almost nothing
runs in production. The useful range is 1–4 bits per dimension, and zquant wins every band
there on both corpora.

**Revisited: much of the limit was the estimator, not the codes.** The first bullet above
compared the same estimator in f32 and in int8; it never tried a different estimator. Two
changes taken from RaBitQ, now in the library, do better. One computes the corpus mean's
share of the score exactly. The other normalises the per-vector scalar so the score is
exact along the stored vector. With both, SIFT at bits=2 rises from 0.357 to 0.458, at
22 B against the old 20, still short of PQ's 0.511 at 16 B. In a numpy model, encoding
residuals against 256 k-means centroids instead of one mean, which is IVF, takes it to
0.661, though on a corpus this small the centroids add 13 B per vector. And at d=768–1536 on
real embeddings, PQ trails zquant at every budget (see the section below). See
`docs/notes.md`, "Partitioning and the low-bit gap".

## Real embeddings at d=768 and 1536

Cohere multilingual-22-12 (100k × 768) and OpenAI ada-002 on C4 (50k × 1536), both from
VectorDBBench, plus fashion-mnist (60k × 784, pixels rather than embeddings). 1000 queries,
k=100, exact inner-product ground truth, one thread on the M5, every row from one run of
`bench/py/highdim.py`. turbovec was not rerun. FAISS is 1.16.0 from pip. Its fast-scan
indexes store codes in blocks of 32 and score them with SIMD table lookups; they are its
speed-tuned versions, and the plain ones are kept for reference.

| corpus | system | B/vec | R@10 | QPS, 1 thread |
|---|---|---|---|---|
| cohere-768 | FAISS PQ M=96 | 96 | 0.582 | 510 |
| | FAISS PQ fast-scan M=192 | 96 | 0.540 | 3,514 |
| | sign-bit BQ | 96 | 0.496 | 3,552 |
| | zquant bits=2 +cal | 102 | **0.749** | 3,210 |
| | FAISS RaBitQ 1-bit | 104 | 0.738 | 476 |
| | FAISS RaBitQ fast-scan 1-bit | 104 | 0.739 | 2,502 |
| | FAISS PQ M=192 | 192 | 0.753 | 177 |
| | FAISS PQ fast-scan M=384 | 192 | 0.719 | 1,847 |
| | zquant bits=3 +cal | 198 | **0.868** | 1,803 |
| | FAISS RaBitQ 2-bit | 212 | 0.861 | 405 |
| | FAISS RaBitQ fast-scan 2-bit | 212 | 0.861 | 1,467 |
| | FAISS PQ M=384 | 384 | 0.917 | 61 |
| | FAISS PQ fast-scan M=768 | 384 | 0.895 | 906 |
| | zquant bits=5 +cal | 390 | **0.960** | 1,834 |
| | FAISS RaBitQ 4-bit | 404 | 0.952 | 380 |
| | FAISS RaBitQ fast-scan 4-bit | 404 | 0.952 | 1,160 |
| openai-1536 | FAISS PQ M=192 | 192 | 0.669 | 366 |
| | FAISS PQ fast-scan M=384 | 192 | 0.643 | 3,572 |
| | sign-bit BQ | 192 | 0.605 | 4,807 |
| | zquant bits=2 +cal | 198 | 0.792 | 3,201 |
| | FAISS RaBitQ 1-bit | 200 | 0.786 | 771 |
| | FAISS RaBitQ fast-scan 1-bit | 200 | **0.795** | 2,839 |
| | FAISS PQ M=384 | 384 | 0.828 | 123 |
| | FAISS PQ fast-scan M=768 | 384 | 0.788 | 1,900 |
| | zquant bits=3 +cal | 390 | **0.894** | 1,645 |
| | FAISS RaBitQ 2-bit | 404 | 0.890 | 545 |
| | FAISS RaBitQ fast-scan 2-bit | 404 | 0.890 | 1,344 |
| | FAISS PQ M=768 | 768 | 0.944 | 52 |
| | FAISS PQ fast-scan M=1536 | 768 | 0.860 | 949 |
| | zquant bits=5 +cal | 774 | **0.967** | 1,845 |
| | FAISS RaBitQ 4-bit | 788 | 0.962 | 504 |
| | FAISS RaBitQ fast-scan 4-bit | 788 | 0.962 | 1,135 |
| fashion-mnist-784 | FAISS PQ M=98 | 98 | 0.562 | 880 |
| | FAISS PQ fast-scan M=196 | 98 | 0.403 | 5,720 |
| | sign-bit BQ | 98 | 0.238 | 4,966 |
| | FAISS RaBitQ 1-bit | 106 | 0.361 | 741 |
| | FAISS RaBitQ fast-scan 1-bit | 106 | 0.361 | 4,109 |
| | zquant bits=2 +cal | 118 | **0.727** | 4,360 |
| | FAISS PQ M=196 | 196 | 0.737 | 385 |
| | FAISS PQ fast-scan M=392 | 196 | 0.618 | 3,066 |
| | FAISS RaBitQ 2-bit | 216 | 0.646 | 610 |
| | FAISS RaBitQ fast-scan 2-bit | 216 | 0.646 | 2,084 |
| | zquant bits=3 +cal | 230 | **0.850** | 2,462 |
| | FAISS PQ M=392 | 392 | 0.897 | 120 |
| | FAISS PQ fast-scan M=784 | 392 | 0.832 | 1,602 |
| | FAISS RaBitQ 4-bit | 412 | 0.910 | 567 |
| | FAISS RaBitQ fast-scan 4-bit | 412 | 0.910 | 1,643 |
| | zquant bits=5 +cal | 454 | **0.951** | 2,626 |

**Recall: zquant finds the most at every budget but one.** On the two embedding corpora it
is 0.4 to 1.0 points ahead of RaBitQ, plain or fast-scan, which find the same neighbours as
each other. The exception is OpenAI at one bit per dimension, where RaBitQ's fast-scan
version finds 0.795 against zquant's 0.792. PQ trails by 12 to 17 points at one bit per
dimension and by 2 to 4 at four, and its fast-scan version, with 4-bit sub-quantizers,
trails further. On fashion-mnist, which is pixels, RaBitQ manages only 0.361 at one bit,
against zquant's 0.727 at 118 B.

**Speed: zquant is faster than RaBitQ's fast-scan index at every width.** At one bit it
scores sixteen vectors at a time by table lookup, as the fast-scan indexes do
([notes.md](notes.md), "A one-bit kernel"), and is 1.1 to 1.3 times as fast as RaBitQ's
(3,210 QPS against 2,502 on Cohere). At two and four bits it is 1.2 to 1.6 times as fast.
PQ's fast-scan version is still the fastest index here at one bit, 1.1 to 1.3 times
zquant's speed, finding 15 to 32 points fewer (21 on Cohere). At two bits it is faster
than zquant too, by 2 to 25%, finding 11 to 23 points fewer. Before the one-bit kernel, RaBitQ's
fast-scan index was 1.5 to 1.7 times faster than zquant at one bit. An earlier version of
this section said zquant searched 2.2 to 4.8 times faster than RaBitQ; that was against
FAISS's plain `IndexRaBitQ`, whose scalar kernels are much slower than its fast-scan ones,
and was the wrong baseline for a speed claim.

All of this is on Apple silicon. FAISS's fast-scan kernels were written for AVX2 and run on
Arm through a NEON translation layer, so an x86 machine may favour FAISS more than these
numbers do. Neither system has been compared on x86.

Calibration is worth 9 to 22 points at 1 to 2 bits per dimension here, because both
embedding corpora sit far off the origin (centroid norms 0.849 and 0.833). RaBitQ's former
lead on recall was in how it assembled a score, not in its codes: its 1-bit codes are the
same as zquant's. Adopting its treatment of the corpus centre closed the gap
(`docs/notes.md`, "Partitioning and the low-bit gap", and the section after it).

## Partitioned search against FAISS IVF-RaBitQ

Both systems partition into the same number of k-means lists (about √n) and probe the same
number per query; one thread, k=10, 1000 queries, from one run of `bench/py/partition.py`.
FAISS here is `IndexIVFRaBitQFastScan`, its fastest IVF-RaBitQ. Each cell is R10@10, then
queries per second:

| corpus | bits/dim | probes | zquant | FAISS IVF-RaBitQ fast-scan |
|---|---|---|---|---|
| Cohere, 100k × 768, 316 lists | 1 (108 / 104 B) | 4 | **0.687 / 39,457** | 0.678 / 38,192 |
| | | 19 | **0.775 / 20,553** | 0.763 / 8,409 |
| | | 39 | **0.786 / 13,383** | 0.772 / 4,144 |
| | 2 (204 / 212 B) | 19 | **0.863 / 12,050** | 0.852 / 6,565 |
| | | 39 | **0.881 / 7,794** | 0.867 / 3,580 |
| | 4 (396 / 404 B) | 19 | **0.928 / 12,526** | 0.921 / 6,485 |
| | | 39 | **0.953 / 7,818** | 0.943 / 3,416 |
| OpenAI, 50k × 1536, 224 lists | 1 (204 / 200 B) | 3 | 0.664 / 28,705 | **0.670 / 28,805** |
| | | 28 | **0.809 / 13,418** | 0.802 / 3,286 |
| | 2 (396 / 404 B) | 28 | **0.890 / 7,863** | 0.881 / 2,816 |
| | 4 (780 / 788 B) | 28 | **0.948 / 7,144** | 0.935 / 2,716 |

From about 6% of the lists up, zquant finds 0.7 to 1.4 points more and is 1.8 to 3.6 times
as fast as the faster of FAISS's two IVF-RaBitQ indexes, the gap widening with more probes.
At the fewest probes the two are level: on Cohere zquant is 3% faster and finds 0.9 points
more, on OpenAI FAISS is 2% faster and finds 0.2 to 0.6 more. With 3 or 4 lists scanned,
ranking the centroids is a large share of the work, and FAISS does that part faster. FAISS
trains its lists on the whole corpus, zquant on at most 256 rows per list. The full sweep is
in `data/partition.csv` after a run, with FAISS's plain `IndexIVFRaBitQ`, which finds the
same and is slower on Cohere, but on OpenAI is faster than the fast-scan one at most
settings (3,687 QPS at one bit and 28 probes, 3,534 at two); the ratios above use it there.
At one bit, zquant's numbers are from the table-lookup kernel; before it, FAISS was faster
at the fewest probes on both corpora and zquant 1.4 to 2.9 times as fast from 6% up.

The partitioned index's recall also beats the flat calibrated index's on every corpus here,
since each vector is stored against a nearby centroid rather than the corpus centre
([notes.md](notes.md), "Partitioning, measured through the library").

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

**Superseded.** Calibration now fits a centre instead (see "Current position"). This
section describes the per-coordinate fit it replaced, which the centred estimator matched
or beat on every corpus measured.

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

**Since reversed.** The library now centres, and it gains 9 to 22 points at bits 2 and 3 on
every corpus whose centroid norm is 0.65 or more. It differs from the attempts below in two
ways: the mean's share of each score is computed exactly instead of passing through the
quantizer, and the per-vector scalar is the along-vector-exact one rather than a
least-squares fit. Which of the two the attempts below were missing was not isolated. In the
numpy model, centring with the exact mean term but the least-squares scalar scored 0.569 on
Cohere at bits=2 against 0.739 with both, so the scalar is at least a large part of it
(`docs/notes.md`, "Partitioning and the low-bit gap").

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

- **The low-bit regime is the remaining weakness for a flat index, at d=128 and 256.** On
  SIFT, PQ leads by 5 points at the bottom (0.511 at 16 B against 0.458 at 22 B), down from
  15 before the estimator change. This was thought structural, since PQ uses *vector*
  codebooks. Mostly it was not. Partitioned, zquant reaches 0.651 at 26 B, passing PQ's
  16 B and tying its 32 B, though on a corpus this small the centroids add 5 B per vector.
  See `docs/notes.md`, "Partitioning, measured through the library".
- **turbovec's calibration was more effective than ours at low bit-widths** — +8.1 points
  at bits=2 against our old −1.5. Ours now gains +12.0 there (0.338 uncalibrated, 0.458
  calibrated), by handling the mean, which was the suspect.
- **turbovec's bits=3 stores 68 B — the same as its bits=4** — so it pads 3-bit codes
  rather than packing them. Our bit-plane layout does not, and at 68 B we beat that
  configuration on RMS (0.01210 against 0.01370). The gap is specifically against their
  *4-bit* mode.
- Our index path costs ~22% RMS over the f32 reference estimator (int8 kernels plus f16
  scalars). Real, but far too small to account for 1.69×.

## Caveats

- The main tables cover two corpora. SIFT10K is 10k vectors of image descriptors — easier
  than SIFT1M and not an embedding model's output. nytimes-256 is real text embeddings but
  sampled to 100k. Real embeddings at d=768 and 1536 have their own section above, against
  PQ and RaBitQ only, on one thread; turbovec was not rerun there.
- Results differ substantially between the two, so neither generalizes on its own.
- **Correction (supersedes the paragraph below).** The figure of 38,052 QPS for turbovec on
  nytimes is not reproducible: three matched runs give 23,810, 22,565 and 21,715. It was
  recorded before `compare.py` enforced provenance, and its pairing cannot be reconstructed.
  Every comparison derived from it — "1.34× behind", "1.22× behind", and the fixed-cost /
  scan-rate crossover model — is withdrawn. And turbovec's *serialized* size is 132 B, the
  same as ours; only its *resident* footprint is 270 B.

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

Everything below needs real corpora and their dependencies. For a check that needs neither,
`zig build quickbench` generates its own data and runs in about a minute.

Every number in this document is from a ReleaseFast build. Benchmarks default to it and
`-Doptimize` does not change them — `-Dbench-opt` is the knob. Debug is roughly 100x
slower, which is far too large a gap for a Debug measurement to be mistaken for a real
one.

**Environment for every measurement in this document:** Apple M5 (4 performance + 6
efficiency cores), macOS 26.5, Zig 0.16.0, aarch64, `-Doptimize=ReleaseFast`. Some results
are specific to it — `SMMLA` and `SDOT` have identical int8 MAC throughput on this core, so
the conclusion that there is no faster instruction available does not transfer to hardware
where that is untrue. Parallel throughput drifts with thermal state by up to 25% across
consecutive runs; single-thread figures are stable to roughly 5%, and are the better basis
for comparison.

```sh
tools/fetch_datasets.sh
python bench/py/prepare.py                 # SIFT10K
python bench/py/prepare.py nytimes-256     # or an ann-benchmarks HDF5
python bench/py/baselines.py       # PQ, RaBitQ, turbovec  -> data/baselines.csv
zig build compare_bench            # zquant                -> data/zquant.csv
python bench/py/compare.py         # merged table; refuses to merge mismatched runs
tools/fetch_datasets.sh highdim    # cohere-768, openai-1536 (VectorDBBench parquet)
python bench/py/prepare.py cohere-768    # then openai-1536
python bench/py/highdim.py cohere-768-norm openai-1536-norm   # zquant vs PQ, RaBitQ, sign BQ
python bench/py/ivf_lab.py cohere-768-norm --curve 1024 --faiss  # estimator and IVF variants
zig build diagnose                 # rotation moments, D_mse, estimator RMS
python bench/py/synth.py spectrum --alpha 1.0   # synthetic corpus with one property varied
```

Requires `numpy`, `faiss-cpu`, `turbovec` in a virtualenv, plus `pyarrow` for the
parquet corpora and the zquant Python client (`PYTHONPATH=clients/python`) for the last two.
