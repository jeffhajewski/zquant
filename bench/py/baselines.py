"""Run PQ, RaBitQ, and turbovec on the prepared data and emit results.csv.

All three read data/sift-norm, which zquant's own bench also reads, so the inputs and
ground truth are shared rather than reproduced.

A fairness note that goes in the results, not just the code: PQ and RaBitQ *train* on
the corpus they then index, and here they train on the whole of it. TurboQuant is
data-oblivious and gets no such pass. That favours the baselines, which is the right
direction for a comparison we want to be able to trust.
"""
import csv, sys, time
import numpy as np
import faiss
import turbovec

DATA = "data/" + open("data/dataset.txt").read().strip()
# Stamped into every row so `compare.py` can refuse to merge results measured on
# different corpora. Comparing numbers taken under different conditions is how a
# 1.15x figure got reported that was really 1.26x; see docs/notes.md.
N_BASE = 0
D_BASE = 0
K = 10
# Retrieve deeper than K so the true-NN rank distribution is not censored at K:
# with RETRIEVE == K, "worst rank = 10" only ever means "absent", which hides how
# far outside the top-10 a miss actually landed.
RETRIEVE = 100


def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return np.ascontiguousarray(raw.reshape(-1, dim + 1)[:, 1:]).view(np.float32)


def read_ivecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    k = raw[0]
    return np.ascontiguousarray(raw.reshape(-1, k + 1)[:, 1:])


def score(name, config, per_vector, ids, elapsed, gt, nq):
    """R@10 plus the rank distribution of the true nearest neighbour."""
    recall = np.mean([len(set(ids[i][:K]) & set(gt[i, :K])) / K for i in range(nq)])
    ranks = []
    for i in range(nq):
        hit = np.where(ids[i] == gt[i, 0])[0]
        ranks.append(int(hit[0]) if len(hit) else len(ids[i]))
    ranks = np.array(ranks)
    return dict(
        dataset=DATA, n=N_BASE, d=D_BASE, nq=nq, k=RETRIEVE, threads=0,
        system=name, config=config, bytes_per_vector=round(per_vector, 1),
        recall_at_10=round(float(recall), 4),
        rank_median=int(np.median(ranks)), rank_p90=int(np.percentile(ranks, 90)),
        rank_worst=int(ranks.max()), qps=round(nq / elapsed, 1),
    )


def main():
    global N_BASE, D_BASE
    base, query, gt = (read_fvecs(f"{DATA}/base.fvecs"),
                       read_fvecs(f"{DATA}/query.fvecs"),
                       read_ivecs(f"{DATA}/groundtruth.ivecs"))
    n, d = base.shape
    nq = query.shape[0]
    N_BASE, D_BASE = n, d
    print(f"corpus {n}x{d}, {nq} queries, k={K}, metric=inner product", file=sys.stderr)

    rows = []

    # --- FAISS PQ. M must divide d; nbits is bits per subquantizer code.
    for m, nbits in [(16, 8), (32, 8), (64, 8), (32, 4), (64, 4), (128, 4)]:
        if d % m:
            continue
        ix = faiss.IndexPQ(d, m, nbits, faiss.METRIC_INNER_PRODUCT)
        ix.train(base)
        ix.add(base)
        t = time.perf_counter(); _, ids = ix.search(query, RETRIEVE); dt = time.perf_counter() - t
        rows.append(score("FAISS PQ", f"M={m},nbits={nbits}", ix.sa_code_size(), ids, dt, gt, nq))

    # --- FAISS RaBitQ.
    try:
        for qb in [0, 1, 2, 3, 4, 5]:
            ix = faiss.IndexRaBitQ(d, faiss.METRIC_INNER_PRODUCT)
            if qb:
                ix.qb = qb
            ix.train(base)
            ix.add(base)
            t = time.perf_counter(); _, ids = ix.search(query, RETRIEVE); dt = time.perf_counter() - t
            rows.append(score("FAISS RaBitQ", f"qb={qb}", ix.sa_code_size(), ids, dt, gt, nq))
    except Exception as e:
        print(f"RaBitQ skipped: {type(e).__name__}: {e}", file=sys.stderr)

    # --- turbovec, with and without its TQ+ calibration.
    rng = np.random.default_rng(0)
    sample = base[rng.choice(n, size=min(1024, n), replace=False)]
    for bw in [2, 3, 4]:
        for calibrated in [False, True]:
            ix = turbovec.TurboQuantIndex(d, bw)
            if calibrated:
                ix.calibrate(sample)
            ix.add(base)
            ix.prepare()
            empty = len(turbovec.TurboQuantIndex(d, bw).to_bytes())
            per_vector = (len(ix.to_bytes()) - empty) / n
            t = time.perf_counter(); _, ids = ix.search(query, RETRIEVE); dt = time.perf_counter() - t
            label = f"bits={bw}" + (" +calibrate" if calibrated else "")
            rows.append(score("turbovec", label, per_vector, np.asarray(ids), dt, gt, nq))

    with open("data/baselines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"\nwrote data/baselines.csv ({len(rows)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
