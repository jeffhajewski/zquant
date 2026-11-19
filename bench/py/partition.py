"""Partitioned (IVF) search against flat search, through the Python client, on prepared
corpora named on the command line.

    python bench/py/partition.py cohere-768-norm openai-1536-norm sift-norm

For each bit width: a calibrated flat index, then a partitioned one with about sqrt(n)
lists at a range of probe counts, and FAISS's IndexIVFRaBitQ at 1, 2 and 4 bits with the
same lists and probes, where FAISS is installed. R10@10 is the true top-10 found in the
returned top-10. QPS is one thread, 1000 queries in batches as the client sends them,
k=10. The partitioned index trains on 256 rows per list at most; FAISS trains on the
whole corpus. Needs numpy, faiss-cpu, and the zquant Python client on PYTHONPATH
(`zig build lib`, then PYTHONPATH=clients/python).
"""

import csv, math, sys, time
import numpy as np
import zquant

try:
    import faiss
except ImportError:  # the zquant rows still run
    faiss = None


def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    d = raw[0]
    return np.ascontiguousarray(raw.reshape(-1, d + 1)[:, 1:]).view(np.float32)


def read_ivecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    k = raw[0]
    return np.ascontiguousarray(raw.reshape(-1, k + 1)[:, 1:])


def recall(ids, gt):
    ids = np.asarray(ids)[:, :10]
    return float(np.mean([len(set(ids[i].tolist()) & set(gt[i, :10].tolist())) / 10
                          for i in range(len(gt))]))


def timed(search, repeats=3):
    """The fastest of `repeats` runs, since one run on a laptop is at the mercy of
    whatever else wakes up."""
    best, out = math.inf, None
    for _ in range(repeats):
        t = time.perf_counter()
        out = search()
        best = min(best, time.perf_counter() - t)
    return out, best


rows = []


def report(ds, system, config, bpv, probes, ids, dt, gt):
    r = dict(dataset=ds, system=system, config=config, bytes_per_vector=bpv, probes=probes,
             r10_at_10=round(recall(ids, gt), 4), qps_1thread=round(len(gt) / dt))
    rows.append(r)
    print(f"{system:10s} {config:18s} {bpv:6.1f} B  probes {probes!s:>5}  "
          f"R10@10 {r['r10_at_10']:.3f}  {r['qps_1thread']:7d} QPS", flush=True)


for ds in sys.argv[1:]:
    p = f"data/{ds}"
    base, query, gt = read_fvecs(f"{p}/base.fvecs"), read_fvecs(f"{p}/query.fvecs"), read_ivecs(f"{p}/groundtruth.ivecs")
    n, d = base.shape
    lists = max(1, round(math.sqrt(n)))
    sweep = sorted({max(1, lists // f) for f in (64, 32, 16, 8, 4)} | {lists})
    sample = base[np.random.default_rng(0).choice(n, min(n, 256 * lists), replace=False)]
    print(f"\n{ds}: {n} x {d}, {len(query)} queries, {lists} lists", flush=True)

    for bits in (2, 3, 5):
        ix = zquant.Index(d, bits=bits, metric="ip")
        ix.calibrate(sample[:4096])
        ix.add(base, threads=8)
        (ids, _), dt = timed(lambda: ix.search(query, k=10))
        report(ds, "zquant", f"bits={bits} flat +cal", ix.bytes_per_vector, "-", ids, dt, gt)
        ix.close()

        ix = zquant.Index(d, bits=bits, metric="ip")
        t = time.perf_counter()
        ix.partition(sample, lists, threads=8)
        ix.add(base, threads=8)
        build = time.perf_counter() - t
        print(f"           partition + add: {build:.1f} s", flush=True)
        for pr in sweep:
            (ids, _), dt = timed(lambda: ix.search(query, k=10, probes=pr))
            report(ds, "zquant", f"bits={bits} ivf", ix.bytes_per_vector, pr, ids, dt, gt)
        ix.close()

    if faiss is not None and hasattr(faiss, "IndexIVFRaBitQ"):
        faiss.omp_set_num_threads(8)
        for nb in (1, 2, 4):
            coarse = faiss.IndexFlatIP(d)
            # The last argument is bits per dimension; the one before, owning the lists.
            ix = faiss.IndexIVFRaBitQ(coarse, d, lists, faiss.METRIC_INNER_PRODUCT, True, nb)
            ix.train(base)
            ix.add(base)
            faiss.omp_set_num_threads(1)
            for pr in sweep:
                ix.nprobe = pr
                (_, ids), dt = timed(lambda: ix.search(query, 10))
                report(ds, "FAISS", f"IVFRaBitQ {nb}-bit", ix.code_size, pr, ids, dt, gt)
            faiss.omp_set_num_threads(8)

if rows:
    with open("data/partition.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote data/partition.csv", file=sys.stderr)
