"""zquant against FAISS PQ, FAISS RaBitQ (plain and fast-scan) and sign-bit binary
quantization, at matched bits per dimension, on prepared corpora named on the command line.

    python bench/py/highdim.py cohere-768-norm openai-1536-norm fashion-mnist-784-euclidean-norm

Written to test one claim: that PQ's lead at ~1 bit per dimension, measured at d=128 and
d=256, would carry over to the d=768-1536 embeddings people actually quantize that hard.
It did not (docs/notes.md, "Partitioning and the low-bit gap").

R10@10 is the true top-10 found in the returned top-10. R10@100 is the true top-10 found
in the returned top-100, which is what a rerank stage would have to work with. QPS is one
thread with all 1000 queries in one batch, for every system. PQ and RaBitQ train on the
whole corpus they then index; zquant's calibration sees 4,096 rows. Needs numpy, faiss-cpu,
and the zquant Python client on PYTHONPATH (`zig build lib`, then PYTHONPATH=clients/python).
"""
import csv, os, sys, time
import numpy as np
import faiss
import zquant


def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    d = raw[0]
    return np.ascontiguousarray(raw.reshape(-1, d + 1)[:, 1:]).view(np.float32)


def read_ivecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    k = raw[0]
    return np.ascontiguousarray(raw.reshape(-1, k + 1)[:, 1:])


def recall(ids, gt, depth):
    ids = np.asarray(ids)[:, :depth]
    return float(np.mean([len(set(ids[i].tolist()) & set(gt[i, :10].tolist())) / 10
                          for i in range(len(gt))]))


def one_thread(search):
    faiss.omp_set_num_threads(1)
    try:
        t = time.perf_counter()
        out = search()
        return out, time.perf_counter() - t
    finally:
        faiss.omp_set_num_threads(NCPU)


NCPU = faiss.omp_get_max_threads()
rows = []


def report(ds, d, system, config, bpv, ids, dt, gt):
    r = dict(dataset=ds, d=d, system=system, config=config, bytes_per_vector=round(bpv, 1),
             bits_per_dim=round(bpv * 8 / d, 2), r10_at_10=round(recall(ids, gt, 10), 4),
             r10_at_100=round(recall(ids, gt, 100), 4), qps_1thread=round(len(gt) / dt))
    rows.append(r)
    print(f"{system:10s} {config:16s} {r['bytes_per_vector']:7.1f} B {r['bits_per_dim']:5.2f} b/d  "
          f"R10@10 {r['r10_at_10']:.3f}  R10@100 {r['r10_at_100']:.3f}  {r['qps_1thread']:6d} QPS",
          flush=True)


for ds in sys.argv[1:]:
    p = f"data/{ds}"
    base, query, gt = read_fvecs(f"{p}/base.fvecs"), read_fvecs(f"{p}/query.fvecs"), read_ivecs(f"{p}/groundtruth.ivecs")
    n, d = base.shape
    print(f"\n{ds}: {n} x {d}, {len(query)} queries", flush=True)

    for bits in (2, 3, 4, 5):
        for cal in (False, True):
            ix = zquant.Index(d, bits=bits, metric="ip")
            if cal:
                ix.calibrate(base[np.random.default_rng(0).choice(n, min(4096, n), replace=False)])
            ix.add(base, threads=NCPU)
            (ids, _), dt = one_thread(lambda: ix.search(query, k=100, threads=1))
            report(ds, d, "zquant", f"bits={bits}" + (" +cal" if cal else ""), ix.bytes_per_vector, ids, dt, gt)
            ix.close()

    # Binary quantization as it is usually meant: one sign bit per dimension, Hamming distance.
    bq = faiss.IndexBinaryFlat(d)
    bq.add(np.packbits(base > 0, axis=1))
    (_, ids), dt = one_thread(lambda: bq.search(np.packbits(query > 0, axis=1), 100))
    report(ds, d, "sign BQ", "hamming", d / 8, ids, dt, gt)

    for nb in (1, 2, 4):
        ix = faiss.IndexRaBitQ(d, faiss.METRIC_INNER_PRODUCT, nb)
        ix.train(base)
        ix.add(base)
        (_, ids), dt = one_thread(lambda: ix.search(query, 100))
        report(ds, d, "RaBitQ", f"bits={nb}", ix.sa_code_size(), ids, dt, gt)

    # FAISS's fast-scan layouts: codes in blocks of 32 vectors, scored by SIMD table lookups.
    # These are its speed-tuned versions, and the fair comparison for throughput.
    for nb in (1, 2, 4):
        ix = faiss.IndexRaBitQFastScan(d, faiss.METRIC_INNER_PRODUCT, 32, nb)
        ix.train(base)
        ix.add(base)
        (_, ids), dt = one_thread(lambda: ix.search(query, 100))
        report(ds, d, "RaBitQ FS", f"bits={nb}", ix.code_size, ids, dt, gt)

    for m in (d // 8, d // 4, d // 2):
        ix = faiss.IndexPQ(d, m, 8, faiss.METRIC_INNER_PRODUCT)
        ix.train(base)
        ix.add(base)
        (_, ids), dt = one_thread(lambda: ix.search(query, 100))
        report(ds, d, "FAISS PQ", f"M={m},nbits=8", ix.sa_code_size(), ids, dt, gt)

    for m in (d // 4, d // 2, d):  # 4-bit sub-quantizers, so 1, 2 and 4 bits per dimension
        ix = faiss.IndexPQFastScan(d, m, 4, faiss.METRIC_INNER_PRODUCT)
        ix.train(base)
        ix.add(base)
        (_, ids), dt = one_thread(lambda: ix.search(query, 100))
        report(ds, d, "PQ FS", f"M={m},nbits=4", m / 2, ids, dt, gt)

if rows:
    with open("data/highdim.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote data/highdim.csv", file=sys.stderr)
