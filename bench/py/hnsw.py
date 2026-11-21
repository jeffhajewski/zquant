"""zquant against HNSW graph indexes, through the Python client, on prepared corpora named
on the command line.

    python bench/py/hnsw.py cohere-768-norm openai-1536-norm cohere-768-1m-norm

HNSW is the usual choice when memory is plentiful and latency matters most. This puts
zquant's partitioned index, alone and reranked against the original vectors, beside
hnswlib and FAISS's IndexHNSWFlat and IndexHNSWSQ (8-bit codes), each over a sweep of
search effort: probes for zquant, efSearch for HNSW.

R10@10 is the true top-10 found in the returned top-10. QPS is one thread, k=10, two ways:
all 1000 queries in one call, which zquant regroups by list so each list is read once for
every query that wants it, and one query per call, as a latency-bound service sends them.
Bytes per vector are what the saved index takes, graph and centroids included, plus the
float32 originals where they're used to rerank. Builds run on 8 threads.

zquant runs alone at bits=3 and 5 on about sqrt(n) lists, and reranked at bits=2, which
finds as much as wider codes once reranked and scans fastest, on sqrt(n) and 4 sqrt(n)
lists. HNSW uses M=16 and M=32 with efConstruction=200, FAISS's own default of 40 being
well below what it's usually run at. hnswlib's SIMD distances are x86-only, so on ARM it
runs plain loops; FAISS is the graph baseline there. Needs numpy, faiss-cpu, hnswlib and
the zquant Python client on PYTHONPATH (`zig build lib`, then PYTHONPATH=clients/python).
Writes data/hnsw.csv.
"""

import csv, math, os, sys, tempfile, time
import numpy as np
import zquant

try:
    import faiss
except ImportError:
    faiss = None
try:
    import hnswlib
except ImportError:
    hnswlib = None


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
    """The fastest of `repeats` runs."""
    best, out = math.inf, None
    for _ in range(repeats):
        t = time.perf_counter()
        out = search()
        best = min(best, time.perf_counter() - t)
    return out, best


def one_at_a_time(search, queries, repeats=2):
    """Queries a second with one query per call: the fastest of `repeats` passes."""
    best = math.inf
    for _ in range(repeats):
        t = time.perf_counter()
        for i in range(len(queries)):
            search(queries[i:i + 1])
        best = min(best, time.perf_counter() - t)
    return round(len(queries) / best)


def saved_bytes(save):
    """Bytes the index takes on disk, by saving it to a scratch file."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "index")
        save(path)
        return os.path.getsize(path)


rows = []


def report(ds, system, config, effort, bpv, build_s, ids, dt, single, gt):
    r = dict(dataset=ds, system=system, config=config, effort=effort,
             bytes_per_vector=round(bpv, 1), build_s=round(build_s, 1),
             r10_at_10=round(recall(ids, gt), 4), qps_1thread=round(len(gt) / dt),
             qps_one_per_call=single)
    rows.append(r)
    print(f"{system:8s} {config:28s} {effort!s:>5}  {bpv:7.1f} B  build {build_s:6.1f} s  "
          f"R10@10 {r['r10_at_10']:.3f}  {r['qps_1thread']:7d} QPS  {single:7d} one per call", flush=True)


for ds in sys.argv[1:]:
    p = f"data/{ds}"
    base, query, gt = read_fvecs(f"{p}/base.fvecs"), read_fvecs(f"{p}/query.fvecs"), read_ivecs(f"{p}/groundtruth.ivecs")
    n, d = base.shape
    root = max(1, round(math.sqrt(n)))
    print(f"\n{ds}: {n} x {d}, {len(query)} queries", flush=True)

    def zquant_index(bits, lists):
        sample = base[np.random.default_rng(0).choice(n, min(n, 256 * lists), replace=False)]
        ix = zquant.Index(d, bits=bits, metric="ip")
        t = time.perf_counter()
        ix.partition(sample, lists, threads=8)
        ix.add(base, threads=8)
        return ix, time.perf_counter() - t, saved_bytes(ix.save) / n

    def probe_sweep(lists):
        return sorted({max(1, lists // f) for f in (64, 32, 16, 8, 4, 2)})

    # zquant alone, on about sqrt(n) lists as in the README's million-vector example.
    for bits in (3, 5):
        ix, build, bpv = zquant_index(bits, root)
        for pr in probe_sweep(root):
            (ids, _), dt = timed(lambda: ix.search(query, k=10, probes=pr))
            single = one_at_a_time(lambda q: ix.search(q, k=10, probes=pr), query)
            report(ds, "zquant", f"bits={bits} lists={root}", pr, bpv, build, ids, dt, single, gt)
        ix.close()

    # Reranked: fetch more candidates from the codes, then rescore them exactly against the
    # original vectors, which then have to be kept too.
    for lists in (root, 4 * root):
        ix, build, bpv = zquant_index(2, lists)
        for cand in (40, 100):
            for pr in probe_sweep(lists):
                (ids, _), dt = timed(lambda: ix.search_rerank(query, base, k=10, candidates=cand, probes=pr))
                single = one_at_a_time(lambda q: ix.search_rerank(q, base, k=10, candidates=cand, probes=pr), query)
                report(ds, "zquant", f"bits=2 lists={lists} rerank {cand}", pr, bpv + 4 * d, build, ids, dt, single, gt)
        ix.close()

    ef_sweep = (16, 32, 64, 128, 256, 512)
    if hnswlib is not None:
        for m in (16, 32):
            ix = hnswlib.Index(space="ip", dim=d)
            t = time.perf_counter()
            ix.init_index(max_elements=n, M=m, ef_construction=200, random_seed=0)
            ix.add_items(base, np.arange(n), num_threads=8)
            build = time.perf_counter() - t
            bpv = saved_bytes(ix.save_index) / n
            for ef in ef_sweep:
                ix.set_ef(max(ef, 10))
                (ids, _), dt = timed(lambda: ix.knn_query(query, k=10, num_threads=1))
                single = one_at_a_time(lambda q: ix.knn_query(q, k=10, num_threads=1), query)
                report(ds, "hnswlib", f"M={m}", ef, bpv, build, ids, dt, single, gt)
            del ix

    if faiss is not None:
        for name, m in (("HNSWFlat", 16), ("HNSWFlat", 32), ("HNSWSQ8", 32)):
            faiss.omp_set_num_threads(8)
            if name == "HNSWFlat":
                ix = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
            else:
                ix = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, m, faiss.METRIC_INNER_PRODUCT)
            ix.hnsw.efConstruction = 200
            t = time.perf_counter()
            ix.train(base)
            ix.add(base)
            build = time.perf_counter() - t
            bpv = saved_bytes(lambda path: faiss.write_index(ix, path)) / n
            faiss.omp_set_num_threads(1)
            for ef in ef_sweep:
                ix.hnsw.efSearch = ef
                (_, ids), dt = timed(lambda: ix.search(query, 10))
                single = one_at_a_time(lambda q: ix.search(q, 10), query)
                report(ds, "FAISS", f"{name} M={m}", ef, bpv, build, ids, dt, single, gt)
            del ix

if rows:
    with open("data/hnsw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote data/hnsw.csv", file=sys.stderr)
