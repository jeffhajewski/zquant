"""Normalize a corpus and compute exact inner-product ground truth.

Every system in the comparison reads the prepared files, so the inputs are identical
by construction rather than by three separate loaders agreeing.

Normalization matters: turbovec is an inner-product index, published ground truth is
usually L2 or angular, and FAISS supports both. On unit-norm vectors all three metrics
induce the *same* ranking, so normalizing removes the metric mismatch outright.

Usage:
    python bench/py/prepare.py                 # SIFT10K (fvecs)
    python bench/py/prepare.py nytimes-256     # an ann-benchmarks HDF5
"""
import numpy as np, os, sys


def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return raw.reshape(-1, dim + 1)[:, 1:].copy().view(np.float32)


def write_fvecs(path, arr):
    n, d = arr.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = np.ascontiguousarray(arr, dtype=np.float32).view(np.int32)
    out.tofile(path)


def write_ivecs(path, arr):
    n, k = arr.shape
    out = np.empty((n, k + 1), dtype=np.int32)
    out[:, 0] = k
    out[:, 1:] = arr.astype(np.int32)
    out.tofile(path)


def load_siftsmall():
    src = "data/siftsmall"
    if not os.path.isdir(src):
        sys.exit(f"{src} missing - run tools/fetch_datasets.sh")
    base = read_fvecs(f"{src}/siftsmall_base.fvecs")
    query = read_fvecs(f"{src}/siftsmall_query.fvecs")
    # 100 queries gives a binomial standard error near 1 point of R@10, the same size
    # as the differences between systems. Top up from the learn split - held out from
    # base, so still a fair query distribution - until the error bar can resolve them.
    learn = read_fvecs(f"{src}/siftsmall_learn.fvecs")
    rng = np.random.default_rng(0)
    query = np.vstack([query, learn[rng.choice(len(learn), size=900, replace=False)]])
    return base, query, "sift-norm"


def load_hdf5(name):
    import h5py
    path = f"data/{name}-angular.hdf5"
    if not os.path.exists(path):
        path = f"data/{name}.hdf5"
    if not os.path.exists(path):
        sys.exit(f"{path} missing")
    with h5py.File(path, "r") as f:
        base = np.array(f["train"], dtype=np.float32)
        query = np.array(f["test"], dtype=np.float32)
    # Cap the corpus so a brute-force ground truth stays tractable and every system
    # sees the same subset.
    limit = 100_000
    if len(base) > limit:
        rng = np.random.default_rng(0)
        base = base[np.sort(rng.choice(len(base), size=limit, replace=False))]
    if len(query) > 1000:
        query = query[:1000]
    return base, query, f"{name}-norm"


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "siftsmall"
    base, query, out_name = load_siftsmall() if which == "siftsmall" else load_hdf5(which)

    dst = f"data/{out_name}"
    os.makedirs(dst, exist_ok=True)

    def unit(a):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return a / n

    base_n, query_n = unit(base), unit(query)

    # Exact inner-product ground truth on the data every system will actually index.
    # Chunked so a large corpus does not need an n x m similarity matrix at once.
    gt = np.empty((len(query_n), 100), dtype=np.int32)
    top = np.empty((len(query_n), 100), dtype=np.float32)
    for i in range(0, len(query_n), 128):
        sims = query_n[i : i + 128] @ base_n.T
        idx = np.argpartition(-sims, 100, axis=1)[:, :100]
        ordered = np.take_along_axis(sims, idx, 1).argsort(axis=1)[:, ::-1]
        gt[i : i + 128] = np.take_along_axis(idx, ordered, 1)
        top[i : i + 128] = np.take_along_axis(sims, gt[i : i + 128], 1)

    write_fvecs(f"{dst}/base.fvecs", base_n)
    write_fvecs(f"{dst}/query.fvecs", query_n)
    write_ivecs(f"{dst}/groundtruth.ivecs", gt)
    with open("data/dataset.txt", "w") as f:
        f.write(out_name)

    se = np.sqrt(0.9 * 0.1 / (len(query_n) * 10))
    print(f"{out_name}: base {base_n.shape}  query {query_n.shape}")
    print(f"  sim(NN) {top[:,0].mean():.4f}  sim(100th) {top[:,99].mean():.4f}  "
          f"separation {np.mean(top[:,0]-top[:,99]):.4f}")
    print(f"  binomial se on R@10 at ~0.9: {se*100:.2f} points")
    print(f"  wrote -> {dst}, and data/dataset.txt")


if __name__ == "__main__":
    main()
