"""Normalize SIFT10K and compute inner-product ground truth.

Every system in the comparison reads these files, so the inputs are identical by
construction rather than by three separate loaders agreeing.

Normalization matters: turbovec is an inner-product index, SIFT's published ground
truth is L2, and FAISS supports both. On unit-norm vectors the three metrics induce
the *same* ranking, so normalizing removes the metric mismatch entirely instead of
papering over it.
"""
import numpy as np, os, sys

SRC = "data/siftsmall"
DST = "data/sift-norm"


def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return raw.reshape(-1, dim + 1)[:, 1:].copy().view(np.float32)


def write_fvecs(path, arr):
    n, d = arr.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = arr.astype(np.float32).view(np.int32)
    out.tofile(path)


def write_ivecs(path, arr):
    n, k = arr.shape
    out = np.empty((n, k + 1), dtype=np.int32)
    out[:, 0] = k
    out[:, 1:] = arr.astype(np.int32)
    out.tofile(path)


def main():
    if not os.path.isdir(SRC):
        sys.exit(f"{SRC} missing — run tools/fetch_datasets.sh")
    os.makedirs(DST, exist_ok=True)

    base = read_fvecs(f"{SRC}/siftsmall_base.fvecs")
    query = read_fvecs(f"{SRC}/siftsmall_query.fvecs")

    def unit(a):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return a / n

    base_n, query_n = unit(base), unit(query)

    # Exact inner-product ground truth, top-100, computed on the normalized data
    # that every system will actually index.
    sims = query_n @ base_n.T
    gt = np.argsort(-sims, axis=1)[:, :100]

    write_fvecs(f"{DST}/base.fvecs", base_n)
    write_fvecs(f"{DST}/query.fvecs", query_n)
    write_ivecs(f"{DST}/groundtruth.ivecs", gt)

    # Separation ratio: how much closer the true NN is than the 100th. A corpus
    # where this is near 1.0 makes every method look identical.
    top = np.take_along_axis(sims, gt, axis=1)
    print(f"base {base_n.shape}  query {query_n.shape}  gt {gt.shape}")
    print(f"sim(NN) mean {top[:,0].mean():.4f}   sim(100th) mean {top[:,99].mean():.4f}")
    print(f"separation (NN-100th) mean {np.mean(top[:,0]-top[:,99]):.4f}")
    print(f"wrote -> {DST}")


if __name__ == "__main__":
    main()
