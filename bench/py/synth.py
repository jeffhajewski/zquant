"""Synthetic corpora with controlled structure.

Real corpora tell you *that* something happens; a knob tells you *when*. This exists
because the measured results have asymmetries no real dataset here explains — calibration
is worth +7 points on SIFT and ~0 on nytimes, and the only difference offered so far is
"SIFT is anisotropic", which is a label rather than a threshold.

Each preset isolates one property:

  iso        Isotropic Gaussian directions. The control: a random rotation cannot improve
             on it, so calibration should be worth nothing and anything that claims a gain
             here is measuring noise or a bug.

  spectrum   Power-law covariance, lambda_i proportional to i**-alpha. alpha=0 is `iso`;
             larger alpha concentrates variance in fewer directions. Real embeddings have
             power-law spectra, so this sweeps the axis that separates SIFT from nytimes
             with a number attached.

  outlier    A few channels with much larger scale and a nonzero mean, which is what
             transformer KV caches actually look like ("massive activations"). A stated
             target of this library that no benchmark here has ever exercised.

  offset     Isotropic directions plus a shared mean vector, so the corpus centroid sits
             away from the origin. A random rotation preserves that offset, and a
             per-coordinate shift is exactly what removes it — this is the knob the
             `spectrum` sweep showed was the real one, since calibration pays nothing at
             effective rank 31 but SIFT is nonnegative histograms with a large centroid.

  cluster    Mixture of von Mises-Fisher-ish clusters. Real corpora are clumpy, and
             neighbourhood density is what recall@10 is actually sensitive to; uniform
             corpora make retrieval look easier than it is.

Ground truth is exact inner product over the normalized corpus, computed the same chunked
way prepare.py does it, so every system reads identical files.

Synthetic data is for explaining behaviour under a controlled sweep. Competitive claims
belong on the real corpora.
"""
import argparse, os
import numpy as np


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


def random_rotation(d, rng):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    # Sign-fix the QR so the result is Haar-distributed rather than merely orthogonal.
    return (q * np.sign(np.diag(r))).astype(np.float32)


def gen(kind, n, d, rng, alpha, clusters, n_outliers, outlier_scale):
    if kind == "iso":
        return rng.standard_normal((n, d)).astype(np.float32)

    if kind == "spectrum":
        scale = (np.arange(1, d + 1) ** (-alpha / 2.0)).astype(np.float32)
        x = rng.standard_normal((n, d)).astype(np.float32) * scale
        return x @ random_rotation(d, rng)

    if kind == "outlier":
        x = rng.standard_normal((n, d)).astype(np.float32)
        idx = rng.choice(d, size=n_outliers, replace=False)
        # Large scale *and* a large shared mean: KV-cache outlier channels are offset,
        # not just wide, and an offset is what a per-coordinate shift can remove while a
        # scale alone cannot.
        x[:, idx] *= outlier_scale
        x[:, idx] += outlier_scale
        return x

    if kind == "offset":
        # `mu` is per-coordinate; after normalization what matters is the norm of the
        # centroid of the *directions*, which `describe` reports as `centroid`.
        x = rng.standard_normal((n, d)).astype(np.float32)
        return x + alpha

    if kind == "cluster":
        centres = rng.standard_normal((clusters, d)).astype(np.float32)
        centres /= np.linalg.norm(centres, axis=1, keepdims=True)
        who = rng.integers(0, clusters, size=n)
        # Concentration chosen so clusters overlap somewhat rather than separating
        # cleanly, which would make retrieval trivially easy.
        return centres[who] + 0.35 * rng.standard_normal((n, d)).astype(np.float32)

    raise SystemExit(f"unknown kind {kind}")


def describe(x):
    """Diagnostics that say whether the knob actually did anything."""
    sd = x.std(axis=0)
    cov_eig = np.linalg.eigvalsh(np.cov(x[: min(len(x), 20000)].T.astype(np.float64)))
    cov_eig = np.clip(cov_eig, 0, None)
    # Participation ratio: effective number of directions carrying variance.
    eff = (cov_eig.sum() ** 2) / max((cov_eig**2).sum(), 1e-30)
    # Norm of the centroid of the unit directions: how far off-origin the corpus sits,
    # which is what a per-coordinate shift can remove and a scale cannot.
    u = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
    centroid = float(np.linalg.norm(u.mean(axis=0)))
    return dict(eff_rank=eff, dim=x.shape[1], centroid=centroid,
                chan_ratio=float(sd.max() / max(sd.mean(), 1e-9)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["iso", "spectrum", "outlier", "cluster", "offset"])
    p.add_argument("-n", type=int, default=100_000)
    p.add_argument("-d", type=int, default=768)
    p.add_argument("--nq", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=1.0, help="spectrum decay exponent")
    p.add_argument("--clusters", type=int, default=200)
    p.add_argument("--outliers", type=int, default=4, help="heavy channels")
    p.add_argument("--outlier-scale", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None)
    a = p.parse_args()

    gib = a.n * a.d * 4 / 2**30
    print(f"generating {a.kind}: {a.n} x {a.d} ({gib:.2f} GiB base)")

    rng = np.random.default_rng(a.seed)
    both = gen(a.kind, a.n + a.nq, a.d, rng, a.alpha, a.clusters, a.outliers, a.outlier_scale)
    stats = describe(both)
    base, query = both[: a.n], both[a.n :]

    def unit(v):
        nrm = np.linalg.norm(v, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return (v / nrm).astype(np.float32)

    base, query = unit(base), unit(query)

    gt = np.empty((a.nq, 100), dtype=np.int32)
    top = np.empty((a.nq, 100), dtype=np.float32)
    for i in range(0, a.nq, 64):
        sims = query[i : i + 64] @ base.T
        idx = np.argpartition(-sims, 100, axis=1)[:, :100]
        order = np.take_along_axis(sims, idx, 1).argsort(axis=1)[:, ::-1]
        gt[i : i + 64] = np.take_along_axis(idx, order, 1)
        top[i : i + 64] = np.take_along_axis(sims, gt[i : i + 64], 1)

    tag = a.tag or {
        "iso": "iso",
        "spectrum": f"spec{a.alpha:g}",
        "outlier": f"out{a.outliers}x{a.outlier_scale:g}",
        "cluster": f"clu{a.clusters}",
        "offset": f"off{a.alpha:g}",
    }[a.kind]
    name = f"synth-{tag}-{a.n}x{a.d}"
    dst = f"data/{name}"
    os.makedirs(dst, exist_ok=True)
    write_fvecs(f"{dst}/base.fvecs", base)
    write_fvecs(f"{dst}/query.fvecs", query)
    write_ivecs(f"{dst}/groundtruth.ivecs", gt)
    with open("data/dataset.txt", "w") as f:
        f.write(name)

    sep = float(np.mean(top[:, 0] - top[:, 99]))
    print(f"  effective rank {stats['eff_rank']:.1f} of {stats['dim']}   "
          f"max/mean channel sd {stats['chan_ratio']:.1f}   "
          f"centroid norm {stats['centroid']:.3f}")
    print(f"  sim(NN) {top[:,0].mean():.4f}  sim(100th) {top[:,99].mean():.4f}  separation {sep:.4f}")
    print(f"  wrote -> {dst}, and data/dataset.txt")


if __name__ == "__main__":
    main()
