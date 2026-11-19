"""Estimator and IVF experiments on a numpy copy of zquant's encoder.

    python bench/py/ivf_lab.py cohere-768-norm --nlists 1,64,256,1024 --curve 1024 --faiss

The library cannot yet express the variants tested here, so this re-implements its index
encoder in numpy and checks the copy against the library first. The copy is the
algorithm in src/index/flat.zig with one substitution, a dense random orthogonal rotation
in place of the randomized Hadamard transform:

    y = R x/|x|,  z = (y + shift) * scale,  code = nearest Lloyd-Max level,
    y^ = level/scale - shift,  a = <y,y^>/|y^|^2,  score = |x| a <Rq, y^>

with (shift, scale) from `calibrate`'s quantile anchoring. Its recall lands within a point
of the library's on every corpus tried except fashion-mnist, where it is up to 3.7 points
lower; the `library` and `lab` rows show both.

The variants:

  flat a=exact      a = |y|^2/<y,y^> instead of the least-squares a. The score is then
                    exact when the query points along the stored vector, which is where
                    the top of a ranking is decided. (RaBitQ normalises this way.)
  ivfN split        k-means into N lists; each vector is stored as its list centroid c
                    plus a quantized residual r = y - c, normalised to unit length before
                    quantizing. The score is <p,c> + <c,r> + a<p-c, r^>: the first two
                    terms are exact (one per list per query, one per vector at encode
                    time) and only the last is estimated. N=1 is centring on the mean.
  +cal              the quantile-anchored (shift, scale), fitted on normalised residuals.

Every list is probed in the first part, so any change there is the codes alone. The
second part (--curve N) probes the nearest lists only and reports recall against the
fraction of the corpus scanned, beside the exact-score ceiling for the same lists and,
with --faiss, FAISS IVF-RaBitQ and IVF-PQ built on the same centroids. Queries are scored
in float here, where the library quantizes them to int8; the library and lab rows show how
far apart the two land. Needs numpy, faiss-cpu, and the zquant Python client.
"""
import argparse, csv, os, sys
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


def sphere_grid(d, points=400_001):
    """The marginal of one coordinate of a uniform unit vector in d dimensions."""
    t = np.linspace(-1, 1, points)
    w = np.clip(1 - t * t, 0, None) ** ((d - 3) / 2)
    return t, w / w.sum()


def lloyd_max(d, bits, iters=300):
    t, w = sphere_grid(d)
    levels = 2 ** bits
    c = np.interp((np.arange(levels) + 0.5) / levels, np.cumsum(w), t)
    for _ in range(iters):
        cell = np.searchsorted((c[1:] + c[:-1]) / 2, t)
        c = (np.bincount(cell, weights=w * t, minlength=levels)
             / np.maximum(np.bincount(cell, weights=w, minlength=levels), 1e-300))
    return c


class Encoder:
    def __init__(self, d, bits, seed=0):
        """`bits` as Index takes it: the codebook uses bits-1."""
        q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
        self.R = (q * np.sign(np.diag(r))).astype(np.float32)
        self.levels = lloyd_max(d, bits - 1)
        self.bounds = (self.levels[1:] + self.levels[:-1]) / 2
        self.outer = float(np.abs(self.levels).max())
        t, w = sphere_grid(d)
        self.p_hi = float(w[t <= self.outer].sum())

    def rotate(self, x):
        return x @ self.R.T

    def fit(self, y):
        """Per-coordinate (shift, scale) by quantile anchoring, as FlatIndex.calibrate."""
        rows = len(y)
        lo = min(int(rows * (1 - self.p_hi)), rows - 2)
        hi = max(min(int(rows * self.p_hi), rows - 1), lo + 1)
        s = np.sort(y, axis=0)
        span = s[hi] - s[lo]
        ok = span > 1e-20
        scale = np.where(ok, 2 * self.outer / np.where(ok, span, 1), 1.0)
        shift = np.where(ok, -0.5 * (s[lo] + s[hi]), 0.0)
        return shift.astype(np.float32), scale.astype(np.float32)

    def quantize(self, z, cal=None):
        """Reconstruction of z in its own (pre-shift) basis."""
        if cal is None:
            return self.levels[np.searchsorted(self.bounds, z)].astype(np.float32)
        shift, scale = cal
        return self.levels[np.searchsorted(self.bounds, (z + shift) * scale)].astype(np.float32) / scale - shift


def alpha(target, recon, mode):
    dot = np.einsum("ij,ij->i", target, recon)
    if mode == "ls":
        return dot / np.maximum(np.einsum("ij,ij->i", recon, recon), 1e-30)
    return np.einsum("ij,ij->i", target, target) / np.where(np.abs(dot) > 1e-30, dot, 1e-30)


def split(enc, y, c, mode, cal=None):
    """Stored form of y = c + r, as (vectors, bias): score = <p, vectors> + bias."""
    r = y - c
    rho = np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30)
    rh = rho * enc.quantize(r / rho, cal)
    a = alpha(r, rh, mode)
    bias = np.einsum("ij,ij->i", c, r) - a * np.einsum("ij,ij->i", c, rh)
    return c + a[:, None] * rh, bias.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--bits", default="2,3,5")
    ap.add_argument("--nlists", default="1,64,256,1024")
    ap.add_argument("--curve", type=int, default=0)
    ap.add_argument("--curve-bits", default="2,3")
    ap.add_argument("--nprobes", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--faiss", action="store_true")
    args = ap.parse_args()

    p = f"data/{args.dataset}"
    base, query = read_fvecs(f"{p}/base.fvecs"), read_fvecs(f"{p}/query.fvecs")
    gt = read_ivecs(f"{p}/groundtruth.ivecs")
    n, d = base.shape
    nq, K = len(query), 100
    ncpu = faiss.omp_get_max_threads()
    sample = np.random.default_rng(0).choice(n, min(4096, n), replace=False)
    print(f"{args.dataset}: {n} x {d}, {nq} queries", flush=True)
    rows = []

    def recall(ids, depth):
        ids = np.asarray(ids)[:, :depth]
        return float(np.mean([len(set(ids[i].tolist()) & set(gt[i, :10].tolist())) / 10 for i in range(nq)]))

    def report(part, variant, bits, nlist, ids, nprobe="", frac=1.0, bpv=""):
        r = dict(dataset=args.dataset, part=part, variant=variant, bits=bits, nlist=nlist, nprobe=nprobe,
                 frac_scanned=round(frac, 4), bytes_per_vector=bpv,
                 r10_at_10=round(recall(ids, 10), 4), r10_at_100=round(recall(ids, 100), 4))
        rows.append(r)
        where = f"nprobe={nprobe:<4} scans {frac:6.2%}" if nprobe != "" else ""
        print(f"  {variant:28s} {where:24s} R10@10 {r['r10_at_10']:.4f}  R10@100 {r['r10_at_100']:.4f}", flush=True)

    def top(stored, pq, probe=None, assign=None):
        vectors, bias = stored if isinstance(stored, tuple) else (stored, None)
        out = np.empty((nq, K), np.int64)
        for i in range(0, nq, 200):
            s = pq[i:i + 200] @ vectors.T
            if bias is not None:
                s += bias[None, :]
            if probe is not None:
                s[~probe[i:i + 200][:, assign]] = -np.inf
            t = np.argpartition(-s, K - 1, axis=1)[:, :K]
            out[i:i + 200] = np.take_along_axis(t, np.argsort(-np.take_along_axis(s, t, 1), axis=1), 1)
        return out

    lists = {}
    for nlist in sorted({int(x) for x in args.nlists.split(",")} | ({args.curve} if args.curve else set())):
        if nlist == 1:
            c = base.mean(0, keepdims=True).astype(np.float32)
        else:
            km = faiss.Kmeans(d, nlist, niter=20, seed=1, max_points_per_centroid=10_000)
            km.train(base)
            c = km.centroids
        lists[nlist] = (c, faiss.knn(base, c, 1)[1][:, 0])

    for bits in [int(b) for b in args.bits.split(",")]:
        print(f"bits={bits}", flush=True)
        for cal in (False, True):
            ix = zquant.Index(d, bits=bits, metric="ip")
            if cal:
                ix.calibrate(base[sample])
            ix.add(base, threads=ncpu)
            report("A", "library flat" + (" +cal" if cal else ""), bits, 0, ix.search(query, k=K, threads=ncpu)[0],
                   bpv=ix.bytes_per_vector)
            ix.close()

        enc = Encoder(d, bits)
        y, pq = enc.rotate(base), enc.rotate(query)
        h = enc.quantize(y)
        report("A", "lab flat", bits, 0, top(alpha(y, h, "ls")[:, None] * h, pq))
        report("A", "lab flat a=exact", bits, 0, top(alpha(y, h, "exact")[:, None] * h, pq))
        h = enc.quantize(y, enc.fit(y[sample]))
        report("A", "lab flat +cal", bits, 0, top(alpha(y, h, "ls")[:, None] * h, pq))

        def residual_cal(c):
            r = y[sample] - c[sample]
            return enc.fit(r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-30))

        for nlist in [int(x) for x in args.nlists.split(",")]:
            cents, assign = lists[nlist]
            c = enc.rotate(cents)[assign]
            report("A", f"ivf{nlist} split", bits, nlist, top(split(enc, y, c, "ls"), pq))
            report("A", f"ivf{nlist} split a=exact", bits, nlist, top(split(enc, y, c, "exact"), pq))
            report("A", f"ivf{nlist} split a=exact +cal", bits, nlist,
                   top(split(enc, y, c, "exact", residual_cal(c)), pq))

        if args.curve and str(bits) in args.curve_bits.split(","):
            cents, assign = lists[args.curve]
            c = enc.rotate(cents)[assign]
            sizes = np.bincount(assign, minlength=args.curve)
            near = 2 * query @ cents.T - (cents * cents).sum(1)[None, :]
            stored = {
                "today's codes (+cal)": enc.quantize(y, enc.fit(y[sample])),
                "split a=exact": split(enc, y, c, "exact"),
                "split a=exact +cal": split(enc, y, c, "exact", residual_cal(c)),
            }
            h = stored["today's codes (+cal)"]
            stored["today's codes (+cal)"] = alpha(y, h, "ls")[:, None] * h
            faiss_ix = []
            if args.faiss:
                for name in ("FAISS IVF-RaBitQ", "FAISS IVF-PQ"):
                    quantizer = faiss.IndexFlatL2(d)
                    quantizer.add(cents)
                    if name == "FAISS IVF-PQ":
                        ix = faiss.IndexIVFPQ(quantizer, d, args.curve, d * (bits - 1) // 8, 8, faiss.METRIC_INNER_PRODUCT)
                    else:
                        ix = faiss.IndexIVFRaBitQ(quantizer, d, args.curve, faiss.METRIC_INNER_PRODUCT, True, bits - 1)
                    ix.train(base)
                    ix.add(base)
                    faiss_ix.append((name, ix, quantizer))
            for nprobe in [int(x) for x in args.nprobes.split(",") if int(x) <= args.curve]:
                probe = np.zeros((nq, args.curve), bool)
                np.put_along_axis(probe, np.argpartition(-near, nprobe - 1, axis=1)[:, :nprobe], True, 1)
                frac = float((probe * sizes[None, :]).sum(1).mean() / n)
                report("B", "exact scores", bits, args.curve, top(y, pq, probe, assign), nprobe, frac)
                for name, s in stored.items():
                    report("B", name, bits, args.curve, top(s, pq, probe, assign), nprobe, frac)
                for name, ix, _ in faiss_ix:
                    ix.nprobe = nprobe
                    report("B", name, bits, args.curve, ix.search(query, K)[1], nprobe, frac, ix.code_size)

    out = f"data/ivf_lab-{args.dataset}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
