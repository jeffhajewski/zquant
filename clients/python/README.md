# zquant (Python)

```python
import numpy as np, zquant

index = zquant.Index(dim=256, bits=5)
index.add(vectors)                       # (n, 256) float32
ids, scores = index.search(queries, k=10, threads=8)
```

## Install

The package needs `libzquant` beside it. During development:

```sh
zig build lib          # at the repository root
cd clients/python && pip install -e .
```

The binding finds the library next to the package, then in `../../zig-out/lib`, and
honours `ZQUANT_LIBRARY` if you want to point it somewhere else.

## What it is

A **flat** index: every vector is scanned for every query. There is no graph or
partitioning, so recall is limited only by quantization and cost grows linearly with the
corpus. That is the right shape up to a few million vectors, or whenever predictable
recall matters more than sublinear search.

Storage is roughly `dim * (bits-1) / 8` bytes per vector; `index.bytes_per_vector` reports
the exact figure. On 100k × 256 embeddings, `bits=5` gives 132 B/vector at R@10 0.916.

## Calibration

`index.calibrate(sample)` fits a per-coordinate shift and scale before any `add`. Whether
it helps is predictable from one statistic — how far the corpus of unit directions sits
from the origin:

```python
u = x / np.linalg.norm(x, axis=1, keepdims=True)
centroid = np.linalg.norm(u.mean(axis=0))     # >0.3: enable it
```

Measured: +11 points of recall at centroid 0.65, nothing at 0.12. It is not free — on
low-rank zero-mean data it has cost several points — which is why it is opt-in.

## Notes

- Vectors are float32 row-major. Other dtypes are accepted and converted, at the cost of a
  copy.
- An `Index` is safe to search from several threads only through separate `search` calls
  with distinct `threads` values; simplest is to pass `threads=N` and let the library
  spread one batch. `add` and `calibrate` are not concurrent-safe.
- `threads>1` only pays for batches at least that large.
