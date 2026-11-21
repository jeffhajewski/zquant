# zquant (Python)

```python
import numpy as np, zquant

index = zquant.Index(dim=256, bits=5)
index.add(vectors)                       # (n, 256), any float array-like
ids, scores = index.search(queries, k=10)
```

## Install

Each GitHub release has a wheel per platform, with the native library inside: Linux
x86_64 and arm64 (manylinux_2_17, so glibc 2.17 or newer), macOS 13+ on arm64 and
x86_64, and Windows x86_64. One wheel serves every Python from 3.9 up. Download them
all and let pip choose:

```sh
pip install --no-index --find-links ./wheels zquant    # after `pip install numpy`
```

The Linux and Windows wheels also carry a portable library, which the package loads
instead on a CPU without SSSE3 (x86_64) or the dot-product extension (arm64). It is
correct but several times slower; `zquant.build_info().portable` says which you got.

The package is plain Python over `ctypes`, so it needs the native library, `libzquant`,
from somewhere. It looks in three places, in this order:

1. `ZQUANT_LIB`, if set: the path to the library file itself. If that file is missing,
   importing fails rather than quietly loading a different one.
2. The package directory, which is where a wheel puts it.
3. The repository's `zig-out/lib`, for working on the binding itself.

For development:

```sh
zig build lib                    # at the repository root
cd clients/python && pip install -e ".[test]"
pytest
```

To build a wheel with the library inside it:

```sh
clients/python/scripts/build_wheel.sh            # wheel lands in clients/python/dist
clients/python/scripts/build_wheel.sh --no-build-isolation   # offline
```

The script builds the library for the same CPU the release builds target, copies it into
the package, runs `pip wheel`, and cleans up after itself. The wheel is tagged for your
platform. `PYTHON`, `WHEEL_DIR`, `ZIG_CPU` and `ZQUANT_PORTABLE` adjust it; the top of the
script explains each. Without a library copied in, `pip wheel` gives a pure-Python wheel
that relies on `ZQUANT_LIB`.

### CPU check

Release libraries use SSSE3 on x86_64 and the dot-product extension on arm64, because
search is several times slower without them. Before using the library, the binding asks
it whether this CPU has those instructions. If not, it loads the portable build from the
same directory instead: `libzquant-portable.so`, `libzquant-portable.dylib` or
`zquant-portable.dll`. If there isn't one,
`import zquant` raises `CpuNotSupportedError`, naming the CPU the library was built for.
The wheel script bundles a portable build wherever the release CPU needs one.

`zquant.build_info()` tells you what got loaded:

```python
>>> zquant.build_info()
BuildInfo(version='0.1.0', cpu='apple_m1', kernels='zquant-kernels shuffle=native dot=native',
          library=PosixPath('.../zquant/libzquant.dylib'), portable=False)
```

If `kernels` says `scalar`, you're on the slow fallback.

## What it is

By default a **flat** index: every query is compared against every vector, so recall is
limited only by quantization, and cost grows linearly with the corpus. Call `partition`
before adding and it becomes a partitioned (IVF) index instead, which scans only the
parts of the corpus nearest each query (see [Partitioning](#partitioning)).

`bits` is 2 to 6. Storage is roughly `dim * (bits-1) / 8` bytes per vector, and
`index.bytes_per_vector` gives the exact figure. On 100k × 256 embeddings, `bits=5` gives
132 B/vector at R@10 0.916. `dim` can be anything from 1 to 65536.

## Adding vectors

```python
index.add(vectors)               # ids continue from len(index)
index.add(vectors, threads=8)    # encode on 8 threads
```

Ids are positions in insertion order: the first vector you add is 0. With `threads`, the
index comes out byte-for-byte the same as without it. Small batches stay on the calling
thread anyway, because starting threads costs more than encoding a few hundred vectors.

A batch with a NaN or infinity anywhere in it is refused whole with
`InvalidArgumentError`, and nothing from it is added.

## Searching

`search` returns `(ids, scores)`, each shaped `(n_queries, k)`, best first. If the index
holds fewer than `k` vectors, the rest of each row is id `2**32-1` and score `-inf`.
`threads=N` splits one batch of queries across N workers, which only helps when the batch
has at least N queries.

### Reranking

The quantized scores rank well but not perfectly. If you still have the original vectors,
`search_rerank` fetches more candidates than you asked for, rescores them exactly, and
keeps the best:

```python
ids, scores = index.search_rerank(queries, originals, k=10, candidates=50)
```

`originals` row `i` must be the vector you added as id `i`, and it has to cover the whole
index. `candidates` defaults to `4 * k`; over-fetching by 4–10× gets close to exact top-k.
The scores that come back are exact, using the index's metric (similarity for `ip` and
`cosine`, negated squared distance for `l2`).

### Scoring particular ids

```python
scores = index.score_ids(query, [17, 3, 942])
```

This gives the same estimate `search` would report for those ids, without scanning
everything. It's useful when something else picked the candidates, like a metadata
filter or another index. An id outside the index raises `IndexError`.

## Calibration

`index.calibrate(sample)` fits the corpus centre before any `add`. Each vector is then
stored relative to the centre, and the centre's share of every score is computed exactly
rather than estimated. It costs 2 bytes per vector. Whether it helps depends on how far
the data's directions sit from the origin:

```python
u = x / np.linalg.norm(x, axis=1, keepdims=True)
centroid = np.linalg.norm(u.mean(axis=0))     # above ~0.3: worth it
```

Measured at bits 2 and 3: 9 to 22 points more recall@10 on corpora at 0.65 and above,
which is where real embeddings sit, and about a point at 0.12.

If you don't know your data, let the library decide:

```python
result = index.calibrate(sample, auto=True)
result.centroid_norm    # what it measured
result.applied          # whether it calibrated (it does at 0.3 and above)
```

Either way, calibration has to happen before the first `add`, and a few hundred rows of
sample is plenty.

## Partitioning

`index.partition(sample, lists=n)` splits the index into `n` lists by k-means, before any
`add`. Each vector goes to the list whose centroid is nearest its direction, and is
stored relative to that centroid. A search ranks the centroids against each query and
scans only the nearest few lists:

```python
index = zquant.Index(dim=768, bits=3)
index.partition(sample, lists=1000, threads=8)   # about sqrt(n) lists
index.add(vectors, threads=8)
ids, scores = index.search(queries, k=10, probes=64)
index.probes = 64                                  # or set it once
```

`probes` is how many lists each query scans, an eighth of them unless you say otherwise.
More finds more of the true neighbours and costs proportionally more time. Storing each
vector against a nearby centroid also makes its estimate more accurate than against the
one corpus centre `calibrate` fits, so a partitioned index can beat a calibrated flat one
on recall as well as speed. On 100k Cohere embeddings at `bits=2`, scanning 6% of the
lists found R@10 0.775 at 11k queries a second on one core, against 0.750 at 1.8k for
the calibrated flat index.

Partitioning takes the place of calibration, costs 6 more bytes per vector (4 for its id
and 2 naming its list) plus `lists * dim * 4` bytes of centroids, and can be done once.
The sample needs at least `lists` rows; 256 per list is all it uses. A partitioned index
saves and loads with its lists, and `score_ids` scores any id in it, whichever list it is
in.

## Saving and loading

```python
index.save("corpus.zq")
index = zquant.Index.load("corpus.zq")

data = index.to_bytes()
index = zquant.Index.from_bytes(data)
```

The format is versioned and checksummed, and it's the same on every platform. A loaded
index searches and accepts new vectors exactly as the saved one did. `save` writes to a
temporary file and renames it, so a crash halfway never leaves a truncated file behind.
Anything that isn't an intact index from a compatible version raises `FormatError`.

A loaded index knows how it was built: `index.bits`, `index.metric`, `index.seed`,
`index.compact` and `index.calibrated` report what was saved, so a file is enough to
build a matching index:

```python
twin = zquant.Index(dim=index.dim, bits=index.bits, metric=index.metric,
                    seed=index.seed, compact=index.compact)
```

## Threads

An `Index` can be shared between threads.

- **Searches run in parallel.** Each `search`, `search_rerank` or `score_ids` call checks
  out its own native searcher from a small pool, and the native call releases the GIL.
- **Writes wait their turn.** `add`, `calibrate` and `close` wait for running searches to
  finish, and new searches wait for them. A steady stream of searches can't starve a
  writer.
- **Closing is safe.** After `close`, every call raises `StateError` instead of touching
  freed memory.

A `Codec` serializes its own calls with a lock, so sharing one is safe, but the calls take
turns. To encode on several threads at once, give each thread its own `Codec`.

## Codec

`Codec` compresses vectors and gives them back, with no index involved:

```python
codec = zquant.Codec(dim=64, bits=4)
codes, norms = codec.encode(keys)      # keep both
keys_back = codec.decode(codes, norms)
```

Codes are bit-packed, so `codec.code_bytes` is the real storage per vector, plus four
bytes for the norm. Codec `bits` is the codebook width (1 to 6), while `Index` keeps one of
its bits for a residual sketch, so `Codec(bits=4)` packs as many bits per coordinate as
`Index(bits=5)`.

As a KV-cache quantizer, it isn't the best one measured. At matched memory, KIVI-style
4-bit quantization had lower perplexity on every model tried. zquant's keys failed
outright on a model whose keys carry a large per-channel offset.
[docs/kv-cache.md](../../docs/kv-cache.md) has the numbers and caveats.

## Errors

Every library failure is a `zquant.ZquantError`, with the status code in `.status`. Each
status also has its own subclass:

| status | exception | also a |
|---|---|---|
| `ZQ_ERR_ALLOC` | `OutOfMemoryError` | `MemoryError` |
| `ZQ_ERR_INVALID` | `InvalidArgumentError` | `ValueError` |
| `ZQ_ERR_STATE` | `StateError` | |
| `ZQ_ERR_UNSUPPORTED` | `UnsupportedError` | |
| `ZQ_ERR_CPU` | `CpuNotSupportedError` | |
| `ZQ_ERR_FORMAT` | `FormatError` | `ValueError` |

Arguments the binding can check itself are checked before any native call, and the
message says which argument was wrong and what range it should be in. That covers `dim`
(1–65536), `bits` (2–6 for an index, 1–6 for a codec), `seed` (0 to 2⁶⁴−1), `threads`
(0–1024), `k`, and shapes. Those raise `ValueError` or `TypeError`. A missing library
raises `LibraryNotFoundError`, which is an `OSError`.

## Input

Vectors can be anything numpy can turn into real numbers: float32 or float64 arrays,
integer arrays, lists, Fortran-ordered or strided views. A 1-D input is one vector. All of
it is converted to C-contiguous float32, so passing that already avoids a copy. Complex,
string and object arrays raise `TypeError`.
