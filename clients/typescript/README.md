# zquant (TypeScript / Node)

```ts
import { Index } from 'zquant';

const index = new Index({ dim: 256, bits: 5 });
index.add(vectors);                                    // Float32Array, n * dim
const { ids, scores } = index.search(queries, { k: 10, threads: 8 });
index.close();
```

Node 18 or later. The binding calls the C library through [koffi](https://koffi.dev), so
installing it needs no compiler.

## Install

Each GitHub release has an npm tarball carrying the native library for every platform
it supports: Linux x86_64 and arm64 (glibc 2.17 or newer), macOS 13+ on arm64 and
x86_64, and Windows x86_64.

```sh
npm install ./zquant-0.1.0.tgz
```

The Linux and Windows libraries come with a portable build, which the binding loads
instead on a CPU without SSSE3 (x86_64) or the dot-product extension (arm64).
`buildInfo().portable` says which you got.

During development, against the repository's own build:

```sh
zig build lib          # at the repository root
cd clients/typescript && npm install && npm test
```

### Where the native library comes from

The binding looks in three places, in this order, and uses the first it finds:

1. `ZQUANT_LIB`, if set: the full path to the library. If the file is not there you get
   an error rather than a quiet fallback to something else. (`ZQUANT_LIBRARY`, the old
   name, still works.)
2. `prebuilds/<platform>-<arch>/` inside the package, e.g. `prebuilds/linux-x64/`. This is
   what a packed tarball uses.
3. `zig-out/lib` in the repository, found by walking up from the package.

To build a tarball that carries the library, run `npm run pack:native` (that's
`scripts/pack.sh`). It copies `zig-out/lib`'s library into `prebuilds/` for the
machine you're on, runs `npm pack`, and removes `prebuilds/` again so a later test run
can't pick up a stale copy. A `libzquant-portable` beside the main library in
`zig-out/lib` is packed too. Build one with
`zig build lib -Dcpu=baseline -Dlib-name=zquant-portable`, or take it from a release's
`-portable` archive. `ZQUANT_PORTABLE_LIB` points at one somewhere else.

### Older CPUs

Release builds use SSSE3 on x86_64 and the dot-product extension on arm64, because
search is 6-10x slower without them. Before calling anything else, the binding asks the
library whether this CPU can run it. If it can't, the binding loads
`libzquant-portable.so` (`.dylib`, or `zquant-portable.dll` on Windows) from the same
directory instead. If there's no portable build there, you get an error that says which
CPU the library was built for, rather than a crash on an illegal instruction.

`buildInfo()` tells you what got loaded:

```ts
import { buildInfo } from 'zquant';
buildInfo();
// { version: '0.1.0', cpu: 'x86_64_v2', kernels: 'zquant-kernels shuffle=native dot=native',
//   libraryPath: '/…/prebuilds/linux-x64/libzquant.so', portable: false }
```

The library is loaded the first time you use it, not at import, so importing `zquant`
never throws.

## What it is

By default a **flat** index: every vector is scanned for every query, so recall is limited
only by quantization and cost grows linearly with the corpus. `partition` turns it into a
partitioned (IVF) index that scans only the parts of the corpus nearest each query.

Measured through this binding on 100k × 256 vectors at `bits=5`:

| | |
|---|---|
| storage | 132 B/vector — 13 MB against 102 MB as float32 |
| build | 227,000 vectors/s |
| search | 5,539 QPS single-threaded, 27,951 across ten threads |

That matches the native benchmark, so the FFI layer costs nothing measurable.

## Creating an index

```ts
new Index({
  dim: 768,          // 1..65536
  bits: 5,           // 2..6, default 5
  metric: 'cosine',  // 'ip' (default), 'inner_product', 'cosine' or 'l2'
  seed: 42n,         // number or bigint, default 0x5eed
  compact: true,     // packed codes (default); false trades ~2x memory for some speed
});
```

`bits` is the total budget per coordinate. One bit goes to a residual correction, so
`bits: 1` would leave nothing for the codes and is refused.

Seeds are 64-bit. A `number` seed has to be a safe integer (below 2^53); for anything
bigger pass a `bigint`, which is carried to the library exactly.

Bad options throw straight away with a message that says what was wrong, before the
library sees them. `MAX_DIM` (65536) and `MAX_THREADS` (1024) are exported if you want
to check limits yourself.

## Vectors

Vectors are float32, one per row. Anywhere the API takes vectors you can pass:

- one flat `Float32Array` of `n * dim` values (no copy, so this is the fastest);
- a `number[][]` or a `Float32Array[]`, one row each;
- a flat `number[]`.

Everything except the flat `Float32Array` gets copied into one. NaN and Infinity are
refused: the whole batch is rejected and nothing from it is added.

## Adding

```ts
index.add(vectors);                   // on the calling thread
index.add(vectors, { threads: 8 });   // encode on up to 8 threads
```

Threaded adds give the same index, byte for byte. Small batches stay on one thread
regardless, since below a few hundred vectors per thread starting threads costs more
than it saves. Ids are assigned in insertion order, starting at 0.

## Searching

```ts
const { ids, scores, k } = index.search(queries, { k: 10, threads: 4 });
```

You get `nq * k` ids and scores, query by query, best first. If the index holds fewer
than `k` vectors, the tail is padded with id `0xFFFFFFFF` and score `-Infinity`.
`threads` splits a batch of queries across workers, so it only helps for batches at least
that large.

### Reranking

Quantized scores are estimates. If you still have the original vectors, the index can
fetch more candidates than you need and rescore them exactly:

```ts
const { ids, scores } = index.searchRerank(queries, originals, { k: 10, candidates: 50 });
```

`originals` must hold every vector in the index, with row `i` being the vector added as
id `i`. `candidates` defaults to `4 * k`; fetching 4-10x `k` gets you close to exact
top-k. The returned scores are exact, in the index's metric.

### Scoring chosen ids

When something else has already picked the candidates (a metadata filter, another
index), score just those instead of scanning everything:

```ts
const scores = index.scoreIds(query, [17, 4, 902]);   // Float32Array, same order as the ids
```

Each score is what `search` would have reported for that id.

## Calibration

`index.calibrate(sample)` fits the corpus centre, and has to come before any `add`. Each
vector is then stored relative to the centre, and the centre's share of every score is
computed exactly, for 2 more bytes per vector. Whether it helps is predictable from one
number: the norm of the mean of the unit-normalized vectors. At 0.65 and above, where real
embeddings sit, it gained 9 to 22 points of recall@10 at bits 2 and 3; at 0.12, about a
point.

If you don't know which side of that line your data falls, let the library measure:

```ts
const { centroidNorm, applied } = index.calibrate(sample, { auto: true });
```

It calibrates only when the norm is at least 0.3, and tells you the norm and whether it
went ahead.

## Partitioning

`index.partition(sample, { lists })` splits the index into lists by k-means, before any
`add`. Each vector goes to the list whose centroid is nearest its direction and is stored
relative to that centroid. A search ranks the centroids against each query and scans only
the nearest few lists:

```ts
index.partition(sample, { lists: 1000, threads: 8 }); // about sqrt(n) lists
index.add(vectors, { threads: 8 });
const result = index.search(queries, { k: 10, probes: 64 });
index.probes = 64; // or set it once
```

`probes` defaults to an eighth of the lists. More finds more of the true neighbours and
costs proportionally more time.
Storing each vector against a nearby centroid also makes its estimate more accurate than
against the one corpus centre that calibration fits, so a partitioned index can beat a
calibrated flat one on recall as well as speed. On 100k Cohere embeddings at bits 2,
scanning 6% of the lists found R@10 0.775 at 11k queries a second on one core, against
0.750 at 1.8k for the calibrated flat index.

Partitioning takes the place of calibration, costs 4 more bytes per vector (for its id)
plus `lists × dim × 4` bytes of centroids, and can be done once, before any vector is
added. The sample needs at least `lists` rows; 256 per list is all it uses.
A partitioned index cannot be saved yet, and `scoreIds` is not available on it.

## Saving and loading

```ts
index.save('vectors.zq');
const again = Index.load('vectors.zq');

const bytes = index.serialize();            // Buffer
const copy = Index.deserialize(bytes);      // also takes a Uint8Array or ArrayBuffer
```

The format is versioned, checksummed and the same on every platform. A loaded index
searches, and accepts new vectors, exactly as the saved one did. Anything that isn't an
intact index from a compatible version throws a `ZquantError` with code
`ZQ_ERR_FORMAT`; damaged bytes never crash the process.

A loaded index knows how it was built. `index.settings` gives the `dim`, `bits`,
`metric`, `seed` and `compact` it was saved with, in a form `new Index()` accepts, and
`index.calibrated` says whether it was calibrated:

```ts
const twin = new Index(again.settings);     // encodes exactly as `again` does
```

## Codec: storing vectors rather than searching them

An index answers "which stored vectors best match this query". Sometimes you want
something else: keep vectors compactly and get them back, as in a KV cache. That's what
`Codec` is for.

```ts
import { Codec } from 'zquant';

const codec = new Codec({ dim: 128, bits: 4 });   // bits 1..6, default 4
const { codes, norms } = codec.encode(vectors);    // Uint8Array, Float32Array
const restored = codec.decode(codes, norms);       // Float32Array, n * dim
codec.close();
```

`codec.codeBytes` is the packed size of one vector's code. Each vector also needs its
norm (one float32), and decoding needs both. Unlike the index, the codec spends every
bit on the code, so `bits: 1` is allowed.

## Errors

Failures inside the library throw a `ZquantError`, with `status` (the number) and `code`
(its name, such as `ZQ_ERR_INVALID` or `ZQ_ERR_FORMAT`). `Status` has the numbers if you
want to compare. Mistakes the binding can catch itself, like a wrong `dim` or a row of
the wrong length, throw a `RangeError` or `TypeError` with the details.

## Cleaning up

`close()` frees the native memory right away, and it's safe to call more than once.
Using an index or codec after closing it throws an error instead of touching freed
memory.

If you forget to close something, it's freed when the garbage collector reclaims it. But
when that happens is up to the engine, and the collector doesn't see native memory, so
code that makes many indexes should close them.
