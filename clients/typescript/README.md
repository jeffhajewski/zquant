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

From a tarball built with the native library inside it:

```sh
npm install ./zquant-0.1.0.tgz
```

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

A **flat** index: every vector is scanned for every query. No graph, no partitioning, so
recall is limited only by quantization and cost grows linearly with the corpus.

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

`index.calibrate(sample)` fits a per-coordinate shift and scale, and has to come before
any `add`. Whether it helps is predictable from one number: the norm of the mean of the
unit-normalized vectors. Above about 0.3 expect a real gain (+11 points of recall was
measured at 0.65); below it expect none, and on low-rank zero-mean data it has cost
recall.

If you don't know which side of that line your data falls, let the library measure:

```ts
const { centroidNorm, applied } = index.calibrate(sample, { auto: true });
```

It calibrates only when the norm is at least 0.3, and tells you the norm and whether it
went ahead.

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
