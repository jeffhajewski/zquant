# zquant

zquant searches compressed vectors. It stores a 768-dimensional embedding in 100 to 400
bytes instead of 3,072, and searches those bytes directly, without decompressing them. At
the same size it finds about as many of the true nearest neighbours as RaBitQ, the
strongest quantizer in FAISS, usually a point or so more, and far more than product
quantization.

It's a small C library with no dependencies, written in Zig, and the Python, Go and
TypeScript bindings all call the same code. The quantizer needs no training, so an index
is ready to use as soon as you create it.

```python
import zquant

index = zquant.Index(dim=768, bits=3)
index.calibrate(sample)              # optional: a few hundred vectors to find the centre
index.add(vectors, threads=8)

ids, scores = index.search(queries, k=10)
```

## A million embeddings

Cohere's multilingual embeddings, a million of them at 768 dimensions, take 3.1 GB as
float32. Here is what zquant does with them on one core of an Apple M5, using the Python
client. Recall@10 is the share of the true ten nearest neighbours that come back in the
top ten.

| index | memory | recall@10 | queries/s |
|---|---|---|---|
| bits=2, flat | 102 MB | 0.779 | 382 |
| bits=2, 1000 lists, 31 searched | 112 MB | 0.780 | 5,409 |
| bits=2, 1000 lists, 125 searched | 112 MB | 0.803 | 1,850 |
| bits=3, 1000 lists, 125 searched | 207 MB | 0.897 | 855 |
| bits=5, 1000 lists, 250 searched | 399 MB | 0.967 | 563 |

A flat index compares every query with every vector. A partitioned one splits the vectors
into lists by k-means and searches only the lists nearest each query. Searching 31 of the
1000 lists finds what the flat index finds, fourteen times faster. Searching more finds
*more* than the flat index does, because each vector is stored relative to its own list's
centre, which describes it better than the centre of the whole corpus.

If you need the top ten to be close to exact, keep the original vectors somewhere and
rerank: fetch a few more candidates from the compressed index and rescore them exactly.
In a separate run, reranking 40 candidates took bits=2 with 62 lists searched from 0.795
to 0.966, for 8% of the throughput, and with 125 lists to 0.983. The compressed index does
the searching; the full vectors are only read for the 40 candidates. Once you rerank,
bits=3 does no better than bits=2: what's left to find is in lists that weren't searched,
so the cheapest width is enough.

```python
index = zquant.Index(dim=768, bits=2)
index.partition(sample, lists=1000, threads=8)   # k-means; about sqrt(n) lists
index.add(vectors, threads=8)
ids, scores = index.search_rerank(queries, vectors, k=10, candidates=40, probes=62)
```

Building the partitioned index took 24 seconds on 8 threads (16 for k-means, 8 to encode
the vectors). A flat index takes 1.5 seconds.

## Why it works

Most quantizers learn their codes from your data. Product quantization, for example, runs
k-means on slices of the vectors, and the codebooks fit the data they were trained on.
zquant uses [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian,
Mirrokni, ICLR 2026) instead, which starts by applying a random rotation to every vector.
After a random rotation, each coordinate of a unit vector follows the same known
distribution, whatever the data looked like. So a single codebook, worked out once for
that distribution, is close to optimal for every coordinate of every dataset, and encoding
needs no training at all.

Two additions make it work well on real embeddings, which tend to sit in a narrow cone
rather than spread around the origin:

1. **Centring.** `calibrate` finds the corpus centre from a sample. Each vector is stored
   relative to it, and the centre's share of every score is computed exactly rather than
   estimated. This is [RaBitQ's](https://arxiv.org/abs/2405.12497) treatment of the
   centre, and on Cohere and OpenAI embeddings it's worth 9 to 22 points of recall at 1 to
   2 bits per dimension.
2. **Partitioning.** `partition` takes the same idea further. With k-means lists, each
   vector is stored relative to its own list's centre, and a search only visits the
   nearest lists.

Each vector costs its codes plus two or three 16-bit numbers: its length, a correction
that makes its score exact along its own direction, and an offset for the centre. The scan
scores codes against the query with SIMD table lookups (the dot-product instructions on
Arm, AVX2 or SSSE3 on x86) and reads each block of codes once for four queries at a time.

`bits` follows the paper's naming, where one bit went to a correction sketch. zquant
replaced that sketch with the per-vector correction, so `bits=b` stores b−1 bits per
dimension: `bits=2` is one bit per dimension, 102 bytes at d=768.

## How it compares

All against FAISS 1.16, on the same machine, one thread, 1000 queries, from one run per
table. FAISS's fast-scan indexes are its speed-tuned versions.
[docs/comparison.md](docs/comparison.md) has every row, more corpora, and how each was
measured.

**Flat**, on 100k Cohere embeddings (768 dimensions), k=100. Recall@10, then queries per
second:

| | 1 bit per dimension | 2 bits | 4 bits |
|---|---|---|---|
| zquant | **0.749** / 3,210 | **0.868** / 1,803 | **0.960** / 1,834 |
| FAISS RaBitQ, fast-scan | 0.739 / 2,502 | 0.861 / 1,467 | 0.952 / 1,160 |
| FAISS PQ, fast-scan | 0.540 / 3,514 | 0.719 / 1,847 | 0.895 / 906 |
| FAISS PQ | 0.582 / 510 | 0.753 / 177 | 0.917 / 61 |

zquant finds the most at every width, in 2 to 14 bytes per vector less than RaBitQ, and
it's faster than RaBitQ's fast-scan index at every width. At one bit it scores sixteen
vectors at a time by table lookup, as FAISS's fast-scan indexes do; only PQ's is faster
there, by 9%, and it finds 21 points fewer. At two bits PQ's is a little faster again,
finding 15 points fewer. At four bits zquant is the fastest here. On OpenAI's
1536-dimensional embeddings the picture is the same, except that at one bit RaBitQ's
fast-scan also finds slightly more (0.795 against 0.792).

**Partitioned**, same corpus, 316 lists in both systems:

| bits per dimension | lists searched | zquant | FAISS IVF-RaBitQ, fast-scan |
|---|---|---|---|
| 1 | 4 | **0.687 / 39,457** | 0.678 / 38,192 |
| 1 | 39 | **0.786 / 13,383** | 0.772 / 4,144 |
| 2 | 39 | **0.881 / 7,794** | 0.867 / 3,580 |
| 4 | 39 | **0.953 / 7,818** | 0.943 / 3,416 |

Searching 6% of the lists or more, zquant finds 0.7 to 1.4 points more and runs 1.8 to 3.2
times as fast. At the fewest lists the two are level: 3% faster here, 2% slower on OpenAI.
zquant regroups each batch of queries by list, so a list is read once for all the queries
that want it.

### Where it falls short

- **Against PQ's fast-scan index at one and two bits.** It's 2 to 31% faster than zquant
  there, though it finds 11 to 32 points fewer. zquant's one-bit scan is now table lookup
  like FAISS's; what's left is the work per candidate after the scan.
- **Small dimensions at tiny budgets.** At d=128 and d=256, below about 25 B per vector on
  SIFT and 100 B on nytimes, FAISS's product quantization finds more.
- **No graph index.** zquant scans codes, in full or by list. It hasn't been compared with
  HNSW or other graph indexes, which are the usual choice when memory is plentiful and
  latency matters most.
- **One machine.** Nearly everything was measured on an Apple M5. FAISS's fast-scan
  kernels were written for x86 and may do better there. The only x86 numbers are from a
  GitHub runner (AMD Zen 3), where the AVX2 kernels run 1.75 times as fast as the SSSE3
  ones.
- **KV caches.** TurboQuant was also proposed for compressing a transformer's key-value
  cache. Measured against KIVI there, zquant loses at the same memory, and badly on
  Qwen2.5, whose keys carry a bias. [docs/kv-cache.md](docs/kv-cache.md) has the details.

## Using it

| language | binding | |
|---|---|---|
| Python | ctypes, with numpy arrays | [clients/python](clients/python) |
| Go | cgo | [clients/go](clients/go) |
| TypeScript and Node | koffi | [clients/typescript](clients/typescript) |
| C | [include/zquant.h](include/zquant.h) | |
| Zig | [src/root.zig](src/root.zig) | |

Every binding has the same features:

- flat or partitioned search, with the number of lists to search set per query or per
  index;
- `search_rerank`, which over-fetches and rescores against your original vectors;
- `score_ids`, which scores ids you choose, such as candidates from a filter;
- save and load, in a versioned, checksummed format that's byte-for-byte the same on every
  platform;
- `Codec`, which encodes and decodes vectors without an index;
- threads: many searches can run at once, adding vectors is safe against searches in
  flight, and multi-threaded encoding produces the same bytes as single-threaded.

The bindings cost nothing you can measure. On 100k × 256 vectors, Python, Go and
TypeScript all search at 5,600 queries a second on one thread, the same as C, and encode
2.4 million vectors a second on ten. Each one replays a shared
[conformance fixture](tests/golden/conformance.json) and has to reproduce its bytes
exactly.

## Building

You need Zig 0.16.0.

```sh
zig build test
zig build lib          # the C library, header and pkg-config file, in zig-out/
zig build quickbench   # a one-minute benchmark with its own data, no downloads
```

No release has been published yet, so for now each client builds against `zig-out/lib`;
its README says how. The release workflow builds Python wheels, an npm package and C
archives for Linux and macOS on x86_64 and arm64, and for Windows on x86_64.

The scan kernels are compiled for a CPU level. Release libraries target SSSE3 on x86_64
and the dot-product extension on arm64, and the x86_64 ones also carry AVX2 kernels that
they switch to at run time when the CPU has them. On an older CPU, creating an index
returns `ZQ_ERR_CPU` rather than crashing, and the Python and TypeScript packages fall
back to a portable build that is correct but several times slower.
`zig build lib` with no options builds for the machine you're on.

On macOS with the Xcode 26 SDK, linking fails with `undefined symbol: _getcwd` and
similar, because the SDK leaves out the plain `arm64-macos` target. Use the Command Line
Tools SDK instead:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools zig build test
```

## Further reading

- [docs/comparison.md](docs/comparison.md): every measurement against FAISS and turbovec.
- [docs/notes.md](docs/notes.md): the engineering log, including each time a result
  here turned out to be wrong, and why.
- [docs/DESIGN.md](docs/DESIGN.md): the design, and where it departs from the paper.
- [docs/kv-cache.md](docs/kv-cache.md): KV-cache compression, measured against KIVI.
