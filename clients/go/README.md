# zquant (Go)

```go
index, err := zquant.New(zquant.Config{Dim: 256, Bits: 5})
defer index.Close()

index.Add(vectors)                          // len(vectors) == n*Dim, float32
res, err := index.Search(queries, 10, 8)    // k=10, 8 threads
ids, scores := res.At(0)
```

Vectors are `[]float32`, row-major: n vectors of `Dim` floats each, back to back. Ids are
assigned in the order you add vectors, starting at 0.

## Install

The package links against `libzquant`. There are two ways to point it there.

**Installed** (the default build):

```sh
zig build install --prefix /usr/local
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig go test ./...
```

**Repository-local**, for working on zquant itself:

```sh
zig build lib
go test -tags repolocal ./...
```

The repo-local build sets an rpath, so the test binary finds the shared library without
`LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH`.

The module lives in a subdirectory of the repository, so Go expects its release tags to
carry that path: releases are tagged `clients/go/vX.Y.Z`, and you fetch one with

```sh
go get github.com/jeffhajewski/zquant/clients/go@vX.Y.Z
```

### CPU support

Release builds of the library use instructions beyond the architecture baseline, SSSE3
on x86_64 and the dot-product extension on arm64, because search is 6-10x slower
without them. Go links the library when you build, so there is no portable copy to fall
back to at run time. Instead, on a CPU that lacks them, `New`, `NewCodec` and
`FromBytes` return an error matching `zquant.ErrCPU`, naming what the library was built
for, rather than crashing. The fix is to link a portable build.

You can check up front, and print what you linked:

```go
if !zquant.CPUSupported() {
    log.Fatalf("libzquant needs a %s CPU", zquant.BuildCPU())
}
log.Println(zquant.Version(), zquant.BuildCPU(), zquant.BuildKernels())
```

## Configuration

| field | meaning |
|---|---|
| `Dim` | vector length, 1 to 65536 |
| `Bits` | bits per coordinate, 2 to 6; zero means 5 |
| `Metric` | `InnerProduct` (the default), `Cosine` or `L2` |
| `Seed` | picks the random rotation; zero means a fixed default |
| `Expanded` | store unpacked int8 codes: about twice the memory, a little faster |

Out-of-range values are rejected by `New` with an error that says which field is wrong.

## Searching

`Search` is the easy form. For more control, create a `Searcher`, which holds the
scratch space a search needs, and pass it to the other search calls:

```go
s, _ := index.NewSearcher(50, 1)     // fetch 50 candidates per query, on 1 thread
defer s.Close()

res, _ := index.SearchWith(s, queries)
```

A searcher belongs to the index that made it, and works only with that index.

**Reranking.** Quantized scores are estimates, so the true top 10 is not always the
estimated top 10. If you still have the original vectors, `SearchRerank` fetches the
searcher's `K` candidates, rescores them exactly against the originals, and keeps the
best `k`. Fetching 4-10x more than you keep gets close to exact results; in the tests,
recall@10 on a 3-bit index went from 0.58 to 0.96.

```go
res, _ := index.SearchRerank(s, queries, originals, 10)   // originals: row i is id i
```

**Scoring chosen ids.** When something else has already picked the candidates, a
metadata filter or another index, `ScoreIDs` scores just those against one query,
with the same estimate a search would give:

```go
scores, _ := index.ScoreIDs(s, query, []uint32{4, 17, 230})
```

## Adding faster

`AddParallel(vectors, threads)` encodes on several threads. The resulting index is
byte-for-byte the same as after `Add`. Small batches, below a few hundred vectors per
thread, are encoded on one thread anyway, since starting threads costs more than it
saves.

## Saving and loading

```go
index.Save("vectors.zq")
index, err := zquant.Load("vectors.zq")
```

`Index` also implements `encoding.BinaryMarshaler`, and `zquant.FromBytes` reads the
result back, for when you store the bytes yourself. The format is versioned,
checksummed and the same on every platform. A loaded index searches, and accepts new
vectors, exactly as the saved one did. A damaged or truncated file, or one written by an
incompatible version, gives an error matching `zquant.ErrFormat` instead of loading
wrong.

## Calibration

Calibration fits a per-coordinate shift and scale, and must happen before the first
`Add`. Whether it helps is predictable from the norm of the mean of the unit-normalized
vectors: above about 0.3 expect a real gain, below it expect none. It is not free; on
low-rank zero-mean data it has cost recall.

If you don't know which side of that line your data falls on, let the library measure:

```go
norm, applied, err := index.CalibrateAuto(sample)
```

It calibrates only when the norm is at least 0.3, and tells you the norm and what it
did. `Calibrate(sample)` calibrates unconditionally.

## Codec

An index answers "which stored vectors best match this query?" Sometimes you want
something else: store these vectors compactly and give them back, as in a KV cache.
That is what `Codec` is for.

```go
codec, _ := zquant.NewCodec(zquant.CodecConfig{Dim: 128, Bits: 4})
defer codec.Close()

codes, norms, _ := codec.Encode(vectors)    // CodeBytes() bytes + one float32 per vector
restored, _ := codec.Decode(codes, norms)
```

Codec bits run from 1 to 6 (zero means 5). The codec spends every bit on the codebook,
where the index reserves one, so the same number buys more accuracy here. Encoder and
decoder need the same `Dim`, `Bits` and `Seed`.

## Errors

Every error from the native library, and every argument the binding rejects before
calling it, is a `*zquant.Error` carrying a status code. Check for a kind of failure
with `errors.Is`:

| sentinel | meaning |
|---|---|
| `ErrInvalid` | a bad argument, including any NaN or infinity in the input |
| `ErrState` | not valid right now, such as calibrating after `Add` or searching an empty index |
| `ErrUnsupported` | a configuration the library can't build |
| `ErrAlloc` | out of memory |
| `ErrCPU` | this CPU can't run the linked library |
| `ErrFormat` | bytes that aren't an intact index from a compatible version |
| `ErrClosed` | the index, searcher or codec has been closed |

## Concurrency

Everything here is safe to call from multiple goroutines. What differs is what runs in
parallel and what waits:

- Searches on one index run in parallel, provided each goroutine uses its own
  `Searcher`. Goroutines that share a searcher take turns. `Search` keeps one cached
  searcher per `(k, threads)`, so concurrent `Search` calls with the same arguments
  take turns too; for throughput, give each goroutine its own searcher and use
  `SearchWith`.
- `Add`, `AddParallel`, `Calibrate` and `CalibrateAuto` change the index, so each waits
  for searches in flight to finish and holds new ones off until it is done.
- A `Codec` handles one call at a time. Create one per goroutine to encode in parallel.
- `Close` waits for calls in flight, can be called more than once, and every call after
  it returns an error matching `ErrClosed`. Closing an index also closes the searchers
  `Search` cached for it, but not the ones you made with `NewSearcher`; close those
  yourself.

Anything you forget to close is released by a finalizer eventually, but native memory
is invisible to the garbage collector, so don't rely on that.

## Measured

Through this binding, 100k × 256 vectors at `Bits: 5`, on an Apple M5:

| | |
|---|---|
| storage | 132 B/vector — 13 MB against 102 MB as float32 |
| build | 227,000 vectors/s |
| search | 5,655 QPS single-threaded, 28,127 across ten threads |

The Python and TypeScript clients report 28,927 and 27,951 on the same shape, and the
native benchmark about 27,000, so no binding costs anything measurable. `go test -bench
Search -tags repolocal` reruns the Go numbers.
