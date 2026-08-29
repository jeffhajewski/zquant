# zquant (Go)

```go
index, err := zquant.New(zquant.Config{Dim: 256, Bits: 5})
defer index.Close()

index.Add(vectors)                          // len(vectors) == n*Dim, float32
res, err := index.Search(queries, 10, 8)    // k=10, 8 threads
ids, scores := res.At(0)
```

## Install

The package links against `libzquant`. Two supported ways:

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

## Concurrency

An `Index` is safe for concurrent `Search` calls, but `Search` serializes them behind a
mutex, because the scratch space a search needs cannot be shared between goroutines. For
concurrent throughput give each goroutine its own searcher:

```go
s, _ := index.NewSearcher(10, 1)
defer s.Close()
res, _ := index.SearchWith(s, queries)
```

`Add` and `Calibrate` are not safe to run concurrently with anything else.

## Measured

Through this binding, 100k × 256 vectors at `Bits: 5`, on an Apple M5:

| | |
|---|---|
| storage | 132 B/vector — 13 MB against 102 MB as float32 |
| build | 227,000 vectors/s |
| search | 5,655 QPS single-threaded, 28,127 across ten threads |

The Python and TypeScript clients report 28,927 and 27,951 on the same shape, and the
native benchmark about 27,000, so no binding costs anything measurable.

## Calibration

`Calibrate` fits a per-coordinate shift and scale and must precede `Add`. Whether it
helps is predictable from the norm of the mean of the unit-normalized vectors: above
about 0.3 expect a real gain, below it expect none. It is not free — on low-rank
zero-mean data it has cost recall — which is why it is opt-in.
