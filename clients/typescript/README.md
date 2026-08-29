# zquant (TypeScript / Node)

```ts
import { Index } from 'zquant';

const index = new Index({ dim: 256, bits: 5 });
index.add(vectors);                                    // Float32Array, n * dim
const { ids, scores } = index.search(queries, { k: 10, threads: 8 });
index.close();
```

## Install

Needs `libzquant` beside the package. During development:

```sh
zig build lib          # at the repository root
cd clients/typescript && npm install && npm run build && npm test
```

The binding looks beside the compiled output, then walks up for a `zig-out/lib` build
tree, and honours `ZQUANT_LIBRARY` if you want to point it elsewhere.

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

## Calibration

`index.calibrate(sample)` fits a per-coordinate shift and scale, and must precede `add`.
Whether it is worth anything is predictable from one statistic — the norm of the mean of
the unit-normalized vectors. Above about 0.3 expect a real gain (+11 points of recall was
measured at 0.65); below it expect none. It is not free: on low-rank zero-mean data it has
cost recall, which is why it is opt-in.

## Notes

- Vectors are float32 row-major. `number[][]` is accepted and flattened, at the cost of a
  copy; pass a `Float32Array` to avoid it.
- `close()` releases the native index. There is no finalizer, so an index that is never
  closed leaks until the process exits.
- `threads > 1` splits one batch across workers and only pays for batches at least that
  large. `add` and `calibrate` are not concurrent-safe.
