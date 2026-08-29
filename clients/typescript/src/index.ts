/**
 * zquant — TurboQuant vector quantization.
 *
 * ```ts
 * import { Index } from 'zquant';
 *
 * const index = new Index({ dim: 256, bits: 5 });
 * index.add(vectors);                          // Float32Array, n * dim
 * const { ids, scores } = index.search(queries, { k: 10, threads: 8 });
 * index.close();
 * ```
 *
 * Vectors are float32, row-major, one per row. `number[][]` is accepted and flattened,
 * which costs a copy; pass a `Float32Array` to avoid it.
 */
import { native, OK } from './native.js';

export type Metric = 'ip' | 'inner_product' | 'cosine' | 'l2';

const METRICS: Record<Metric, number> = {
  ip: 0,
  inner_product: 0,
  cosine: 1,
  l2: 2,
};

export class ZquantError extends Error {
  readonly status: number;
  constructor(status: number, context: string) {
    super(`${context}: ${native.statusString(status)} (${status})`);
    this.name = 'ZquantError';
    this.status = status;
  }
}

function check(status: number, context: string): void {
  if (status !== OK) throw new ZquantError(status, context);
}

export function version(): string {
  return native.version();
}

export interface IndexOptions {
  dim: number;
  /** Total bit budget per coordinate, 2..6. The scalar codebook uses bits-1. */
  bits?: number;
  metric?: Metric;
  seed?: number | bigint;
  /** Packed codes (default) or dequantized int8: ~2x memory for some throughput. */
  compact?: boolean;
}

export interface SearchOptions {
  k?: number;
  /** >1 splits the batch across workers; only pays for batches at least that large. */
  threads?: number;
}

export interface SearchResult {
  /** `nq * k` ids, query-major, best first. */
  ids: Uint32Array;
  /** `nq * k` scores, aligned with `ids`. */
  scores: Float32Array;
  k: number;
}

/** Rows of `dim` floats each: a flat Float32Array, or an array of arrays. */
export type Vectors = Float32Array | number[][] | number[];

function asRows(x: Vectors, dim: number, name: string): Float32Array {
  if (x instanceof Float32Array) {
    if (x.length % dim !== 0) {
      throw new Error(`${name} length ${x.length} is not a multiple of dim ${dim}`);
    }
    return x;
  }
  if (Array.isArray(x) && x.length > 0 && Array.isArray(x[0])) {
    const rows = x as number[][];
    const out = new Float32Array(rows.length * dim);
    rows.forEach((row, i) => {
      if (row.length !== dim) {
        throw new Error(`${name} row ${i} has dim ${row.length}, index expects ${dim}`);
      }
      out.set(row, i * dim);
    });
    return out;
  }
  const flat = x as number[];
  if (flat.length % dim !== 0) {
    throw new Error(`${name} length ${flat.length} is not a multiple of dim ${dim}`);
  }
  return Float32Array.from(flat);
}

/**
 * A flat (exhaustive) quantized index.
 *
 * Every vector is scanned for every query — no graph, no partitioning — so recall is
 * bounded only by quantization and cost grows linearly with the corpus.
 */
export class Index {
  #handle: unknown;
  #dim: number;
  #searchers = new Map<string, unknown>();

  constructor(options: IndexOptions) {
    const metric = options.metric ?? 'ip';
    if (!(metric in METRICS)) {
      throw new Error(`metric must be one of ${Object.keys(METRICS).join(', ')}, got ${metric}`);
    }
    const out = [null];
    check(
      native.indexCreate(
        {
          dim: options.dim,
          bits: options.bits ?? 5,
          metric: METRICS[metric],
          seed: BigInt(options.seed ?? 0x5eed),
          compact: (options.compact ?? true) ? 1 : 0,
        },
        out,
      ),
      'indexCreate',
    );
    this.#handle = out[0];
    this.#dim = options.dim;
  }

  get dim(): number {
    return this.#dim;
  }

  get size(): number {
    return Number(native.indexCount(this.#handle));
  }

  get bytesPerVector(): number {
    return Number(native.indexBytesPerVector(this.#handle));
  }

  /**
   * Fit per-coordinate shift and scale from a sample. Must precede `add`.
   *
   * Worth enabling when the corpus centroid sits away from the origin — the norm of the
   * mean of the unit-normalized vectors. Above about 0.3 expect a real gain, below it
   * expect none. It is not free: on low-rank zero-mean data it has cost recall.
   */
  calibrate(sample: Vectors): void {
    const a = asRows(sample, this.#dim, 'sample');
    check(native.indexCalibrate(this.#handle, a, a.length / this.#dim), 'calibrate');
  }

  add(vectors: Vectors): void {
    const a = asRows(vectors, this.#dim, 'vectors');
    check(native.indexAdd(this.#handle, a, a.length / this.#dim), 'add');
  }

  #searcher(k: number, threads: number): unknown {
    const key = `${k}:${threads}`;
    let s = this.#searchers.get(key);
    if (s === undefined) {
      const out = [null];
      check(native.searcherCreate(this.#handle, 32, k, threads, out), 'searcherCreate');
      s = out[0];
      this.#searchers.set(key, s);
    }
    return s;
  }

  /**
   * Returns `nq * k` ids and scores, query-major, best first. Where the index holds
   * fewer than `k` vectors the tail is `0xFFFFFFFF` / `-Infinity`.
   */
  search(queries: Vectors, options: SearchOptions = {}): SearchResult {
    const k = options.k ?? 10;
    const threads = options.threads ?? 1;
    if (k < 1) throw new Error('k must be at least 1');
    if (this.size === 0) throw new ZquantError(-3, 'search on an empty index');

    const q = asRows(queries, this.#dim, 'queries');
    const nq = q.length / this.#dim;
    const searcher = this.#searcher(k, threads);
    const capacity = Number(native.searcherCapacity(searcher));

    const ids = new Uint32Array(nq * k);
    const scores = new Float32Array(nq * k);
    // The native call takes at most `capacity` queries; chunk here rather than making
    // that the caller's problem.
    for (let start = 0; start < nq; start += capacity) {
      const n = Math.min(capacity, nq - start);
      const idsChunk = new Uint32Array(n * k);
      const scoresChunk = new Float32Array(n * k);
      check(
        native.search(
          this.#handle,
          searcher,
          q.subarray(start * this.#dim, (start + n) * this.#dim),
          n,
          idsChunk,
          scoresChunk,
        ),
        'search',
      );
      ids.set(idsChunk, start * k);
      scores.set(scoresChunk, start * k);
    }
    return { ids, scores, k };
  }

  close(): void {
    for (const s of this.#searchers.values()) native.searcherFree(s);
    this.#searchers.clear();
    if (this.#handle) {
      native.indexFree(this.#handle);
      this.#handle = null;
    }
  }
}
