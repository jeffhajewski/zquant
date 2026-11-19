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
 * Vectors are float32, row-major, one per row. `number[][]` and `Float32Array[]` are
 * accepted and flattened, which costs a copy; pass one flat `Float32Array` to avoid it.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { native, OK } from './native.js';
import {
  type CodecState,
  type IndexState,
  addSearcher,
  adoptCodec,
  adoptIndex,
  dropSearchers,
  releaseCodec,
  releaseIndex,
} from './handles.js';
import { type Vector, type Vectors, asRows } from './rows.js';

export type { Vector, Vectors };

/** Largest dimension the library accepts. */
export const MAX_DIM = 65536;
/** Largest worker count for a search or a parallel add. */
export const MAX_THREADS = 1024;

/** The library's status codes. Every non-OK one surfaces as a {@link ZquantError}. */
export const Status = {
  OK: 0,
  ERR_ALLOC: -1,
  ERR_INVALID: -2,
  ERR_STATE: -3,
  ERR_UNSUPPORTED: -4,
  ERR_CPU: -5,
  ERR_FORMAT: -6,
} as const;

const STATUS_INFO: Record<number, [code: string, message: string]> = {
  [-1]: ['ZQ_ERR_ALLOC', 'out of memory'],
  [-2]: ['ZQ_ERR_INVALID', 'invalid argument'],
  [-3]: ['ZQ_ERR_STATE', 'operation not valid in this state'],
  [-4]: ['ZQ_ERR_UNSUPPORTED', 'unsupported configuration'],
  [-5]: ['ZQ_ERR_CPU', 'this CPU lacks instructions this build of the library needs'],
  [-6]: ['ZQ_ERR_FORMAT', 'not a zquant index, damaged, or saved by an incompatible version'],
};

/** A call into the library failed. `status` is the raw code, `code` its name. */
export class ZquantError extends Error {
  readonly status: number;
  readonly code: string;
  constructor(status: number, context: string, hint?: string) {
    const [code, message] = STATUS_INFO[status] ?? ['ZQ_ERR_UNKNOWN', `unknown status ${status}`];
    let text = `${context}: ${message} (${code})`;
    if (status === Status.ERR_CPU) {
      try {
        const n = native();
        text += `; ${n.path} was built for ${n.buildCpu()}`;
      } catch {
        // The library did not load, so there is nothing more to say about it.
      }
    }
    if (hint) text += `. ${hint}`;
    super(text);
    this.name = 'ZquantError';
    this.status = status;
    this.code = code;
  }
}

function check(status: number, context: string, hint?: string): void {
  if (status !== OK) throw new ZquantError(status, context, hint);
}

export function version(): string {
  return native().version();
}

export interface BuildInfo {
  /** Library version, e.g. "0.1.0". */
  version: string;
  /** The CPU model the library was compiled for, e.g. "x86_64_v2" or "apple_m1". */
  cpu: string;
  /** Which scan kernels it uses. "scalar" means the portable fallback: correct, slower. */
  kernels: string;
  /** Absolute path of the library that was loaded. */
  libraryPath: string;
  /** True if this CPU could not run the release build and the portable one was loaded. */
  portable: boolean;
}

/** What was loaded and what it was built for. Loads the library if it is not yet loaded. */
export function buildInfo(): BuildInfo {
  const n = native();
  return {
    version: n.version(),
    cpu: n.buildCpu(),
    kernels: n.buildKernels(),
    libraryPath: n.path,
    portable: n.portable,
  };
}

export type Metric = 'ip' | 'inner_product' | 'cosine' | 'l2';

const METRICS: Record<Metric, number> = {
  ip: 0,
  inner_product: 0,
  cosine: 1,
  l2: 2,
};

/**
 * A seed for the random rotation. Seeds are 64-bit: pass a `bigint` for anything above
 * 2^53, since a `number` cannot hold such a value exactly.
 */
export type Seed = number | bigint;

const DEFAULT_SEED = 0x5eedn;
const U64_MAX = (1n << 64n) - 1n;

function toSeed(seed: Seed | undefined): bigint {
  if (seed === undefined) return DEFAULT_SEED;
  if (typeof seed === 'bigint') {
    if (seed < 0n || seed > U64_MAX) {
      throw new RangeError(`seed must be between 0 and 2^64 - 1, got ${seed}`);
    }
    return seed;
  }
  if (typeof seed === 'number') {
    if (!Number.isSafeInteger(seed) || seed < 0) {
      throw new RangeError(
        `seed must be a non-negative safe integer, got ${seed}; pass a bigint for seeds above 2^53`,
      );
    }
    return BigInt(seed);
  }
  throw new TypeError(`seed must be a number or a bigint, got ${typeof seed}`);
}

function checkInt(value: unknown, name: string, min: number, max: number, note = ''): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < min || value > max) {
    throw new RangeError(`${name} must be an integer from ${min} to ${max}${note}, got ${String(value)}`);
  }
  return value;
}

function checkDim(dim: unknown): number {
  return checkInt(dim, 'dim', 1, MAX_DIM);
}

function checkThreads(threads: unknown): number {
  return checkInt(threads, 'threads', 0, MAX_THREADS);
}

function checkProbes(probes: unknown): number {
  return checkInt(probes, 'probes', 1, 65536);
}

function checkK(k: unknown, name = 'k'): number {
  if (typeof k !== 'number' || !Number.isInteger(k) || k < 1) {
    throw new RangeError(`${name} must be at least 1, got ${String(k)}`);
  }
  return k;
}

const NON_FINITE_HINT = 'Every value must be finite: NaN and Infinity are refused';

export interface IndexOptions {
  /** Dimension of every vector, 1..65536. */
  dim: number;
  /**
   * Total bit budget per coordinate, 2..6 (default 5). The scalar codebook uses bits-1
   * and the last bit goes to a residual correction, which is why 1 is not allowed.
   */
  bits?: number;
  metric?: Metric;
  seed?: Seed;
  /** Packed codes (default) or dequantized int8: ~2x memory for some throughput. */
  compact?: boolean;
}

export interface AddOptions {
  /**
   * Encode on up to this many threads (0..1024). The index comes out byte-for-byte the
   * same as a single-threaded add. Below a few hundred vectors per thread the library
   * stays on the calling thread, since starting threads would cost more.
   */
  threads?: number;
}

export interface SearchOptions {
  k?: number;
  /** >1 splits the batch across workers; only pays for batches at least that large. */
  threads?: number;
  /**
   * Lists each query scans on a partitioned index, for this call: 1..65536, clamped to
   * the number of lists. Defaults to `index.probes`, or an eighth of the lists.
   */
  probes?: number;
}

export interface PartitionOptions {
  /** Number of lists, 1..65536. Around sqrt(n) suits a corpus of n vectors. */
  lists: number;
  /** k-means on up to this many threads (0..1024). */
  threads?: number;
}

export interface RerankOptions extends SearchOptions {
  /**
   * How many candidates to fetch per query before rescoring them exactly; at least k.
   * Defaults to 4k. Fetching 4-10x k and reranking recovers near-exact top-k.
   */
  candidates?: number;
}

export interface SearchResult {
  /** `nq * k` ids, query-major, best first. */
  ids: Uint32Array;
  /** `nq * k` scores, aligned with `ids`. */
  scores: Float32Array;
  k: number;
}

/** An index's settings, as read back from the library. */
export interface IndexSettings {
  dim: number;
  bits: number;
  metric: 'ip' | 'cosine' | 'l2';
  seed: bigint;
  compact: boolean;
}

const METRIC_NAMES = ['ip', 'cosine', 'l2'] as const;

export interface CalibrateOptions {
  /**
   * Measure the sample first and calibrate only if its centroid norm is at least 0.3,
   * where calibration measured a gain. The sensible choice when you do not know your data.
   */
  auto?: boolean;
}

export interface CalibrationResult {
  /** Norm of the mean of the unit-normalized sample vectors. */
  centroidNorm: number;
  /** Whether calibration was applied. */
  applied: boolean;
}

// Set by Index.deserialize just before it calls the constructor, so the constructor can
// adopt an existing handle without exposing a way to do that in the public signature.
let adopting: unknown = null;

/** Queries per native search call per worker. */
const BATCH = 32;

/**
 * A quantized index: flat (exhaustive) unless `partition` splits it into lists.
 *
 * Flat, every vector is scanned for every query, so recall is bounded only by
 * quantization and cost grows linearly with the corpus.
 *
 * Call `close()` when done. An index that is garbage collected without being closed is
 * freed then, but when that happens is up to the engine.
 */
export class Index {
  #state: IndexState;
  #dim: number;
  #settings: Readonly<IndexSettings> | null;

  constructor(options: IndexOptions) {
    let handle: unknown;
    if (adopting !== null) {
      handle = adopting;
      adopting = null;
    } else {
      if (typeof options !== 'object' || options === null) {
        throw new TypeError('Index needs an options object, e.g. new Index({ dim: 256 })');
      }
      const dim = checkDim(options.dim);
      const bits = checkInt(options.bits ?? 5, 'bits', 2, 6, ' for an index');
      const metric = options.metric ?? 'ip';
      if (!Object.prototype.hasOwnProperty.call(METRICS, metric)) {
        throw new RangeError(
          `metric must be one of ${Object.keys(METRICS).join(', ')}, got ${String(metric)}`,
        );
      }
      const seed = toSeed(options.seed);
      const out = [null];
      check(
        native().indexCreate(
          { dim, bits, metric: METRICS[metric], seed, compact: (options.compact ?? true) ? 1 : 0 },
          out,
        ),
        'Index',
      );
      handle = out[0];
    }
    this.#state = adoptIndex(this, handle);
    this.#dim = native().indexDim(handle);
    // Read back rather than remembered from the options, so a loaded index reports what
    // it was saved with. An index saved through the Zig API can carry options the C
    // configuration has no field for; it works, but cannot describe itself.
    const cfg: { dim?: number; bits?: number; metric?: number; seed?: number | bigint; compact?: number } = {};
    this.#settings =
      native().indexConfig(handle, cfg) === Status.OK
        ? Object.freeze({
            dim: cfg.dim as number,
            bits: cfg.bits as number,
            metric: METRIC_NAMES[cfg.metric as number],
            seed: BigInt(cfg.seed as number | bigint),
            compact: cfg.compact !== 0,
          })
        : null;
  }

  /** Rebuild an index from bytes produced by {@link Index.serialize}. */
  static deserialize(bytes: Uint8Array | ArrayBuffer): Index {
    const buf = bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : bytes;
    if (!(buf instanceof Uint8Array)) {
      throw new TypeError('deserialize takes a Uint8Array, Buffer or ArrayBuffer');
    }
    if (buf.length === 0) {
      throw new ZquantError(Status.ERR_FORMAT, 'deserialize', 'The input is empty');
    }
    const out = [null];
    check(native().indexDeserialize(buf, buf.length, out), 'deserialize');
    adopting = out[0];
    try {
      return new Index(undefined as unknown as IndexOptions);
    } finally {
      adopting = null;
    }
  }

  /** Load an index saved with {@link Index.save}. */
  static load(path: string): Index {
    return Index.deserialize(readFileSync(path));
  }

  #handle(): unknown {
    const h = this.#state.handle;
    if (h === null) throw new Error('this Index is closed');
    return h;
  }

  get dim(): number {
    return this.#dim;
  }

  get size(): number {
    return Number(native().indexCount(this.#handle()));
  }

  get bytesPerVector(): number {
    return Number(native().indexBytesPerVector(this.#handle()));
  }

  /**
   * The settings the index was created with, or saved with if it was loaded. Passing
   * them to `new Index` gives an index that encodes identically, apart from calibration
   * (see {@link Index.calibrated}). Throws `ZQ_ERR_UNSUPPORTED` for an index saved
   * through the Zig API with options these settings cannot describe.
   */
  get settings(): Readonly<IndexSettings> {
    if (this.#settings === null) {
      throw new ZquantError(
        Status.ERR_UNSUPPORTED,
        'settings',
        'This index was saved through the Zig API with options the C configuration cannot describe',
      );
    }
    return this.#settings;
  }

  /** Whether `calibrate` has been applied, here or before the index was saved. */
  get calibrated(): boolean {
    return native().indexCalibrated(this.#handle()) !== 0;
  }

  /** True once `close()` has been called. */
  get closed(): boolean {
    return this.#state.handle === null;
  }

  /**
   * Fit the corpus centre from a sample. Must precede `add`.
   *
   * Each vector is then stored relative to the centre, and the centre's share of every
   * score is computed exactly, for 2 more bytes per vector. Worth it when the corpus sits
   * away from the origin — the norm of the mean of the unit-normalized vectors. At 0.65
   * and above it gained 9 to 22 points of recall@10 at bits 2 and 3; at 0.12, about a
   * point.
   *
   * With `{ auto: true }` the library measures that norm and calibrates only if it is at
   * least 0.3, and the result says what it found and whether it calibrated.
   */
  calibrate(sample: Vectors, options: CalibrateOptions & { auto: true }): CalibrationResult;
  calibrate(sample: Vectors, options?: CalibrateOptions): CalibrationResult | undefined;
  calibrate(sample: Vectors, options: CalibrateOptions = {}): CalibrationResult | undefined {
    const h = this.#handle();
    const a = asRows(sample, this.#dim, 'sample');
    const n = a.length / this.#dim;
    if (n === 0) throw new RangeError('calibrate needs at least one sample vector');
    const hint = this.size > 0 ? 'Calibrate before adding any vectors' : NON_FINITE_HINT;
    if (options.auto) {
      const norm = new Float32Array(1);
      const applied = new Int32Array(1);
      check(native().indexCalibrateAuto(h, a, n, norm, applied), 'calibrate', hint);
      return { centroidNorm: norm[0], applied: applied[0] !== 0 };
    }
    check(native().indexCalibrate(h, a, n), 'calibrate', hint);
    return undefined;
  }

  /**
   * Split the index into lists by k-means over a sample. Must precede `add`, and can be
   * done once.
   *
   * Each vector added afterwards goes to the list whose centroid is nearest its direction
   * and is stored relative to that centroid. A search then scans only the lists nearest
   * each query: `probes` of them, an eighth unless set. Storing vectors against nearby
   * centroids also makes their estimates more accurate, so recall rises even with every
   * list scanned. The sample needs at least `lists` rows; past 256 per list the rest are
   * not used. Partitioning takes the place of `calibrate`. A partitioned index cannot yet
   * be saved, and `scoreIds` is not available on it.
   */
  partition(sample: Vectors, options: PartitionOptions): void {
    const h = this.#handle();
    const lists = options.lists;
    if (!Number.isInteger(lists) || lists < 1 || lists > 65536) {
      throw new RangeError(`lists must be an integer in 1..65536, got ${lists}`);
    }
    const threads = checkThreads(options.threads ?? 1);
    const a = asRows(sample, this.#dim, 'sample');
    const n = a.length / this.#dim;
    check(native().indexPartition(h, a, n, lists, threads), 'partition', NON_FINITE_HINT);
    // Searchers made before partitioning have no lists to probe.
    dropSearchers(this.#state);
  }

  /** Lists in a partitioned index, or 0 for a flat one. */
  get lists(): number {
    return Number(native().indexLists(this.#handle()));
  }

  #probes: number | undefined = undefined;

  /**
   * Lists each query scans on a partitioned index when `search` is not given `probes`.
   * `undefined` means an eighth of the lists.
   */
  get probes(): number | undefined {
    return this.#probes;
  }

  set probes(value: number | undefined) {
    if (value !== undefined) checkProbes(value);
    this.#probes = value;
  }

  #applyProbes(searcher: unknown, probes: number | undefined): void {
    const lists = this.lists;
    if (probes !== undefined) {
      checkProbes(probes);
      if (lists === 0) {
        throw new ZquantError(Status.ERR_STATE, 'search', 'probes applies only to a partitioned index');
      }
    }
    if (lists === 0) return;
    const p = probes ?? this.#probes ?? Math.max(1, Math.floor(lists / 8));
    check(native().searcherSetProbes(searcher, p), 'search');
  }

  /** Append vectors. Ids are assigned in insertion order, starting at 0. */
  add(vectors: Vectors, options: AddOptions = {}): void {
    const h = this.#handle();
    const a = asRows(vectors, this.#dim, 'vectors');
    const n = a.length / this.#dim;
    if (options.threads === undefined) {
      check(native().indexAdd(h, a, n), 'add', NON_FINITE_HINT);
    } else {
      const threads = checkThreads(options.threads);
      check(native().indexAddParallel(h, a, n, threads), 'add', NON_FINITE_HINT);
    }
  }

  #searcher(k: number, threads: number): unknown {
    const key = `${k}:${threads}`;
    let s = this.#state.searchers.get(key);
    if (s === undefined) {
      const out = [null];
      check(native().searcherCreate(this.#handle(), BATCH, k, threads, out), 'searcher');
      s = out[0];
      addSearcher(this.#state, key, s);
    }
    return s;
  }

  /**
   * Returns `nq * k` ids and scores, query-major, best first. Where the index holds
   * fewer than `k` vectors the tail is `0xFFFFFFFF` / `-Infinity`, as it is on a
   * partitioned index where the lists scanned hold fewer than `k`.
   */
  search(queries: Vectors, options: SearchOptions = {}): SearchResult {
    const h = this.#handle();
    const k = checkK(options.k ?? 10);
    const threads = checkThreads(options.threads ?? 1);
    if (this.size === 0) throw new ZquantError(Status.ERR_STATE, 'search', 'The index is empty');

    const q = asRows(queries, this.#dim, 'queries');
    const searcher = this.#searcher(k, threads);
    this.#applyProbes(searcher, options.probes);
    return this.#chunked(q, k, searcher, (chunk, n, ids, scores) =>
      check(native().search(h, searcher, chunk, n, ids, scores), 'search', NON_FINITE_HINT),
    );
  }

  /**
   * Search, then rescore each query's candidates exactly against the original vectors
   * and keep the best `k`. `originals` must hold every vector in the index, row `i`
   * being the vector added as id `i`. Scores are exact, in the index's metric.
   */
  searchRerank(queries: Vectors, originals: Vectors, options: RerankOptions = {}): SearchResult {
    const h = this.#handle();
    const k = checkK(options.k ?? 10);
    const candidates = checkK(options.candidates ?? 4 * k, 'candidates');
    if (candidates < k) {
      throw new RangeError(`candidates (${candidates}) must be at least k (${k})`);
    }
    const threads = checkThreads(options.threads ?? 1);
    const size = this.size;
    if (size === 0) throw new ZquantError(Status.ERR_STATE, 'searchRerank', 'The index is empty');

    const q = asRows(queries, this.#dim, 'queries');
    const orig = asRows(originals, this.#dim, 'originals');
    const nOrig = orig.length / this.#dim;
    if (nOrig < size) {
      throw new RangeError(
        `originals holds ${nOrig} vectors but the index holds ${size}; ` +
          'row i must be the vector added as id i, for every id',
      );
    }
    const searcher = this.#searcher(candidates, threads);
    this.#applyProbes(searcher, options.probes);
    return this.#chunked(q, k, searcher, (chunk, n, ids, scores) =>
      check(
        native().searchRerank(h, searcher, chunk, n, orig, nOrig, k, ids, scores),
        'searchRerank',
        NON_FINITE_HINT,
      ),
    );
  }

  // The native calls take at most `capacity` queries; chunk here rather than making
  // that the caller's problem.
  #chunked(
    q: Float32Array,
    k: number,
    searcher: unknown,
    run: (chunk: Float32Array, n: number, ids: Uint32Array, scores: Float32Array) => void,
  ): SearchResult {
    const d = this.#dim;
    const nq = q.length / d;
    const capacity = Number(native().searcherCapacity(searcher));
    const ids = new Uint32Array(nq * k);
    const scores = new Float32Array(nq * k);
    for (let start = 0; start < nq; start += capacity) {
      const n = Math.min(capacity, nq - start);
      run(
        q.subarray(start * d, (start + n) * d),
        n,
        ids.subarray(start * k, (start + n) * k),
        scores.subarray(start * k, (start + n) * k),
      );
    }
    return { ids, scores, k };
  }

  /**
   * Score chosen stored vectors against one query: `scores[j]` is the estimate search
   * would report for `ids[j]`. For ranking a candidate set picked some other way, such
   * as a metadata filter, without scanning the whole index.
   */
  scoreIds(query: Vector, ids: Uint32Array | readonly number[]): Float32Array {
    const h = this.#handle();
    const q = asRows(query, this.#dim, 'query');
    if (q.length !== this.#dim) {
      throw new RangeError(`query must be one vector of ${this.#dim} floats, got ${q.length}`);
    }
    const size = this.size;
    const idArr = ids instanceof Uint32Array ? ids : new Uint32Array(ids.length);
    for (let j = 0; j < ids.length; j++) {
      const id = ids[j];
      if (!Number.isInteger(id) || id < 0 || id >= size) {
        throw new RangeError(`ids[${j}] is ${id}, but the index holds ids 0 to ${size - 1}`);
      }
      if (idArr !== ids) idArr[j] = id;
    }
    const scores = new Float32Array(idArr.length);
    if (idArr.length === 0) return scores;
    const searcher = this.#searcher(1, 1);
    check(native().scoreIds(h, searcher, q, idArr, idArr.length, scores), 'scoreIds', NON_FINITE_HINT);
    return scores;
  }

  /**
   * The index as bytes: configuration, calibration and codes, in a versioned,
   * checksummed, little-endian format that is the same on every platform.
   */
  serialize(): Buffer {
    const h = this.#handle();
    const size = Number(native().indexSerializedSize(h));
    const buf = Buffer.alloc(size);
    const written = [0];
    check(native().indexSerialize(h, buf, size, written), 'serialize');
    return buf.subarray(0, Number(written[0]));
  }

  /** Write {@link Index.serialize}'s bytes to a file. */
  save(path: string): void {
    writeFileSync(path, this.serialize());
  }

  /** Free the native index and its searchers. Safe to call more than once. */
  close(): void {
    releaseIndex(this.#state);
  }
}

export interface CodecOptions {
  /** Dimension of every vector, 1..65536. */
  dim: number;
  /** Codebook width, 1..6 (default 4). Unlike the index, no bit is held back. */
  bits?: number;
  seed?: Seed;
}

export interface Encoded {
  /** `n * codeBytes` bytes of packed codes. */
  codes: Uint8Array;
  /** One norm per vector; decoding needs these as well as the codes. */
  norms: Float32Array;
}

/**
 * Encode vectors compactly and decode them back, without an index.
 *
 * For storing vectors (a KV cache, say) rather than searching them. The index's scores
 * are tuned to preserve ranking; a codec reconstructs the vectors themselves, which is
 * what you want when every value matters.
 *
 * Call `close()` when done; as with Index, garbage collection is only a safety net.
 */
export class Codec {
  #state: CodecState;
  #dim: number;
  #codeBytes: number;

  constructor(options: CodecOptions) {
    if (typeof options !== 'object' || options === null) {
      throw new TypeError('Codec needs an options object, e.g. new Codec({ dim: 128 })');
    }
    const dim = checkDim(options.dim);
    const bits = checkInt(options.bits ?? 4, 'bits', 1, 6, ' for a codec');
    const seed = toSeed(options.seed);
    const out = [null];
    check(native().codecCreate({ dim, bits, seed }, out), 'Codec');
    this.#state = adoptCodec(this, out[0]);
    this.#dim = dim;
    this.#codeBytes = Number(native().codecCodeBytes(out[0]));
  }

  #handle(): unknown {
    const h = this.#state.handle;
    if (h === null) throw new Error('this Codec is closed');
    return h;
  }

  get dim(): number {
    return this.#dim;
  }

  /** Packed bytes of code per vector. Each vector also needs one float32 norm. */
  get codeBytes(): number {
    return this.#codeBytes;
  }

  get closed(): boolean {
    return this.#state.handle === null;
  }

  encode(vectors: Vectors): Encoded {
    const h = this.#handle();
    const a = asRows(vectors, this.#dim, 'vectors');
    const n = a.length / this.#dim;
    const codes = new Uint8Array(n * this.#codeBytes);
    const norms = new Float32Array(n);
    if (n > 0) check(native().codecEncode(h, a, n, codes, norms), 'encode', NON_FINITE_HINT);
    return { codes, norms };
  }

  /** Decode `norms.length` vectors into one flat Float32Array of `n * dim` floats. */
  decode(codes: Uint8Array, norms: Float32Array | readonly number[]): Float32Array {
    const h = this.#handle();
    const n = norms.length;
    if (!(codes instanceof Uint8Array)) throw new TypeError('codes must be a Uint8Array');
    if (codes.length !== n * this.#codeBytes) {
      throw new RangeError(
        `codes holds ${codes.length} bytes; ${n} norms need ${n * this.#codeBytes} ` +
          `(${this.#codeBytes} per vector)`,
      );
    }
    const nrm = norms instanceof Float32Array ? norms : Float32Array.from(norms);
    const rows = new Float32Array(n * this.#dim);
    if (n > 0) check(native().codecDecode(h, codes, nrm, n, rows), 'decode');
    return rows;
  }

  /** Free the native codec. Safe to call more than once. */
  close(): void {
    releaseCodec(this.#state);
  }
}
