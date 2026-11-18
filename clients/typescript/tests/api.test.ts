/** The calls added with the second version of the C ABI, and the checks made before them. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Index, MAX_DIM, MAX_THREADS, Status, ZquantError, buildInfo } from '../src/index.js';
import { corpus, exactTopK, recall, splitRows } from './helpers.js';

test('dim and bits are checked before the library sees them', () => {
  assert.throws(() => new Index({ dim: 0 }), /dim must be an integer from 1 to 65536, got 0/);
  assert.throws(() => new Index({ dim: MAX_DIM + 1 }), /dim must be an integer from 1 to 65536/);
  assert.throws(() => new Index({ dim: 1.5 }), /dim must be an integer/);
  assert.throws(() => new Index({ dim: '8' as never }), /dim must be an integer/);
  assert.throws(() => new Index({ dim: 8, bits: 1 }), /bits must be an integer from 2 to 6 for an index, got 1/);
  assert.throws(() => new Index({ dim: 8, bits: 7 }), /bits must be an integer from 2 to 6/);
  assert.throws(() => new Index({ dim: 8, metric: 'nope' as never }), /metric must be one of/);
  assert.throws(() => new Index(undefined as never), /options object/);
  // The extremes are accepted.
  for (const opts of [{ dim: 1, bits: 2 }, { dim: MAX_DIM, bits: 6 }]) {
    const ix = new Index(opts);
    assert.equal(ix.dim, opts.dim);
    ix.close();
  }
});

test('seeds: bigint over the full 64 bits, number only when exactly representable', () => {
  for (const seed of [0, 1, Number.MAX_SAFE_INTEGER, 0n, (1n << 64n) - 1n]) {
    new Index({ dim: 8, seed }).close();
  }
  assert.throws(() => new Index({ dim: 8, seed: 2 ** 53 }), /safe integer.*bigint/);
  assert.throws(() => new Index({ dim: 8, seed: -1 }), /non-negative/);
  assert.throws(() => new Index({ dim: 8, seed: 1.5 }), /safe integer/);
  assert.throws(() => new Index({ dim: 8, seed: 1n << 64n }), /2\^64 - 1/);
  assert.throws(() => new Index({ dim: 8, seed: -1n }), /between 0/);
  assert.throws(() => new Index({ dim: 8, seed: '1' as never }), TypeError);
});

test('a number seed and the same bigint seed build the same index', () => {
  const x = corpus(50, 16, 21);
  const a = new Index({ dim: 16, seed: 12345 });
  const b = new Index({ dim: 16, seed: 12345n });
  try {
    a.add(x);
    b.add(x);
    assert.deepEqual(a.serialize(), b.serialize());
  } finally {
    a.close();
    b.close();
  }
});

test('threads are checked', () => {
  const ix = new Index({ dim: 8 });
  try {
    const x = corpus(10, 8, 22);
    assert.throws(() => ix.add(x, { threads: MAX_THREADS + 1 }), /threads must be an integer from 0 to 1024/);
    assert.throws(() => ix.add(x, { threads: -1 }), /threads/);
    ix.add(x);
    assert.throws(() => ix.search(x, { threads: 2.5 }), /threads/);
  } finally {
    ix.close();
  }
});

test('non-finite input is refused, and the error says so', () => {
  const ix = new Index({ dim: 4 });
  try {
    const bad = new Float32Array([1, 2, NaN, 4]);
    assert.throws(
      () => ix.add(bad),
      (e: unknown) =>
        e instanceof ZquantError &&
        e.status === Status.ERR_INVALID &&
        e.code === 'ZQ_ERR_INVALID' &&
        /must be finite/.test(e.message),
    );
    assert.equal(ix.size, 0, 'nothing from a refused batch is added');
    ix.add(new Float32Array([1, 2, 3, 4]));
    assert.throws(() => ix.search(new Float32Array([1, Infinity, 3, 4]), { k: 1 }), ZquantError);
  } finally {
    ix.close();
  }
});

test('a parallel add produces the same index as a serial one', () => {
  const d = 64;
  const x = corpus(4000, d, 23);
  const serial = new Index({ dim: d, bits: 4 });
  const parallel = new Index({ dim: d, bits: 4 });
  try {
    serial.add(x);
    parallel.add(x, { threads: 4 });
    assert.equal(parallel.size, 4000);
    assert.deepEqual(parallel.serialize(), serial.serialize());
    // threads: 0 and 1 mean the calling thread.
    const one = new Index({ dim: d, bits: 4 });
    one.add(x, { threads: 0 });
    assert.deepEqual(one.serialize(), serial.serialize());
    one.close();
  } finally {
    serial.close();
    parallel.close();
  }
});

test('rerank recovers near-exact top-k, with exact scores', () => {
  const d = 64;
  const x = corpus(3000, d, 24);
  const q = corpus(40, d, 25);
  const exact = exactTopK(x, q, d, 10);
  const ix = new Index({ dim: d, bits: 3 });
  try {
    ix.add(x);
    const plain = ix.search(q, { k: 10 });
    const reranked = ix.searchRerank(q, x, { k: 10, candidates: 80 });
    assert.equal(reranked.k, 10);
    assert.equal(reranked.ids.length, 40 * 10);
    const before = recall(plain.ids, 10, exact);
    const after = recall(reranked.ids, 10, exact);
    assert.ok(after >= before, `rerank lowered recall: ${before} -> ${after}`);
    assert.ok(after > 0.97, `reranked recall ${after}`);
    // Scores are exact inner products against the originals.
    const id = reranked.ids[0];
    let dot = 0;
    for (let c = 0; c < d; c++) dot += q[c] * x[id * d + c];
    assert.ok(Math.abs(reranked.scores[0] - dot) < 1e-4, `${reranked.scores[0]} vs ${dot}`);
    // Best first.
    for (let j = 1; j < 10; j++) assert.ok(reranked.scores[j - 1] >= reranked.scores[j]);
  } finally {
    ix.close();
  }
});

test('rerank chunks large batches, threaded or not, and takes originals as rows', () => {
  const d = 32;
  const x = corpus(500, d, 26);
  const q = corpus(100, d, 27);
  const ix = new Index({ dim: d, bits: 4 });
  try {
    ix.add(x);
    const a = ix.searchRerank(q, x, { k: 5 });
    const b = ix.searchRerank(q, splitRows(x, d), { k: 5, threads: 3 });
    assert.deepEqual(a.ids, b.ids);
    const one = ix.searchRerank(q.subarray(70 * d, 71 * d), x, { k: 5 });
    assert.deepEqual(one.ids, a.ids.subarray(70 * 5, 71 * 5));
  } finally {
    ix.close();
  }
});

test('rerank checks its arguments', () => {
  const d = 8;
  const x = corpus(20, d, 28);
  const ix = new Index({ dim: d });
  try {
    assert.throws(() => ix.searchRerank(x.subarray(0, d), x), /empty/);
    ix.add(x);
    assert.throws(() => ix.searchRerank(x.subarray(0, d), x.subarray(0, 19 * d)), /originals holds 19 vectors but the index holds 20/);
    assert.throws(() => ix.searchRerank(x.subarray(0, d), x, { k: 5, candidates: 4 }), /candidates \(4\) must be at least k \(5\)/);
    assert.throws(() => ix.searchRerank(x.subarray(0, d), x, { k: 0 }), /k must be at least 1/);
  } finally {
    ix.close();
  }
});

test('scoreIds gives the scores search reports', () => {
  const d = 32;
  const x = corpus(300, d, 29);
  const ix = new Index({ dim: d, bits: 5 });
  try {
    ix.add(x);
    const q = x.subarray(7 * d, 8 * d);
    const { ids, scores } = ix.search(q, { k: 10 });
    const got = ix.scoreIds(q, ids);
    for (let j = 0; j < 10; j++) assert.ok(Math.abs(got[j] - scores[j]) < 1e-5, `id ${ids[j]}`);
    // number[] ids and number[] query work too, in any order.
    const rev = ix.scoreIds(Array.from(q), Array.from(ids).reverse());
    assert.deepEqual(Array.from(rev), Array.from(got).reverse());
    assert.equal(ix.scoreIds(q, []).length, 0);
  } finally {
    ix.close();
  }
});

test('scoreIds checks ids and the query', () => {
  const d = 8;
  const ix = new Index({ dim: d });
  try {
    ix.add(corpus(5, d, 30));
    const q = corpus(1, d, 31);
    assert.throws(() => ix.scoreIds(q, [0, 5]), /ids\[1\] is 5, but the index holds ids 0 to 4/);
    assert.throws(() => ix.scoreIds(q, [-1]), /ids\[0\] is -1/);
    assert.throws(() => ix.scoreIds(q, [1.5]), /ids\[0\] is 1.5/);
    assert.throws(() => ix.scoreIds(corpus(2, d, 32), [0]), /one vector of 8 floats, got 16/);
  } finally {
    ix.close();
  }
});

test('calibrate auto reports the centroid norm and whether it calibrated', () => {
  const d = 64;
  // Zero-mean data: a centroid near the origin, so no calibration.
  const centred = new Index({ dim: d });
  try {
    const r = centred.calibrate(corpus(500, d, 33), { auto: true });
    assert.ok(r.centroidNorm >= 0 && r.centroidNorm < 0.3, `norm ${r.centroidNorm}`);
    assert.equal(r.applied, false);
    // Not calibrated, so adding then calibrating is still refused as for any index.
    centred.add(corpus(10, d, 34));
    assert.throws(() => centred.calibrate(corpus(10, d, 35), { auto: true }), /Calibrate before adding/);
  } finally {
    centred.close();
  }

  // Offset data: the centroid sits well away from the origin.
  const offset = new Index({ dim: d });
  try {
    const r = offset.calibrate(corpus(500, d, 36, 1.5), { auto: true });
    assert.ok(r.centroidNorm > 0.3, `norm ${r.centroidNorm}`);
    assert.equal(r.applied, true);
  } finally {
    offset.close();
  }

  // A plain calibrate returns nothing; an auto one changes the index exactly as a plain
  // one does when it applies.
  const sample = corpus(500, d, 36, 1.5);
  const a = new Index({ dim: d });
  const b = new Index({ dim: d });
  try {
    assert.equal(a.calibrate(sample), undefined);
    b.calibrate(sample, { auto: true });
    const x = corpus(100, d, 37, 1.5);
    a.add(x);
    b.add(x);
    assert.deepEqual(a.serialize(), b.serialize());
  } finally {
    a.close();
    b.close();
  }
});

test('calibrate refuses an empty sample', () => {
  const ix = new Index({ dim: 8 });
  try {
    assert.throws(() => ix.calibrate(new Float32Array(0)), /at least one sample/);
  } finally {
    ix.close();
  }
});

test('build info names the library and what it was built for', () => {
  const info = buildInfo();
  assert.match(info.version, /^\d+\.\d+\.\d+$/);
  assert.ok(info.cpu.length > 0);
  assert.match(info.kernels, /shuffle=\w+ dot=\w+/);
  assert.ok(info.libraryPath.length > 0);
  assert.equal(typeof info.portable, 'boolean');
});

test('ZquantError names every status', () => {
  const codes: Record<number, string> = {
    [-1]: 'ZQ_ERR_ALLOC',
    [-2]: 'ZQ_ERR_INVALID',
    [-3]: 'ZQ_ERR_STATE',
    [-4]: 'ZQ_ERR_UNSUPPORTED',
    [-5]: 'ZQ_ERR_CPU',
    [-6]: 'ZQ_ERR_FORMAT',
  };
  for (const [status, code] of Object.entries(codes)) {
    const e = new ZquantError(Number(status), 'ctx');
    assert.equal(e.code, code);
    assert.equal(e.status, Number(status));
    assert.ok(e.message.startsWith('ctx: '), e.message);
    assert.ok(e.message.includes(code), e.message);
  }
  // CPU errors say what the library was built for.
  assert.match(new ZquantError(Status.ERR_CPU, 'Index').message, /was built for /);
  assert.match(new ZquantError(-99, 'ctx').message, /unknown status -99/);
});
