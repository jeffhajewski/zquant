/** Behaviour a user would rely on, and the mistakes they would actually make. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Index, ZquantError, version } from '../src/index.js';

function corpus(n: number, d: number, seed = 1, offset = 0): Float32Array {
  // xorshift, so the corpora are identical across runs and platforms.
  let s = BigInt(seed) * 88172645463325252n;
  const rnd = (): number => {
    s ^= (s << 13n) & 0xffffffffffffffffn;
    s ^= s >> 7n;
    s ^= (s << 17n) & 0xffffffffffffffffn;
    return Number(s % 1000000n) / 1000000;
  };
  const out = new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    let sq = 0;
    for (let j = 0; j < d; j++) {
      const u = Math.max(rnd(), 1e-9);
      const v = rnd();
      const g = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) + offset;
      out[i * d + j] = g;
      sq += g * g;
    }
    const inv = 1 / Math.sqrt(sq);
    for (let j = 0; j < d; j++) out[i * d + j] *= inv;
  }
  return out;
}

function exactTopK(x: Float32Array, q: Float32Array, d: number, k: number): number[][] {
  const n = x.length / d;
  const out: number[][] = [];
  for (let i = 0; i < q.length / d; i++) {
    const scored: Array<[number, number]> = [];
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let c = 0; c < d; c++) s += q[i * d + c] * x[j * d + c];
      scored.push([s, j]);
    }
    scored.sort((a, b) => b[0] - a[0]);
    out.push(scored.slice(0, k).map(([, j]) => j));
  }
  return out;
}

test('version is reported', () => {
  assert.match(version(), /^\d+\.\d+\.\d+$/);
});

test('a vector retrieves itself', () => {
  const d = 64;
  const x = corpus(2000, d);
  const ix = new Index({ dim: d, bits: 5 });
  try {
    ix.add(x);
    assert.equal(ix.size, 2000);
    const { ids, k } = ix.search(x.subarray(0, 50 * d), { k: 10 });
    let self = 0;
    for (let i = 0; i < 50; i++) {
      if (ids.subarray(i * k, (i + 1) * k).includes(i)) self++;
    }
    assert.ok(self >= 48, `self-retrieval ${self}/50`);
  } finally {
    ix.close();
  }
});

test('recall against an exact scan', () => {
  const d = 128;
  const x = corpus(3000, d, 2);
  const q = corpus(100, d, 3);
  const exact = exactTopK(x, q, d, 10);
  const ix = new Index({ dim: d, bits: 5 });
  try {
    ix.add(x);
    const { ids, k } = ix.search(q, { k: 10 });
    let hits = 0;
    for (let i = 0; i < exact.length; i++) {
      const got = new Set(ids.subarray(i * k, (i + 1) * k));
      hits += exact[i].filter((j) => got.has(j)).length;
    }
    const recall = hits / (exact.length * 10);
    assert.ok(recall > 0.85, `recall ${recall.toFixed(3)} too low`);
  } finally {
    ix.close();
  }
});

test('threaded and single-threaded results agree', () => {
  const d = 64;
  const ix = new Index({ dim: d, bits: 5 });
  try {
    ix.add(corpus(3000, d, 4));
    const q = corpus(64, d, 5);
    assert.deepEqual(ix.search(q, { k: 10, threads: 1 }).ids, ix.search(q, { k: 10, threads: 4 }).ids);
  } finally {
    ix.close();
  }
});

test('more queries than one native call accepts', () => {
  const d = 32;
  const ix = new Index({ dim: d, bits: 5 });
  try {
    ix.add(corpus(1000, d, 6));
    const q = corpus(300, d, 7);
    const many = ix.search(q, { k: 5 });
    for (let i = 0; i < 300; i += 37) {
      const one = ix.search(q.subarray(i * d, (i + 1) * d), { k: 5 });
      assert.deepEqual(one.ids, many.ids.subarray(i * 5, (i + 1) * 5), `query ${i}`);
    }
  } finally {
    ix.close();
  }
});

test('an index smaller than k pads the tail', () => {
  const d = 16;
  const ix = new Index({ dim: d, bits: 5 });
  try {
    const x = corpus(3, d, 8);
    ix.add(x);
    const { ids, scores } = ix.search(x.subarray(0, d), { k: 10 });
    for (let j = 3; j < 10; j++) {
      assert.equal(ids[j], 0xffffffff);
      assert.equal(scores[j], -Infinity);
    }
  } finally {
    ix.close();
  }
});

test('errors surface as exceptions', () => {
  const d = 16;
  const ix = new Index({ dim: d, bits: 5 });
  try {
    const x = corpus(100, d, 9);
    assert.throws(() => ix.search(x.subarray(0, d), { k: 10 }), ZquantError); // empty
    ix.add(x);
    assert.throws(() => ix.calibrate(x), ZquantError); // calibrate after add
    assert.throws(() => ix.search(x.subarray(0, d), { k: 0 }), /k must be/);
    assert.throws(() => ix.add(new Float32Array(17)), /multiple of dim/);
  } finally {
    ix.close();
  }
  assert.throws(() => new Index({ dim: 8, metric: 'nope' as never }), /metric/);
});

test('accepts arrays of arrays', () => {
  const d = 8;
  const ix = new Index({ dim: d, bits: 5 });
  try {
    const rows = Array.from({ length: 50 }, (_, i) =>
      Array.from({ length: d }, (_, j) => Math.sin(i * d + j)),
    );
    ix.add(rows);
    assert.equal(ix.size, 50);
    assert.equal(ix.search(rows[0], { k: 3 }).ids.length, 3);
  } finally {
    ix.close();
  }
});
