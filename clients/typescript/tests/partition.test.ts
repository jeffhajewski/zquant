/** Partitioned indexes: training, probing, and what a partitioned index refuses. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Index, Status, ZquantError } from '../src/index.js';
import { corpus, exactTopK, recall } from './helpers.js';

const isStatus = (status: number) => (e: unknown) => e instanceof ZquantError && e.status === status;

test('partitioning: probes trade recall for speed, and the default is an eighth', () => {
  const d = 32;
  const x = corpus(3000, d, 80, 0.3);
  const q = corpus(40, d, 81, 0.3);
  const exact = exactTopK(x, q, d, 10);
  const ix = new Index({ dim: d, bits: 4 });
  try {
    assert.equal(ix.lists, 0);
    ix.partition(x, { lists: 32, threads: 2 });
    assert.equal(ix.lists, 32);
    ix.add(x, { threads: 2 });
    const one = recall(ix.search(q, { probes: 1 }).ids, 10, exact);
    const all = recall(ix.search(q, { probes: 32 }).ids, 10, exact);
    // Isotropic data at d=32 tops out near 0.75 at 4 bits, flat or not.
    assert.ok(one < all && all > 0.65, `recall ${one} with 1 probe, ${all} with all`);
    assert.deepEqual(ix.search(q).ids, ix.search(q, { probes: 4 }).ids);
    ix.probes = 32;
    assert.deepEqual(ix.search(q).ids, ix.search(q, { probes: 32 }).ids);
    assert.throws(() => (ix.probes = 0), RangeError);
  } finally {
    ix.close();
  }
});

test('what a partitioned index refuses', () => {
  const d = 16;
  const x = corpus(400, d, 82, 0.3);
  const ix = new Index({ dim: d, bits: 3 });
  try {
    assert.throws(() => ix.partition(x, { lists: 0 }), RangeError);
    assert.throws(() => ix.partition(x.subarray(0, 3 * d), { lists: 8 }), isStatus(Status.ERR_INVALID));
    ix.partition(x, { lists: 8 });
    assert.throws(() => ix.partition(x, { lists: 8 }), isStatus(Status.ERR_STATE));
    assert.throws(() => ix.calibrate(x), isStatus(Status.ERR_STATE));
    ix.add(x);
    assert.throws(() => ix.serialize(), isStatus(Status.ERR_UNSUPPORTED));
    assert.throws(() => ix.scoreIds(x.subarray(0, d), [0]), isStatus(Status.ERR_UNSUPPORTED));
  } finally {
    ix.close();
  }
  const flat = new Index({ dim: d, bits: 3 });
  try {
    flat.add(x);
    assert.throws(() => flat.search(x.subarray(0, d), { probes: 2 }), isStatus(Status.ERR_STATE));
  } finally {
    flat.close();
  }
});
