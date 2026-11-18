/** Every accepted shape of input rows normalizes to the same flat Float32Array. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Index } from '../src/index.js';
import { asRows } from '../src/rows.js';
import { corpus, splitRows } from './helpers.js';

const d = 4;
const flat = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);

test('a flat Float32Array passes through without a copy', () => {
  assert.equal(asRows(flat, d, 'x'), flat);
});

test('number[][], Float32Array[] and number[] all give the same rows', () => {
  const nested = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
  ];
  assert.deepEqual(asRows(nested, d, 'x'), flat);
  assert.deepEqual(asRows([1, 2, 3, 4, 5, 6, 7, 8], d, 'x'), flat);
  // The bug: an array of Float32Arrays was taken for a flat array of numbers.
  assert.deepEqual(asRows([flat.subarray(0, 4), flat.subarray(4, 8)], d, 'x'), flat);
  assert.deepEqual(asRows([new Float32Array([1, 2, 3, 4]), [5, 6, 7, 8]], d, 'x'), flat);
});

test('other typed arrays are accepted, flat or as rows', () => {
  assert.deepEqual(asRows(new Float64Array(flat) as never, d, 'x'), flat);
  assert.deepEqual(asRows([new Float64Array([1, 2, 3, 4]), new Float64Array([5, 6, 7, 8])] as never, d, 'x'), flat);
});

test('an empty array is no rows', () => {
  assert.equal(asRows([], d, 'x').length, 0);
});

test('wrong shapes are named in the error', () => {
  assert.throws(() => asRows(new Float32Array(5), d, 'vectors'), /vectors length 5 is not a multiple of dim 4/);
  assert.throws(() => asRows([new Float32Array(4), new Float32Array(3)], d, 'vectors'), /vectors row 1 has dim 3, expected 4/);
  assert.throws(() => asRows([[1, 2, 3, 4], 5] as never, d, 'vectors'), /row 1 is not an array/);
  assert.throws(() => asRows([1, 2, '3', 4] as never, d, 'vectors'), /vectors\[2\] is string/);
  assert.throws(() => asRows('nope' as never, d, 'vectors'), TypeError);
});

test('an index built from Float32Array rows matches one built from the flat array', () => {
  const dim = 32;
  const x = corpus(200, dim, 11);
  const a = new Index({ dim, bits: 4 });
  const b = new Index({ dim, bits: 4 });
  try {
    a.add(x);
    b.add(splitRows(x, dim));
    assert.equal(b.size, 200);
    // Same codes give the same scores for every stored vector, to the bit.
    const fromRows = b.search(splitRows(x, dim), { k: 200 });
    const fromFlat = a.search(x, { k: 200 });
    assert.deepEqual(fromRows.ids, fromFlat.ids);
    assert.deepEqual(fromRows.scores, fromFlat.scores);
  } finally {
    a.close();
    b.close();
  }
});
