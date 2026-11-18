/** close(), use after close, and freeing what was never closed. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { setFlagsFromString } from 'node:v8';
import { runInNewContext } from 'node:vm';
import { Codec, Index } from '../src/index.js';
import { live } from '../src/handles.js';
import { corpus } from './helpers.js';

// Expose the collector without needing --expose-gc on the command line, which
// `node --test` does not pass to the processes it runs test files in on every version.
setFlagsFromString('--expose-gc');
const gc = runInNewContext('gc') as () => void;

/** Collect until finalizers have had a chance to run. */
async function collect(): Promise<void> {
  for (let i = 0; i < 10; i++) {
    gc();
    await new Promise((r) => setImmediate(r));
  }
}

test('close is idempotent, and frees searchers with the index', () => {
  const before = { ...live };
  const ix = new Index({ dim: 16 });
  const x = corpus(50, 16, 60);
  ix.add(x);
  ix.search(x, { k: 3 });
  ix.search(x, { k: 5, threads: 2 });
  ix.scoreIds(x.subarray(0, 16), [1, 2]);
  assert.equal(live.index, before.index + 1);
  assert.equal(live.searcher, before.searcher + 3);
  assert.equal(ix.closed, false);
  ix.close();
  assert.equal(ix.closed, true);
  assert.equal(live.index, before.index);
  assert.equal(live.searcher, before.searcher);
  ix.close();
  ix.close();
  assert.equal(live.index, before.index);

  const codec = new Codec({ dim: 8 });
  assert.equal(live.codec, before.codec + 1);
  codec.close();
  codec.close();
  assert.equal(codec.closed, true);
  assert.equal(live.codec, before.codec);
});

test('using an index after close throws instead of touching freed memory', () => {
  const ix = new Index({ dim: 8 });
  const x = corpus(10, 8, 61);
  ix.add(x);
  ix.close();
  const calls: Array<[string, () => unknown]> = [
    ['size', () => ix.size],
    ['bytesPerVector', () => ix.bytesPerVector],
    ['add', () => ix.add(x)],
    ['add threaded', () => ix.add(x, { threads: 2 })],
    ['calibrate', () => ix.calibrate(x)],
    ['calibrate auto', () => ix.calibrate(x, { auto: true })],
    ['search', () => ix.search(x)],
    ['searchRerank', () => ix.searchRerank(x, x)],
    ['scoreIds', () => ix.scoreIds(x.subarray(0, 8), [0])],
    ['serialize', () => ix.serialize()],
    ['save', () => ix.save('/nonexistent/never-written')],
  ];
  for (const [name, call] of calls) assert.throws(call, /this Index is closed/, name);
  // dim is plain data and stays readable.
  assert.equal(ix.dim, 8);

  const codec = new Codec({ dim: 8 });
  const enc = codec.encode(x);
  codec.close();
  assert.throws(() => codec.encode(x), /this Codec is closed/);
  assert.throws(() => codec.decode(enc.codes, enc.norms), /this Codec is closed/);
});

test('an index or codec that is never closed is freed when collected', async () => {
  await collect();
  const before = { ...live };
  (function leak() {
    for (let i = 0; i < 20; i++) {
      const ix = new Index({ dim: 32 });
      const x = corpus(20, 32, 62 + i);
      ix.add(x);
      ix.search(x, { k: 2 });
      new Codec({ dim: 32 }).encode(x);
    }
  })();
  assert.equal(live.index, before.index + 20);
  assert.equal(live.searcher, before.searcher + 20);
  assert.equal(live.codec, before.codec + 20);
  await collect();
  assert.equal(live.index, before.index, 'indexes freed');
  assert.equal(live.searcher, before.searcher, 'searchers freed');
  assert.equal(live.codec, before.codec, 'codecs freed');
});

test('a closed index is not freed again when collected', async () => {
  await collect();
  const before = { ...live };
  (function closeThenDrop() {
    for (let i = 0; i < 10; i++) {
      const ix = new Index({ dim: 8 });
      ix.add(corpus(5, 8, 90 + i));
      ix.search(corpus(1, 8, 91), { k: 1 });
      ix.close();
    }
  })();
  await collect();
  // A double free would drive the count below where it started (or crash the process).
  assert.deepEqual({ ...live }, before);
});
