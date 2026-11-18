/** Saving and loading, to bytes and to files. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Index, Status, ZquantError } from '../src/index.js';
import { corpus } from './helpers.js';

function isFormatError(e: unknown): boolean {
  return e instanceof ZquantError && e.status === Status.ERR_FORMAT && e.code === 'ZQ_ERR_FORMAT';
}

test('a loaded index searches exactly as the saved one did', () => {
  const d = 48;
  const x = corpus(800, d, 40, 0.5);
  const q = corpus(20, d, 41, 0.5);
  const ix = new Index({ dim: d, bits: 4, metric: 'cosine', seed: 0xfffffffffffffff1n, compact: false });
  try {
    ix.calibrate(x.subarray(0, 200 * d));
    ix.add(x);
    const bytes = ix.serialize();
    assert.ok(bytes instanceof Uint8Array);
    const copy = Index.deserialize(bytes);
    try {
      assert.equal(copy.dim, d);
      assert.equal(copy.size, 800);
      assert.equal(copy.bytesPerVector, ix.bytesPerVector);
      const a = ix.search(q, { k: 10 });
      const b = copy.search(q, { k: 10 });
      assert.deepEqual(b.ids, a.ids);
      assert.deepEqual(b.scores, a.scores);
      // It also accepts new vectors as the original would.
      const more = corpus(10, d, 42, 0.5);
      ix.add(more);
      copy.add(more);
      assert.deepEqual(copy.serialize(), ix.serialize());
    } finally {
      copy.close();
    }
  } finally {
    ix.close();
  }
});

test('a loaded index reports the settings it was saved with', () => {
  const d = 40;
  const x = corpus(600, d, 44, 0.5);
  const cases = [
    { metric: 'ip', compact: true, seed: 0x5eedn },
    { metric: 'cosine', compact: false, seed: 0n },
    { metric: 'l2', compact: true, seed: 0xffffffffffffffffn },
  ] as const;
  for (const c of cases) {
    const ix = new Index({ dim: d, bits: 3, metric: c.metric, seed: c.seed, compact: c.compact });
    try {
      assert.deepEqual(ix.settings, { dim: d, bits: 3, metric: c.metric, seed: c.seed, compact: c.compact });
      assert.equal(ix.calibrated, false);
      ix.calibrate(x);
      assert.equal(ix.calibrated, true);
      ix.add(x);
      const bytes = ix.serialize();
      const back = Index.deserialize(bytes);
      try {
        assert.deepEqual(back.settings, ix.settings);
        assert.equal(back.calibrated, true);
        assert.ok(Object.isFrozen(back.settings));
        // The settings are enough to build an index that encodes identically.
        const twin = new Index(back.settings);
        try {
          twin.calibrate(x);
          twin.add(x);
          assert.deepEqual(twin.serialize(), bytes);
        } finally {
          twin.close();
        }
      } finally {
        back.close();
      }
    } finally {
      ix.close();
    }
  }
});

test('settings the C configuration cannot describe are refused, not guessed', () => {
  const ix = new Index({ dim: 16 });
  let bytes: Uint8Array;
  try {
    ix.add(corpus(20, 16, 45));
    bytes = ix.serialize();
  } finally {
    ix.close();
  }
  bytes[22] = 1; // the exact scan, which only the Zig API can set
  new DataView(bytes.buffer, bytes.byteOffset).setUint32(bytes.length - 4, crc32(bytes.subarray(0, bytes.length - 4)), true);
  const back = Index.deserialize(bytes);
  try {
    assert.equal(back.size, 20);
    assert.throws(() => back.settings, (e: unknown) => e instanceof ZquantError && e.status === Status.ERR_UNSUPPORTED);
  } finally {
    back.close();
  }
});

function crc32(data: Uint8Array): number {
  let c = 0xffffffff;
  for (const byte of data) {
    c ^= byte;
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return (c ^ 0xffffffff) >>> 0;
}

test('deserialize takes a Uint8Array view, a Buffer or an ArrayBuffer', () => {
  const ix = new Index({ dim: 8 });
  try {
    ix.add(corpus(5, 8, 43));
    const bytes = ix.serialize();
    // A view at an offset into a larger buffer.
    const padded = new Uint8Array(bytes.length + 16);
    padded.set(bytes, 7);
    const view = padded.subarray(7, 7 + bytes.length);
    const arrayBuffer = new Uint8Array(bytes).buffer;
    for (const input of [bytes, view, arrayBuffer]) {
      const copy = Index.deserialize(input);
      assert.equal(copy.size, 5);
      copy.close();
    }
  } finally {
    ix.close();
  }
});

test('save and load through a file', () => {
  const dir = mkdtempSync(join(tmpdir(), 'zquant-'));
  try {
    const path = join(dir, 'index.zq');
    const d = 16;
    const x = corpus(100, d, 44);
    const ix = new Index({ dim: d });
    ix.add(x);
    ix.save(path);
    const loaded = Index.load(path);
    try {
      assert.deepEqual(loaded.search(x.subarray(0, d), { k: 5 }), ix.search(x.subarray(0, d), { k: 5 }));
    } finally {
      loaded.close();
      ix.close();
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('an empty index round-trips', () => {
  const ix = new Index({ dim: 12, bits: 3 });
  const copy = Index.deserialize(ix.serialize());
  assert.equal(copy.size, 0);
  assert.equal(copy.dim, 12);
  ix.close();
  copy.close();
});

test('anything that is not an intact index is ZQ_ERR_FORMAT, not a crash', () => {
  const ix = new Index({ dim: 8 });
  ix.add(corpus(20, 8, 45));
  const good = ix.serialize();
  ix.close();

  assert.throws(() => Index.deserialize(new Uint8Array(0)), isFormatError);
  assert.throws(() => Index.deserialize(new Uint8Array(64)), isFormatError);
  assert.throws(() => Index.deserialize(Buffer.from('definitely not an index')), isFormatError);
  assert.throws(() => Index.deserialize(good.subarray(0, good.length - 1)), isFormatError);
  // One flipped bit anywhere fails the checksum.
  for (const at of [0, 9, good.length >> 1, good.length - 1]) {
    const bad = Buffer.from(good);
    bad[at] ^= 0x10;
    assert.throws(() => Index.deserialize(bad), isFormatError, `bit flip at ${at}`);
  }
  assert.throws(() => Index.deserialize('x' as never), TypeError);

  const dir = mkdtempSync(join(tmpdir(), 'zquant-'));
  try {
    const path = join(dir, 'junk.zq');
    writeFileSync(path, 'junk');
    assert.throws(() => Index.load(path), isFormatError);
    assert.throws(() => Index.load(join(dir, 'missing.zq')), /ENOENT/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
