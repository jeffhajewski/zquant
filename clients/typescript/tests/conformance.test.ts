/**
 * Replays tests/golden/conformance.json through the public API.
 *
 * The fixture's hashes are checked against the C library by the Zig suite, so a
 * mismatch here is a binding bug: a seed that lost bits on its way to uint64, a struct
 * field at the wrong offset, a buffer read at the wrong length.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Codec, Index, type Metric } from '../src/index.js';

interface IndexCase {
  dim: number;
  bits: number;
  metric: Metric;
  seed: string;
  compact: boolean;
  calibrate: boolean;
  rows: number;
  salt: number;
  sha256: string;
}

interface CodecCase {
  dim: number;
  bits: number;
  seed: string;
  rows: number;
  salt: number;
  sha256: string;
}

interface Fixture {
  index: IndexCase[];
  codec: CodecCase[];
}

function findFixture(): string {
  let dir = dirname(fileURLToPath(import.meta.url));
  for (let i = 0; i < 8; i++) {
    const path = join(dir, 'tests', 'golden', 'conformance.json');
    if (existsSync(path)) return path;
    const parent = resolve(dir, '..');
    if (parent === dir) break;
    dir = parent;
  }
  throw new Error('tests/golden/conformance.json not found above this test');
}

const fixture = JSON.parse(readFileSync(findFixture(), 'utf8')) as Fixture;

/** x[r][j] = ((r*7919 + j*104729 + salt) mod 2001 - 1000) / 1024, exactly as float32. */
function input(rows: number, dim: number, salt: number): Float32Array {
  const out = new Float32Array(rows * dim);
  for (let r = 0; r < rows; r++) {
    for (let j = 0; j < dim; j++) {
      out[r * dim + j] = (((r * 7919 + j * 104729 + salt) % 2001) - 1000) / 1024;
    }
  }
  return out;
}

function sha256(...parts: Uint8Array[]): string {
  const h = createHash('sha256');
  for (const p of parts) h.update(p);
  return h.digest('hex');
}

/** Seeds are hex strings, some above 2^53, so they go to BigInt and never through a number. */
function seedOf(hex: string): bigint {
  return BigInt(hex);
}

test('the fixture has the cases this test expects', () => {
  assert.ok(fixture.index.length >= 5, 'index cases');
  assert.ok(fixture.codec.length >= 3, 'codec cases');
  // At least one seed must exceed 2^53, or the BigInt path is not being exercised.
  assert.ok(fixture.index.some((c) => seedOf(c.seed) > BigInt(Number.MAX_SAFE_INTEGER)));
  assert.ok(fixture.codec.some((c) => seedOf(c.seed) > BigInt(Number.MAX_SAFE_INTEGER)));
});

fixture.index.forEach((c, i) => {
  test(`index case ${i}: dim ${c.dim}, bits ${c.bits}, ${c.metric}, seed ${c.seed}`, () => {
    const x = input(c.rows, c.dim, c.salt);
    const ix = new Index({
      dim: c.dim,
      bits: c.bits,
      metric: c.metric,
      seed: seedOf(c.seed),
      compact: c.compact,
    });
    try {
      if (c.calibrate) ix.calibrate(x);
      ix.add(x);
      assert.equal(sha256(ix.serialize()), c.sha256);
    } finally {
      ix.close();
    }
  });
});

fixture.codec.forEach((c, i) => {
  test(`codec case ${i}: dim ${c.dim}, bits ${c.bits}, seed ${c.seed}`, () => {
    const x = input(c.rows, c.dim, c.salt);
    const codec = new Codec({ dim: c.dim, bits: c.bits, seed: seedOf(c.seed) });
    try {
      const { codes, norms } = codec.encode(x);
      // Norms as little-endian float32, whatever this machine's byte order.
      const normBytes = new Uint8Array(norms.length * 4);
      const view = new DataView(normBytes.buffer);
      norms.forEach((v, r) => view.setFloat32(r * 4, v, true));
      assert.equal(sha256(codes, normBytes), c.sha256);
    } finally {
      codec.close();
    }
  });
});
