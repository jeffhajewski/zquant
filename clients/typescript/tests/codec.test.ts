/** Encoding and decoding without an index. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Codec, MAX_DIM, ZquantError } from '../src/index.js';
import { corpus, splitRows } from './helpers.js';

function relativeError(a: Float32Array, b: Float32Array): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i++) {
    num += (a[i] - b[i]) ** 2;
    den += a[i] ** 2;
  }
  return Math.sqrt(num / den);
}

test('decode reconstructs, and more bits reconstruct better', () => {
  const d = 128;
  const x = corpus(200, d, 50);
  let last = Infinity;
  for (const bits of [1, 2, 4, 6]) {
    const codec = new Codec({ dim: d, bits });
    try {
      const { codes, norms } = codec.encode(x);
      assert.equal(codes.length, 200 * codec.codeBytes);
      assert.equal(norms.length, 200);
      // Bit-packed: never more than a byte per coordinate, and at least `bits` bits.
      assert.ok(codec.codeBytes <= d && codec.codeBytes >= (d * bits) / 8, `${codec.codeBytes} bytes`);
      const back = codec.decode(codes, norms);
      assert.equal(back.length, x.length);
      const err = relativeError(x, back);
      assert.ok(err < last, `bits ${bits}: error ${err} not below ${last}`);
      last = err;
    } finally {
      codec.close();
    }
  }
  assert.ok(last < 0.05, `6-bit error ${last}`);
});

test('norms are the vectors\' norms', () => {
  const d = 16;
  const x = corpus(10, d, 51).map((v) => v * 3);
  const codec = new Codec({ dim: d });
  try {
    const { norms } = codec.encode(x);
    for (const n of norms) assert.ok(Math.abs(n - 3) < 1e-4, `${n}`);
  } finally {
    codec.close();
  }
});

test('encoding is deterministic per seed, and rows can be given separately', () => {
  const d = 32;
  const x = corpus(20, d, 52);
  const a = new Codec({ dim: d, bits: 3, seed: 99n });
  const b = new Codec({ dim: d, bits: 3, seed: 99 });
  const c = new Codec({ dim: d, bits: 3, seed: 100 });
  try {
    const ea = a.encode(x);
    assert.deepEqual(b.encode(splitRows(x, d)), ea);
    assert.notDeepEqual(c.encode(x).codes, ea.codes);
    // Decode takes number[] norms too.
    assert.deepEqual(a.decode(ea.codes, Array.from(ea.norms)), a.decode(ea.codes, ea.norms));
  } finally {
    a.close();
    b.close();
    c.close();
  }
});

test('an empty batch encodes and decodes to nothing', () => {
  const codec = new Codec({ dim: 8 });
  try {
    const { codes, norms } = codec.encode(new Float32Array(0));
    assert.equal(codes.length, 0);
    assert.equal(norms.length, 0);
    assert.equal(codec.decode(codes, norms).length, 0);
  } finally {
    codec.close();
  }
});

test('codec options and inputs are checked', () => {
  assert.throws(() => new Codec({ dim: 0 }), /dim must be an integer from 1 to 65536/);
  assert.throws(() => new Codec({ dim: MAX_DIM + 1 }), /dim/);
  assert.throws(() => new Codec({ dim: 8, bits: 0 }), /bits must be an integer from 1 to 6 for a codec, got 0/);
  assert.throws(() => new Codec({ dim: 8, bits: 7 }), /bits/);
  assert.throws(() => new Codec({ dim: 8, seed: 2 ** 60 }), /bigint/);
  assert.throws(() => new Codec(null as never), /options object/);

  const codec = new Codec({ dim: 8, bits: 4 });
  try {
    assert.equal(codec.dim, 8);
    assert.throws(() => codec.encode(new Float32Array(7)), /multiple of dim 8/);
    assert.throws(() => codec.encode(new Float32Array([1, 2, 3, NaN, 5, 6, 7, 8])), ZquantError);
    const { codes, norms } = codec.encode(corpus(3, 8, 53));
    assert.throws(() => codec.decode(codes.subarray(1), norms), /codes holds 11 bytes; 3 norms need 12 \(4 per vector\)/);
    assert.throws(() => codec.decode([1, 2] as never, norms), /Uint8Array/);
  } finally {
    codec.close();
  }
});
