/** Deterministic test data, shared by the test files. */

/** n unit vectors of dimension d, Gaussian before normalizing, shifted by `offset`. */
export function corpus(n: number, d: number, seed = 1, offset = 0): Float32Array {
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

/** Exact inner products of each query against every row, best first, top k. */
export function exactTopK(x: Float32Array, q: Float32Array, d: number, k: number): number[][] {
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

export function recall(ids: Uint32Array, k: number, exact: number[][]): number {
  let hits = 0;
  for (let i = 0; i < exact.length; i++) {
    const got = new Set(ids.subarray(i * k, (i + 1) * k));
    hits += exact[i].filter((j) => got.has(j)).length;
  }
  return hits / (exact.length * exact[0].length);
}

/** Rows of a flat array as separate Float32Arrays, viewing the same memory. */
export function splitRows(x: Float32Array, d: number): Float32Array[] {
  return Array.from({ length: x.length / d }, (_, i) => x.subarray(i * d, (i + 1) * d));
}
