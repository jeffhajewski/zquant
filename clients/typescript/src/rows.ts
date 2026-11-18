/** Turning the shapes callers pass vectors in into the flat float32 rows the library takes. */

/** One vector: `dim` numbers. */
export type Vector = Float32Array | readonly number[];

/**
 * Rows of `dim` floats each: one flat `Float32Array` (or `number[]`) holding them end to
 * end, or an array of rows, each a `number[]` or a `Float32Array`.
 */
export type Vectors = Float32Array | readonly number[] | readonly Vector[];

function isTyped(x: unknown): x is ArrayLike<number> {
  return ArrayBuffer.isView(x) && !(x instanceof DataView);
}

function isRow(x: unknown): x is ArrayLike<number> {
  return Array.isArray(x) || isTyped(x);
}

/** Normalize rows to one flat Float32Array, copying only when the input is not one already. */
export function asRows(x: Vectors, dim: number, name: string): Float32Array {
  if (x instanceof Float32Array) {
    if (x.length % dim !== 0) {
      throw new RangeError(`${name} length ${x.length} is not a multiple of dim ${dim}`);
    }
    return x;
  }
  if (isTyped(x)) {
    // Another typed array, such as a Float64Array: flat, converted.
    if (x.length % dim !== 0) {
      throw new RangeError(`${name} length ${x.length} is not a multiple of dim ${dim}`);
    }
    return Float32Array.from(x);
  }
  if (!Array.isArray(x)) {
    throw new TypeError(
      `${name} must be a Float32Array, an array of numbers, or an array of rows; got ${typeof x}`,
    );
  }
  if (x.length === 0) return new Float32Array(0);

  // Checking the first element for a row, not for an Array, is the point: a
  // Float32Array is not an Array, and an array of them is rows, not numbers.
  if (isRow(x[0])) {
    const rows = x as readonly unknown[];
    const out = new Float32Array(rows.length * dim);
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      if (!isRow(row)) {
        throw new TypeError(`${name} row ${i} is not an array or typed array`);
      }
      if (row.length !== dim) {
        throw new RangeError(`${name} row ${i} has dim ${row.length}, expected ${dim}`);
      }
      out.set(row, i * dim);
    }
    return out;
  }

  const flat = x as readonly unknown[];
  for (let i = 0; i < flat.length; i++) {
    if (typeof flat[i] !== 'number') {
      throw new TypeError(`${name}[${i}] is ${typeof flat[i]}, not a number`);
    }
  }
  if (flat.length % dim !== 0) {
    throw new RangeError(`${name} length ${flat.length} is not a multiple of dim ${dim}`);
  }
  return Float32Array.from(flat as readonly number[]);
}
