/**
 * Locating and declaring the native library.
 *
 * koffi rather than a compiled N-API addon: installing then needs no compiler, and the
 * ABI exercised here is the same one the Python and Go clients use, so a mistake in it
 * surfaces in all three rather than hiding in one.
 */
import koffi from 'koffi';
import { existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { platform } from 'node:process';

const LIB_NAME =
  platform === 'darwin' ? 'libzquant.dylib' : platform === 'win32' ? 'zquant.dll' : 'libzquant.so';

function here(): string {
  // Works under both CommonJS and ESM output.
  if (typeof __dirname !== 'undefined') return __dirname;
  return dirname(fileURLToPath(import.meta.url));
}

function candidates(): string[] {
  const out: string[] = [];
  if (process.env.ZQUANT_LIBRARY) out.push(process.env.ZQUANT_LIBRARY);
  const base = here();
  // Packaged: beside the compiled output, or one level up from it.
  out.push(join(base, LIB_NAME));
  out.push(join(base, '..', LIB_NAME));
  // Development: walk up for a build tree rather than assuming a depth, since the
  // compiled output sits one level deeper than the sources it was built from.
  let dir = base;
  for (let i = 0; i < 6; i++) {
    out.push(join(dir, 'zig-out', 'lib', LIB_NAME));
    const parent = resolve(dir, '..');
    if (parent === dir) break;
    dir = parent;
  }
  return out;
}

function loadLibrary(): koffi.IKoffiLib {
  const tried: string[] = [];
  for (const path of candidates()) {
    if (existsSync(path)) return koffi.load(path);
    tried.push(path);
  }
  throw new Error(
    `could not find ${LIB_NAME}. Build it with \`zig build lib\` at the repository root, ` +
      `or set ZQUANT_LIBRARY to its path.\nLooked in:\n  ${tried.join('\n  ')}`,
  );
}

const lib = loadLibrary();

// Opaque handles: koffi passes them through without describing their layout, so the
// binding cannot depend on the library's internals.
koffi.opaque('zq_index');
koffi.opaque('zq_searcher');

koffi.struct('zq_config', {
  dim: 'uint32_t',
  bits: 'uint8_t',
  metric: 'int',
  seed: 'uint64_t',
  compact: 'int',
});

export const native = {
  version: lib.func('const char *zq_version()'),
  statusString: lib.func('const char *zq_status_string(int status)'),

  indexCreate: lib.func('int zq_index_create(zq_config *config, _Out_ zq_index **out)'),
  indexFree: lib.func('void zq_index_free(zq_index *index)'),
  indexCalibrate: lib.func('int zq_index_calibrate(zq_index *index, float *rows, size_t n)'),
  indexAdd: lib.func('int zq_index_add(zq_index *index, float *rows, size_t n)'),
  indexCount: lib.func('size_t zq_index_count(zq_index *index)'),
  indexBytesPerVector: lib.func('size_t zq_index_bytes_per_vector(zq_index *index)'),
  indexDim: lib.func('uint32_t zq_index_dim(zq_index *index)'),

  searcherCreate: lib.func(
    'int zq_searcher_create(zq_index *index, size_t batch, size_t k, size_t threads, _Out_ zq_searcher **out)',
  ),
  searcherFree: lib.func('void zq_searcher_free(zq_searcher *searcher)'),
  searcherCapacity: lib.func('size_t zq_searcher_capacity(zq_searcher *searcher)'),

  search: lib.func(
    'int zq_search(zq_index *index, zq_searcher *searcher, float *queries, size_t nq, _Out_ uint32_t *out_ids, _Out_ float *out_scores)',
  ),
};

export const OK = 0;
