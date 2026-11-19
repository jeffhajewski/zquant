/**
 * Finding, checking and declaring the native library.
 *
 * koffi rather than a compiled N-API addon: installing then needs no compiler, and the
 * ABI exercised here is the same one the Python and Go clients use, so a mistake in it
 * surfaces in all three rather than hiding in one.
 *
 * The library is loaded on first use rather than at import, so importing the package
 * never throws, and a missing or unusable library is reported by the first call that
 * needs it.
 */
import koffi from 'koffi';
import { existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export interface Platform {
  platform: string;
  arch: string;
}

/** File names of the release library and its portable fallback on a platform. */
export function libraryNames(platform: string): { main: string; portable: string } {
  if (platform === 'darwin') return { main: 'libzquant.dylib', portable: 'libzquant-portable.dylib' };
  if (platform === 'win32') return { main: 'zquant.dll', portable: 'zquant-portable.dll' };
  return { main: 'libzquant.so', portable: 'libzquant-portable.so' };
}

/** The directory inside the package that `scripts/pack.sh` fills. */
export function prebuildDir(packageRoot: string, p: Platform): string {
  return join(packageRoot, 'prebuilds', `${p.platform}-${p.arch}`);
}

/** What the loader needs from an opened library, so tests can supply a fake. */
export interface Probe {
  /** zq_cpu_supported() != 0. */
  cpuSupported(): boolean;
  /** zq_build_cpu(), the CPU model the library was compiled for. */
  buildCpu(): string;
  /** Release the library. Called on a library that is opened and then rejected. */
  close(): void;
}

export interface ResolveOptions<P extends Probe> {
  env: Record<string, string | undefined>;
  packageRoot: string;
  platform: Platform;
  exists(path: string): boolean;
  open(path: string): P;
}

export interface Resolved<P extends Probe> {
  path: string;
  probe: P;
  /** True when the release library could not run here and the portable one was used. */
  portable: boolean;
}

/** The environment variable that names the library explicitly. */
export const ENV_VAR = 'ZQUANT_LIB';
/** The older name, still honoured so existing setups keep working. */
export const LEGACY_ENV_VAR = 'ZQUANT_LIBRARY';

/**
 * Pick the library to load, open it, and apply the CPU check.
 *
 * Search order: the `ZQUANT_LIB` environment variable, then `prebuilds/<platform>-<arch>`
 * inside the package, then a `zig-out/lib` found walking up from the package (the
 * repository's build tree during development).
 *
 * Whatever is found, `zq_cpu_supported()` is the first thing called on it. A release
 * library uses instructions (SSSE3, or dotprod on arm64) some CPUs lack, and calling
 * anything else on such a CPU could die of an illegal instruction. If the check fails,
 * a portable build in the same directory is used instead; with none, the error names
 * the CPU the library needs.
 */
export function resolveLibrary<P extends Probe>(o: ResolveOptions<P>): Resolved<P> {
  const names = libraryNames(o.platform.platform);
  let primary: string | undefined;
  const tried: string[] = [];

  const explicitVar = o.env[ENV_VAR] ? ENV_VAR : o.env[LEGACY_ENV_VAR] ? LEGACY_ENV_VAR : undefined;
  if (explicitVar) {
    const path = resolve(o.env[explicitVar] as string);
    if (!o.exists(path)) {
      throw new Error(`${explicitVar} is set to ${path}, but there is no file there`);
    }
    primary = path;
  } else {
    for (const dir of searchDirs(o.packageRoot, o.platform)) {
      // A directory holding only the portable build is still a valid install.
      for (const name of [names.main, names.portable]) {
        const path = join(dir, name);
        if (o.exists(path)) {
          primary = path;
          break;
        }
        tried.push(path);
      }
      if (primary) break;
    }
  }

  if (!primary) {
    throw new Error(
      `could not find the zquant native library (${names.main}). Build it with ` +
        '`zig build lib` at the repository root, or set ZQUANT_LIB to its path.\n' +
        `Looked in:\n  ${tried.join('\n  ')}`,
    );
  }

  const probe = o.open(primary);
  if (probe.cpuSupported()) return { path: primary, probe, portable: false };

  const builtFor = probe.buildCpu();
  probe.close();
  const portablePath = join(dirname(primary), names.portable);
  if (portablePath !== primary && o.exists(portablePath)) {
    const fallback = o.open(portablePath);
    if (fallback.cpuSupported()) return { path: portablePath, probe: fallback, portable: true };
    const fallbackCpu = fallback.buildCpu();
    fallback.close();
    throw new Error(
      `this CPU cannot run ${primary} (built for ${builtFor}), nor the portable build ` +
        `beside it, ${portablePath} (built for ${fallbackCpu})`,
    );
  }
  throw new Error(
    `this CPU cannot run ${primary}: it was built for ${builtFor} and needs instructions ` +
      `this CPU lacks (${requiredFeature(o.platform.arch)}). No portable build was found at ` +
      `${portablePath}; put one there, or point ZQUANT_LIB at one.`,
  );
}

function requiredFeature(arch: string): string {
  if (arch === 'x64') return 'SSSE3 or later';
  if (arch === 'arm64') return 'the dot-product extension';
  return 'see zq_build_cpu';
}

function searchDirs(packageRoot: string, p: Platform): string[] {
  const dirs = [prebuildDir(packageRoot, p)];
  // Development: walk up for the repository's build tree rather than assuming a depth.
  let dir = packageRoot;
  for (let i = 0; i < 6; i++) {
    dirs.push(join(dir, 'zig-out', 'lib'));
    // Windows puts DLLs in bin, not lib.
    if (p.platform === 'win32') dirs.push(join(dir, 'zig-out', 'bin'));
    const parent = resolve(dir, '..');
    if (parent === dir) break;
    dir = parent;
  }
  return dirs;
}

function here(): string {
  return dirname(fileURLToPath(import.meta.url));
}

/**
 * The package directory: the nearest ancestor of this file holding a package.json.
 * The compiled file sits in `dist/` when installed, and deeper when the tests compile
 * the sources separately, so walking up is what works for both.
 */
export function packageRoot(start: string = here()): string {
  let dir = start;
  for (let i = 0; i < 8; i++) {
    if (existsSync(join(dir, 'package.json'))) return dir;
    const parent = resolve(dir, '..');
    if (parent === dir) break;
    dir = parent;
  }
  return resolve(start, '..');
}

// Opaque handles: koffi passes them through without describing their layout, so the
// binding cannot depend on the library's internals.
koffi.opaque('zq_index');
koffi.opaque('zq_searcher');
koffi.opaque('zq_codec');

koffi.struct('zq_config', {
  dim: 'uint32_t',
  bits: 'uint8_t',
  metric: 'int',
  seed: 'uint64_t',
  compact: 'int',
});

koffi.struct('zq_codec_config', {
  dim: 'uint32_t',
  bits: 'uint8_t',
  seed: 'uint64_t',
});

interface KoffiProbe extends Probe {
  lib: koffi.IKoffiLib;
}

function openKoffi(path: string): KoffiProbe {
  const lib = koffi.load(path);
  let supported: () => number;
  let buildCpu: () => string;
  try {
    supported = lib.func('int zq_cpu_supported()');
    buildCpu = lib.func('const char *zq_build_cpu()');
  } catch {
    lib.unload?.();
    throw new Error(
      `${path} is too old for this binding: it has no zq_cpu_supported. Rebuild it with ` +
        '`zig build lib`, or point ZQUANT_LIB at a current build.',
    );
  }
  return {
    lib,
    cpuSupported: () => supported() !== 0,
    buildCpu: () => buildCpu(),
    close: () => lib.unload?.(),
  };
}

function declare(lib: koffi.IKoffiLib) {
  return {
    version: lib.func('const char *zq_version()'),
    statusString: lib.func('const char *zq_status_string(int status)'),
    buildCpu: lib.func('const char *zq_build_cpu()'),
    buildKernels: lib.func('const char *zq_build_kernels()'),

    indexCreate: lib.func('int zq_index_create(zq_config *config, _Out_ zq_index **out)'),
    indexFree: lib.func('void zq_index_free(zq_index *index)'),
    indexCalibrate: lib.func('int zq_index_calibrate(zq_index *index, float *rows, size_t n)'),
    indexCalibrateAuto: lib.func(
      'int zq_index_calibrate_auto(zq_index *index, float *rows, size_t n, _Out_ float *centroid_norm, _Out_ int *applied)',
    ),
    indexAdd: lib.func('int zq_index_add(zq_index *index, float *rows, size_t n)'),
    indexAddParallel: lib.func(
      'int zq_index_add_parallel(zq_index *index, float *rows, size_t n, size_t threads)',
    ),
    indexSerializedSize: lib.func('size_t zq_index_serialized_size(zq_index *index)'),
    indexSerialize: lib.func(
      'int zq_index_serialize(zq_index *index, _Out_ uint8_t *buf, size_t len, _Out_ size_t *written)',
    ),
    indexDeserialize: lib.func(
      'int zq_index_deserialize(uint8_t *buf, size_t len, _Out_ zq_index **out)',
    ),
    indexCount: lib.func('size_t zq_index_count(zq_index *index)'),
    indexBytesPerVector: lib.func('size_t zq_index_bytes_per_vector(zq_index *index)'),
    indexDim: lib.func('uint32_t zq_index_dim(zq_index *index)'),
    indexConfig: lib.func('int zq_index_config(zq_index *index, _Out_ zq_config *out)'),
    indexCalibrated: lib.func('int zq_index_calibrated(zq_index *index)'),
    indexPartition: lib.func(
      'int zq_index_partition(zq_index *index, float *rows, size_t n, uint32_t lists, size_t threads)',
    ),
    indexLists: lib.func('uint32_t zq_index_lists(zq_index *index)'),

    searcherCreate: lib.func(
      'int zq_searcher_create(zq_index *index, size_t batch, size_t k, size_t threads, _Out_ zq_searcher **out)',
    ),
    searcherFree: lib.func('void zq_searcher_free(zq_searcher *searcher)'),
    searcherCapacity: lib.func('size_t zq_searcher_capacity(zq_searcher *searcher)'),
    searcherSetProbes: lib.func('int zq_searcher_set_probes(zq_searcher *searcher, size_t probes)'),

    search: lib.func(
      'int zq_search(zq_index *index, zq_searcher *searcher, float *queries, size_t nq, _Out_ uint32_t *out_ids, _Out_ float *out_scores)',
    ),
    searchRerank: lib.func(
      'int zq_search_rerank(zq_index *index, zq_searcher *searcher, float *queries, size_t nq, float *originals, size_t n_originals, size_t k, _Out_ uint32_t *ids, _Out_ float *scores)',
    ),
    scoreIds: lib.func(
      'int zq_score_ids(zq_index *index, zq_searcher *searcher, float *query, uint32_t *ids, size_t n, _Out_ float *scores)',
    ),

    codecCreate: lib.func('int zq_codec_create(zq_codec_config *config, _Out_ zq_codec **out)'),
    codecFree: lib.func('void zq_codec_free(zq_codec *codec)'),
    codecCodeBytes: lib.func('size_t zq_codec_code_bytes(zq_codec *codec)'),
    codecDim: lib.func('uint32_t zq_codec_dim(zq_codec *codec)'),
    codecEncode: lib.func(
      'int zq_codec_encode(zq_codec *codec, float *rows, size_t n, _Out_ uint8_t *codes, _Out_ float *norms)',
    ),
    codecDecode: lib.func(
      'int zq_codec_decode(zq_codec *codec, uint8_t *codes, float *norms, size_t n, _Out_ float *rows)',
    ),
  };
}

export type Native = ReturnType<typeof declare> & {
  /** Absolute path of the library that was loaded. */
  path: string;
  /** True if the portable fallback was loaded because this CPU cannot run the release build. */
  portable: boolean;
};

let loaded: Native | undefined;
let loadError: Error | undefined;

/** The loaded library, loading it on first use. A failure is remembered and rethrown. */
export function native(): Native {
  if (loaded) return loaded;
  if (loadError) throw loadError;
  try {
    const r = resolveLibrary({
      env: process.env,
      packageRoot: packageRoot(),
      platform: { platform: process.platform, arch: process.arch },
      exists: existsSync,
      open: openKoffi,
    });
    loaded = { ...declare(r.probe.lib), path: r.path, portable: r.portable };
    return loaded;
  } catch (e) {
    loadError = e instanceof Error ? e : new Error(String(e));
    throw loadError;
  }
}

export const OK = 0;
