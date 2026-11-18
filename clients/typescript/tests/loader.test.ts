/**
 * Where the library is looked for, and the CPU check with its portable fallback.
 *
 * Driven through a fake file system and a fake library, because the interesting case — a
 * CPU that cannot run the release build — is not one a test machine can be made into.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join, resolve } from 'node:path';
import { type Probe, libraryNames, packageRoot, resolveLibrary } from '../src/native.js';

interface FakeLib extends Probe {
  path: string;
  closed: boolean;
}

function setup(files: Record<string, { supported: boolean; cpu?: string }>, env: Record<string, string> = {}) {
  const opened: FakeLib[] = [];
  const result = () =>
    resolveLibrary<FakeLib>({
      env,
      packageRoot: '/pkg/node_modules/zquant',
      platform: { platform: 'linux', arch: 'x64' },
      exists: (p) => p in files,
      open: (path) => {
        const f = files[path];
        assert.ok(f, `opened ${path}, which does not exist`);
        const lib: FakeLib = {
          path,
          closed: false,
          cpuSupported: () => f.supported,
          buildCpu: () => f.cpu ?? 'x86_64_v2',
          close: () => {
            lib.closed = true;
          },
        };
        opened.push(lib);
        return lib;
      },
    });
  return { result, opened };
}

const PREBUILD = '/pkg/node_modules/zquant/prebuilds/linux-x64';

test('library names per platform', () => {
  assert.deepEqual(libraryNames('linux'), { main: 'libzquant.so', portable: 'libzquant-portable.so' });
  assert.deepEqual(libraryNames('darwin'), { main: 'libzquant.dylib', portable: 'libzquant-portable.dylib' });
  assert.deepEqual(libraryNames('win32'), { main: 'zquant.dll', portable: 'zquant-portable.dll' });
});

test('ZQUANT_LIB wins over everything else', () => {
  const explicit = resolve('/opt/zq/libzquant.so');
  const { result } = setup(
    { [explicit]: { supported: true }, [`${PREBUILD}/libzquant.so`]: { supported: true } },
    { ZQUANT_LIB: '/opt/zq/libzquant.so' },
  );
  const r = result();
  assert.equal(r.path, explicit);
  assert.equal(r.portable, false);
});

test('the older ZQUANT_LIBRARY is still honoured, after ZQUANT_LIB', () => {
  const a = resolve('/a/libzquant.so');
  const b = resolve('/b/libzquant.so');
  const files = { [a]: { supported: true }, [b]: { supported: true } };
  assert.equal(setup(files, { ZQUANT_LIBRARY: '/b/libzquant.so' }).result().path, b);
  assert.equal(setup(files, { ZQUANT_LIB: '/a/libzquant.so', ZQUANT_LIBRARY: '/b/libzquant.so' }).result().path, a);
});

test('a ZQUANT_LIB naming a missing file is an error, not a silent fallback', () => {
  const { result } = setup({ [`${PREBUILD}/libzquant.so`]: { supported: true } }, { ZQUANT_LIB: '/nope/libzquant.so' });
  assert.throws(result, /ZQUANT_LIB is set to .*nope.*but there is no file there/);
});

test('the packaged prebuild comes before a repository build', () => {
  const { result } = setup({
    [`${PREBUILD}/libzquant.so`]: { supported: true },
    ['/pkg/node_modules/zquant/zig-out/lib/libzquant.so']: { supported: true },
  });
  assert.equal(result().path, `${PREBUILD}/libzquant.so`);
});

test('without a prebuild, zig-out/lib is found walking up', () => {
  const { result } = setup({ ['/pkg/zig-out/lib/libzquant.so']: { supported: true } });
  assert.equal(result().path, '/pkg/zig-out/lib/libzquant.so');
});

test('nothing found: the error lists where it looked', () => {
  const { result } = setup({});
  assert.throws(result, (e: unknown) => {
    const m = (e as Error).message;
    return /could not find/.test(m) && m.includes(`${PREBUILD}/libzquant.so`) && /zig build lib/.test(m) && /ZQUANT_LIB/.test(m);
  });
});

test('a CPU that cannot run the release build gets the portable one beside it', () => {
  const { result, opened } = setup({
    [`${PREBUILD}/libzquant.so`]: { supported: false },
    [`${PREBUILD}/libzquant-portable.so`]: { supported: true, cpu: 'x86_64' },
  });
  const r = result();
  assert.equal(r.path, `${PREBUILD}/libzquant-portable.so`);
  assert.equal(r.portable, true);
  // The rejected library was checked, then released, and nothing else was called on it.
  assert.equal(opened[0].path, `${PREBUILD}/libzquant.so`);
  assert.equal(opened[0].closed, true);
  assert.equal(opened[1].closed, false);
});

test('the portable fallback also applies to an explicit ZQUANT_LIB', () => {
  const dir = resolve('/opt/zq');
  const { result } = setup(
    {
      [join(dir, 'libzquant.so')]: { supported: false },
      [join(dir, 'libzquant-portable.so')]: { supported: true },
    },
    { ZQUANT_LIB: join(dir, 'libzquant.so') },
  );
  assert.equal(result().path, join(dir, 'libzquant-portable.so'));
});

test('no portable build: the error names the CPU the library needs', () => {
  const { result, opened } = setup({ [`${PREBUILD}/libzquant.so`]: { supported: false, cpu: 'x86_64_v3' } });
  assert.throws(result, (e: unknown) => {
    const m = (e as Error).message;
    return /this CPU cannot run/.test(m) && /built for x86_64_v3/.test(m) && /SSSE3/.test(m) && m.includes('libzquant-portable.so');
  });
  assert.equal(opened[0].closed, true);
});

test('a portable build that cannot run either is reported, not loaded', () => {
  const { result } = setup({
    [`${PREBUILD}/libzquant.so`]: { supported: false, cpu: 'x86_64_v3' },
    [`${PREBUILD}/libzquant-portable.so`]: { supported: false, cpu: 'x86_64' },
  });
  assert.throws(result, /nor the portable build/);
});

test('a directory holding only the portable build is used', () => {
  const { result } = setup({ [`${PREBUILD}/libzquant-portable.so`]: { supported: true } });
  const r = result();
  assert.equal(r.path, `${PREBUILD}/libzquant-portable.so`);
});

test('the package root is found above the compiled file', () => {
  // These tests run from build-test/tests, two levels below the package.
  const root = packageRoot();
  assert.ok(root.endsWith(join('clients', 'typescript')), root);
});
