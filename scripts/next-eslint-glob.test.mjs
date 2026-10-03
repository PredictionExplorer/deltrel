import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import vm from 'node:vm';
import test from 'node:test';
import { Linter } from 'eslint';
import nextPlugin from '@next/eslint-plugin-next';

const require = createRequire(import.meta.url);
const adapter = require('../vendor/next-eslint-glob/index.cjs');
const pluginRequire = createRequire(require.resolve('@next/eslint-plugin-next'));
const { getRootDirs } = pluginRequire('./utils/get-root-dirs.js');
const project = path.resolve(import.meta.dirname, '..');
const onlyDirectories = { onlyDirectories: true };

function fixture(run) {
  const cwd = process.cwd(), root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'deltrel-eslint-root-')));
  for (const name of ['apps/web/pages', 'apps/admin/app/about', 'apps/.hidden/pages', 'apps/literal space/pages']) {
    fs.mkdirSync(path.join(root, name), { recursive: true });
  }
  fs.writeFileSync(path.join(root, 'apps/web/pages/hello.js'), 'export default function Page() {}\n');
  fs.writeFileSync(path.join(root, 'apps/admin/app/about/page.js'), 'export default function Page() {}\n');
  fs.writeFileSync(path.join(root, 'apps/admin/app/page.js'), 'export default function Page() {}\n');
  fs.writeFileSync(path.join(root, 'apps/file.txt'), 'file');
  fs.symlinkSync('web', path.join(root, 'apps/link'), 'dir');
  fs.symlinkSync('missing', path.join(root, 'apps/broken'), 'dir');
  try { process.chdir(root); return run(root); }
  finally { process.chdir(cwd); fs.rmSync(root, { recursive: true, force: true }); }
}

test('only the pinned Next ESLint dependency edge resolves to this adapter', () => {
  assert.strictEqual(pluginRequire('fast-glob'), adapter);
  assert.equal(pluginRequire('../package.json').version, '16.3.8');
  const source = fs.readFileSync(pluginRequire.resolve('./utils/get-root-dirs.js'), 'utf8');
  assert.match(source, /require\("fast-glob"\)/);
  assert.match(source, /_fastglob\.globSync/);
  assert.match(source, /onlyDirectories: true/);
  assert.deepEqual(Object.keys(adapter), ['globSync']);
});

test('literal roots preserve spelling, files/missing roots stay excluded, and cwd is read at call time', () => fixture(root => {
  for (const value of ['.', 'apps', './apps/web', 'apps/web/', 'apps/literal space', 'apps/link', root]) {
    assert.deepEqual(adapter.globSync(value, onlyDirectories), [value]);
  }
  for (const value of ['apps/missing', 'apps/broken', 'apps/file.txt']) {
    assert.deepEqual(adapter.globSync(value, onlyDirectories), []);
  }
  assert.deepEqual(adapter.globSync('apps/*', onlyDirectories), ['apps/admin', 'apps/link', 'apps/literal space', 'apps/web']);
}));

test('brace/globstar/hidden/absolute roots match the recorded fast-glob directory sets', () => fixture(root => {
  const cases = [
    ['apps/{web,admin}', ['apps/admin', 'apps/web']],
    ['apps/[aw]*', ['apps/admin', 'apps/web']],
    ['apps/???', ['apps/web']],
    ['apps/.*', ['apps/.hidden']],
    ['apps/**/pages', ['apps/link/pages', 'apps/literal space/pages', 'apps/web/pages']],
    ['apps/link/*', ['apps/link/pages']],
    ['!apps/web', []],
  ];
  for (const [pattern, expected] of cases) assert.deepEqual(adapter.globSync(pattern, onlyDirectories), expected, pattern);
  const descendants = ['apps/admin', 'apps/admin/app', 'apps/admin/app/about', 'apps/link', 'apps/link/pages',
    'apps/literal space', 'apps/literal space/pages', 'apps/web', 'apps/web/pages'];
  for (const pattern of ['apps/**', 'apps/**/', 'apps/**/*']) {
    assert.deepEqual(adapter.globSync(pattern, onlyDirectories), descendants, pattern);
  }
  assert.deepEqual(adapter.globSync(root + '/apps/*', onlyDirectories), ['admin', 'link', 'literal space', 'web'].map(name => root + '/apps/' + name));
}));

test('the unchanged Next helper preserves default cwd, array order, duplicates and separator conversion', () => fixture(root => {
  assert.deepEqual(getRootDirs({ cwd: root, settings: {} }), [root]);
  assert.deepEqual(getRootDirs({ cwd: root, settings: { next: { rootDir: ['apps/web/', 'apps/admin', 'apps/web/'] } } }), ['apps/web/', 'apps/admin', 'apps/web/']);
  assert.deepEqual(getRootDirs({ cwd: root, settings: { next: { rootDir: 'apps\\web' } } }), ['apps/web']);
  assert.deepEqual(getRootDirs({ cwd: root, settings: { next: { rootDir: ['apps/{web,admin}', 42, 'apps/link'] } } }), ['apps/admin', 'apps/web', 'apps/link']);
}));

test('real Next no-html-link-for-pages still detects Pages and App Router links through literal/glob/array/symlink roots', () => fixture(root => {
  const linter = new Linter({ cwd: root });
  for (const rootDir of ['apps/*', 'apps/{web,admin}', root + '/apps/*', ['apps/link', 'apps/admin']]) {
    const config = [{ files: ['**/*.jsx'], languageOptions: { parserOptions: { ecmaFeatures: { jsx: true } } },
      plugins: { '@next/next': nextPlugin }, settings: { next: { rootDir } },
      rules: { '@next/next/no-html-link-for-pages': 'error' } }];
    // Check the current plugin's Pages route and App Router root route.
    for (const href of ['/hello', '/']) {
      const messages = linter.verify(`const Component = () => <a href="${href}">Internal</a>;`, config, { filename: path.join(root, 'component.jsx') });
      assert(messages.some(message => message.ruleId === '@next/next/no-html-link-for-pages'), JSON.stringify({ rootDir, href, messages }));
    }
    assert.deepEqual(linter.verify('const Component = () => <a href="https://example.com">External</a>;', config, { filename: path.join(root, 'component.jsx') }), []);
  }
}));

test('directory symlink cycles are bounded while the symlink root itself remains visible', () => fixture(() => {
  fs.symlinkSync('..', 'apps/web/back', 'dir');
  const result = adapter.globSync('apps/**', onlyDirectories);
  assert(result.includes('apps/web/back'));
  assert(!result.some(value => value.includes('/back/')));
  // A explicitly selected alias root is a valid starting point, not a cycle.
  assert(adapter.globSync('apps/web/back/*', onlyDirectories).includes('apps/web/back/web'));
}));

test('cycle ancestry includes POSIX and drive roots without walking either filesystem', () => {
  const source = fs.readFileSync(path.join(project, 'vendor/next-eslint-glob/index.cjs'), 'utf8');
  for (const [paths, root, back] of [[path.posix, '/', '/a/loop'], [path.win32, 'C:\\', 'C:\\a\\loop']]) {
    const reads = [], cjsModule = { exports: {} };
    const fakeFs = { realpathSync: value => value === back ? root : value,
      readdirSync: value => { reads.push(value); return []; } };
    const fakeGlob = { isDynamicPattern: () => true, globSync: (_pattern, options) => {
      options.fs.readdirSync(root, { withFileTypes: true });
      options.fs.readdirSync(back, { withFileTypes: true });
      return [];
    } };
    vm.runInNewContext(source, { module: cjsModule, process: { cwd: () => root }, require: name => {
      if (name === 'node:fs') return fakeFs;
      if (name === 'node:path') return paths;
      if (name === 'tinyglobby') return fakeGlob;
      throw new Error('unexpected dependency');
    } });
    cjsModule.exports.globSync('*', onlyDirectories);
    assert.deepEqual(reads, [root]);
  }
});

test('unsupported options and patterns fail explicitly instead of losing lint coverage', () => {
  for (const options of [undefined, {}, { onlyDirectories: false }, { onlyDirectories: true, dot: true },
    { onlyDirectories: true, [Symbol('unknown')]: true }, Object.assign(Object.create({ onlyDirectories: true }), { unsupported: true })]) {
    assert.throws(() => adapter.globSync('apps/*', options), /supports only/);
  }
  for (const pattern of ['', ['apps/*'], 'apps/{1..3}', 'apps/+(web|admin)', 'apps\\web']) {
    assert.throws(() => adapter.globSync(pattern, onlyDirectories), /rootDir/);
  }
});

test('unreadable dynamic roots fail visibly instead of silently losing route lint', () => fixture(root => {
  const original = fs.readdirSync, denied = Object.assign(new Error('fixture permission denied'), { code: 'EACCES' });
  fs.readdirSync = (directory, options) => {
    if (path.resolve(directory) === path.join(root, 'apps/web')) throw denied;
    return original(directory, options);
  };
  try { assert.throws(() => adapter.globSync('apps/**', onlyDirectories), error => error === denied); }
  finally { fs.readdirSync = original; }
}));

test('adversarial matched/mismatched nesting and excessive length terminate in a bounded child', () => {
  const code = `const assert=require('node:assert/strict');const a=require(${JSON.stringify(path.join(project, 'vendor/next-eslint-glob/index.cjs'))});
    for(const p of ['{'.repeat(2000)+'a'+ '}'.repeat(2000),'{)'.repeat(1000),'['.repeat(2000),'x'.repeat(4097)])
      assert.throws(()=>a.globSync(p,{onlyDirectories:true}),/nesting exceeds|at most4096|at most 4096/);`;
  const result = spawnSync(process.execPath, ['-e', code], { timeout: 2000, encoding: 'utf8' });
  assert.equal(result.error, undefined); assert.equal(result.status, 0, result.stderr);
});

test('current App Router layout remains discoverable', () => {
  const current = getRootDirs({ cwd: project, settings: {} });
  assert.deepEqual(current, [project]);
  assert(fs.existsSync(path.join(current[0], 'src/app/layout.tsx')));
});
