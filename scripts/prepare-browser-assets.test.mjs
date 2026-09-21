import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { prepareBrowserAssets } from './prepare-browser-assets.mjs';
const { DELTREL_ORT_VERSION, DELTREL_ORT_ASSET_FILES } =
  await import('../src/lib/deltrel/ai/runtime-assets.ts');

function fixture(t, version = DELTREL_ORT_VERSION) {
  const root = mkdtempSync(resolve(tmpdir(), 'deltrel-runtime-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const dist = resolve(root, 'node_modules/onnxruntime-web/dist');
  mkdirSync(dist, { recursive: true });
  writeFileSync(resolve(dist, '../package.json'), JSON.stringify({ version }));
  for (const name of DELTREL_ORT_ASSET_FILES) {
    writeFileSync(resolve(dist, name), name.endsWith('.wasm')
      ? Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]) : 'export default function createRuntime() {}');
  }
  return { root, dist };
}

test('copies both exact installed runtime assets into the same-origin versioned directory', (t) => {
  const { root, dist } = fixture(t);
  const result = prepareBrowserAssets(root);
  assert.equal(result.version, DELTREL_ORT_VERSION);
  assert.equal(result.destination, resolve(root, `public/onnxruntime/${DELTREL_ORT_VERSION}`));
  for (const name of DELTREL_ORT_ASSET_FILES) {
    assert.deepEqual(readFileSync(resolve(result.destination, name)), readFileSync(resolve(dist, name)));
  }
});

test('fails before publishing if the npm runtime and browser configuration disagree', (t) => {
  const { root } = fixture(t, '0.0.0');
  assert.throws(() => prepareBrowserAssets(root), /does not match web asset version/);
  assert.equal(existsSync(resolve(root, 'public')), false);
});

test('requires the full asset pair and is repeatable after a clean install', (t) => {
  const { root, dist } = fixture(t);
  const missing = resolve(dist, DELTREL_ORT_ASSET_FILES[1]);
  rmSync(missing);
  assert.throws(() => prepareBrowserAssets(root), /asset is missing/);
  assert.equal(existsSync(resolve(root, 'public')), false);
  writeFileSync(missing, Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]));
  const first = prepareBrowserAssets(root);
  const second = prepareBrowserAssets(root);
  assert.deepEqual(first, second);
});
