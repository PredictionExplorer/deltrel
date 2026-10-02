import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { buildDeltrelWasm } from './build-deltrel-wasm.mjs';
import { runtimeDescriptor, verifiedBytes, verifyLegacyBrowserArtifacts, verifyQualifiedRuntime } from './browser-runtime.mjs';

test('default build verifies immutable assets without invoking a compiler or changing old clients', async () => {
  const root = resolve('.');
  const before = readFileSync(resolve(root, 'public/models/deltrel/manifest.json'));
  const result = await buildDeltrelWasm([], root, () => { throw new Error('must not compile'); });
  assert.equal(result.identity, runtimeDescriptor.runtime_content_sha256);
  assert.equal(result.search, runtimeDescriptor.search_algorithm);
  assert.deepEqual(readFileSync(resolve(root, 'public/models/deltrel/manifest.json')), before);
  assert.equal(verifyLegacyBrowserArtifacts(root), true);
});

test('development build refuses all public and existing output paths', async t => {
  const root = mkdtempSync(resolve(tmpdir(), 'deltrel-build-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(resolve(root, 'public'), { recursive: true });
  const spawn = () => { throw new Error('must not compile'); };
  await assert.rejects(buildDeltrelWasm(['--out-dir', 'public/new'], root, spawn), /outside public/);
  await assert.rejects(buildDeltrelWasm(['--out-dir', 'public/..scratch'], root, spawn), /outside public/);
  await assert.rejects(buildDeltrelWasm(['--out-dir', 'public'], root, spawn), /outside public/);
  symlinkSync(resolve(root, 'public'), resolve(root, 'alias'));
  await assert.rejects(buildDeltrelWasm(['--out-dir', 'alias/new'], root, spawn), /outside public/);
  mkdirSync(resolve(root, 'existing'));
  await assert.rejects(buildDeltrelWasm(['--out-dir', 'existing'], root, spawn), /new directory/);
  await assert.rejects(buildDeltrelWasm(['--unknown'], root, spawn), /Usage/);
});

test('artifact checks reject truncation and same-size substitution', t => {
  const root = mkdtempSync(resolve(tmpdir(), 'deltrel-hash-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const source = resolve('public/models/deltrel', runtimeDescriptor.directory, runtimeDescriptor.module.path);
  const target = resolve(root, 'module.js');
  const bytes = readFileSync(source);
  writeFileSync(target, bytes);
  assert.deepEqual(verifiedBytes(target, runtimeDescriptor.module), bytes);
  bytes[0] ^= 1;
  writeFileSync(target, bytes);
  assert.throws(() => verifiedBytes(target, runtimeDescriptor.module), /SHA-256/);
  writeFileSync(target, 'short');
  assert.throws(() => verifiedBytes(target, runtimeDescriptor.module), /SHA-256/);
});

test('tampered JavaScript is rejected before any module execution', async t => {
  const root = mkdtempSync(resolve(tmpdir(), 'deltrel-no-execute-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const directory = resolve(root, 'public/models/deltrel', runtimeDescriptor.directory);
  mkdirSync(directory, { recursive: true });
  writeFileSync(resolve(directory, runtimeDescriptor.module.path), 'globalThis.__unverifiedRuntimeExecuted = true;');
  await assert.rejects(verifyQualifiedRuntime(root), /SHA-256/);
  assert.equal(globalThis.__unverifiedRuntimeExecuted, undefined);
});
