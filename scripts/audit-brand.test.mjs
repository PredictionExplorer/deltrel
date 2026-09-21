import { test } from 'node:test';
import assert from 'node:assert/strict';
import { inspectBrand } from './audit-brand.mjs';

const prior = String.fromCharCode(115, 116, 97, 114);

test('detects retired branding in paths, package names, prose and camel-case symbols', () => {
  for (const suffix of ['', 'train', 'serve', '-wasm', '/board.ts', 'AiClient']) {
    assert.ok(inspectBrand(`src/${prior}${suffix}`).length);
  }
  for (const text of [prior.toUpperCase() + '_RULES', `Double *${prior}`, 'as' + 'S' + prior.slice(1) + 'AiError', `current_${prior}s_fraction`, `${prior}Count`]) {
    assert.ok(inspectBrand('source.ts', text).length);
  }
});

test('allows unrelated programming terms and ordinary English', () => {
  assert.deepEqual(inspectBrand('training/warm-start.ts',
    'start restart startsWith padStart starvation starmap startGame Deltrel shoreline cape network'), []);
});

test('detects retired coordinates and symbols without rejecting new coordinates', () => {
  assert.ok(inspectBrand('board.ts', "'" + '*' + "10'").length);
  assert.ok(inspectBrand('board.ts', String.fromCharCode(0x2605)).length);
  assert.deepEqual(inspectBrand('board.ts', "['A', 'Z', 'AA', 'JO']"), []);
});
