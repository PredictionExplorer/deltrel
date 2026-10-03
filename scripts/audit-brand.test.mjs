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

test('preserves exact historical deployment identities without exempting other files or symbols', () => {
  for (const evidence of [
    'training/docs/elo-efficiency-runtime-deployment-evidence-20260923.json',
    'training/docs/training-recovery-deployment-20260928.md',
    'training/docs/training-recovery-deployment-evidence-20260928.json',
    'training/docs/strength-recovery-deployment-20261002.md',
  ]) {
    assert.deepEqual(inspectBrand(evidence, `${prior}train.model-pointer`), []);
    assert.ok(inspectBrand(evidence, String.fromCharCode(0x2605)).length);
  }
  assert.ok(inspectBrand('training/docs/new-release.md', `${prior}train.model-pointer`).length);
  assert.ok(inspectBrand('src/app/page.tsx', `${prior}train.model-pointer`).length);
});

test('allows exact predecessor wire identities only at the explicit migration boundary', () => {
  const bridge = 'training/deltreltrain/champion_migration.py';
  const fixtures = 'training/tests/test_champion_migration.py';
  const identifiers = [
    `${prior}train.checkpoint`, `${prior}train.model-manifest`, `${prior}train.model-pointer`,
    `edgeconnect.${prior}.rules.v3`, `edgeconnect.${prior}.action-layout.nodes-only.v1`,
  ];
  for (const path of [bridge, fixtures]) {
    for (const identifier of identifiers) {
      for (const quote of ['"', "'"]) {
        assert.deepEqual(inspectBrand(path, `value = ${quote}${identifier}${quote}`), []);
      }
      assert.ok(inspectBrand(path, `value = "${identifier}.unexpected"`).length);
      assert.ok(inspectBrand('src/app/page.tsx', `"${identifier}"`).length);
    }
    assert.ok(inspectBrand(path, `new ${prior}train product`).length);
    assert.ok(inspectBrand(path, `${prior}Count = 1`).length);
    assert.ok(inspectBrand(path, String.fromCharCode(0x2605)).length);
  }
  assert.deepEqual(inspectBrand(fixtures, `replace("deltreltrain.", "${prior}train.")`), []);
  assert.ok(inspectBrand(bridge, `"${prior}train."`).length);
});

test('limits migration prose exceptions to the documented historical boundary', () => {
  const oldName = prior[0].toUpperCase() + prior.slice(1) + 'Train';
  const bridgeText = `Supports the immediately preceding ${oldName} publication layout only.`;
  const docText = `from a frozen ${oldName} champion and a separately verified, conclusive promotion`;
  assert.deepEqual(inspectBrand('training/deltreltrain/champion_migration.py', bridgeText), []);
  assert.deepEqual(inspectBrand('training/docs/deltrel-rebrand.md', docText), []);
  assert.ok(inspectBrand('src/app/page.tsx', bridgeText).length);
  assert.ok(inspectBrand('training/docs/deltrel-rebrand.md', `Welcome to ${oldName}`).length);
  assert.ok(inspectBrand('training/deltreltrain/champion_migration.py', `"${oldName}"`).length);
});

test('detects retired coordinates and symbols without rejecting new coordinates', () => {
  assert.ok(inspectBrand('board.ts', "'" + '*' + "10'").length);
  assert.ok(inspectBrand('board.ts', String.fromCharCode(0x2605)).length);
  assert.deepEqual(inspectBrand('board.ts', "['A1', 'A10', 'S15', 'T4', 'R12', 'Y15']"), []);
});

test('allows only exact collector compatibility literals in their declared files', () => {
  const native = `${prior}_native`;
  const packageName = `${prior}train`;
  const entries = [
    ['training/scripts/strength_freshness_cpu_collect_identity.py', [native]],
    ['training/scripts/strength_freshness_cpu_collect_records.py', [`${packageName}.model-pointer`]],
    ['training/tests/test_strength_freshness_cpu_collect_identity.py', [
      packageName,
      native,
      `/release/${native}.so`,
      `/release/bin/python\\0-m\\0${packageName}.worker\\0`,
      `100-200 r-xp 0 08:01 10 /release/${native}.so\\n`,
      `100-200 r-xp 0 08:01 99 /release/${native}.so\\n`,
      `100-200 r-xp 0 08:01 10 /release/${native}.so (deleted)\\n`,
      `100-200 r-xp 0 bad 10 /release/${native}.so\\n`,
    ]],
    ['training/tests/test_strength_freshness_cpu_collect_records.py', [`${packageName}.model-pointer`]],
    ['training/tests/test_strength_freshness_cpu_readonly.py', [packageName]],
  ];
  for (const [path, literals] of entries) {
    for (const literal of literals) {
      for (const quote of ['"', "'"]) {
        const text = `value = ${quote}${literal}${quote}`;
        assert.deepEqual(inspectBrand(path, text), [], `${path}: ${literal}`);
        for (const unrelated of ['src/app/page.tsx', 'training/tests/new_collector.py', `${path}.backup`]) {
          assert.ok(inspectBrand(unrelated, text).includes('retired brand in text'));
        }
        assert.ok(inspectBrand(path, `${quote}${literal}.unexpected${quote}`).includes('retired brand in text'));
        assert.ok(inspectBrand(path, `${quote}prefix/${literal}${quote}`).includes('retired brand in text'));
      }
    }
    assert.ok(inspectBrand(path, `new ${packageName} product`).includes('retired brand in text'));
    assert.ok(inspectBrand(path, `${prior}Count = 1`).includes('retired brand in text'));
    assert.ok(inspectBrand(path, `"${packageName}.unexpected"`).includes('retired brand in text'));
    assert.ok(inspectBrand(path, String.fromCharCode(0x2605)).includes('retired visual symbol'));
    assert.ok(inspectBrand(path, String.fromCharCode(113, 117, 97, 114, 107)).includes('retired scoring terminology'));
    assert.ok(inspectBrand(path, "'" + '*' + "10'").includes('retired coordinate notation'));
  }
});

test('collector exceptions do not transfer between compatibility files', () => {
  const native = `${prior}_native`;
  const packageName = `${prior}train`;
  assert.ok(inspectBrand('training/scripts/strength_freshness_cpu_collect_records.py', `"${native}"`).length);
  assert.ok(inspectBrand('training/scripts/strength_freshness_cpu_collect_identity.py', `"${packageName}.model-pointer"`).length);
  assert.ok(inspectBrand('training/tests/test_strength_freshness_cpu_readonly.py', `"${native}"`).length);
  assert.ok(inspectBrand('training/scripts/strength_freshness_cpu_collect_identity.py', `"/release/${native}.so"`).length);
});
