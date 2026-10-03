import { execFileSync } from 'node:child_process';
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// Keep retired branding out of the shipped source, including the audit itself.
const retired = String.fromCharCode(115, 116, 97, 114);
const title = retired[0].toUpperCase() + retired.slice(1);
const names = new RegExp(
  `(?:^|[^a-z])${retired}(?:$|[^a-z]|s(?:$|[^a-z])|ai|board|field|train|serve)`,
  'im',
);
const symbols = new RegExp(`(?:${title}|${retired})(?=[A-Z])`);
const oldTerms = new RegExp(String.fromCharCode(113, 117, 97, 114, 107), 'i');
// Current polar addresses use A–E; only the retired symbol prefix identifies
// obsolete coordinates unambiguously.
const oldCoordinates = /["'`]\\{0,2}\*\d{2}["'`]/;
const oldSymbols = /[\u2605\u2606\u2733-\u273c\u2726\u2727\u274b]/u;

// Historical rollout records name the actual pre-migration service and file
// identities. Preserve that evidence verbatim; this exception never covers
// shipped code, paths, scoring terminology, or visual symbols.
const historicalDeploymentEvidence = new Set([
  'training/docs/elo-efficiency-deployment-20260923.md',
  'training/docs/elo-efficiency-release-20260923.md',
  'training/docs/elo-efficiency-runtime-deployment-evidence-20260923.json',
  'training/docs/training-recovery-deployment-20260928.md',
  'training/docs/training-recovery-deployment-evidence-20260928.json',
  'training/docs/strength-recovery-deployment-20261002.md',
]);

// The explicit migration boundary must recognize exact historical wire values.
// Exempt only quoted protocol identifiers and the two sentences documenting
// that boundary, never whole implementation/test files or new product text.
const migrationBoundaryFiles = new Set([
  'training/deltreltrain/champion_migration.py',
  'training/tests/test_champion_migration.py',
]);
const predecessorPackage = `${retired}train`;
const predecessorIdentifiers = [
  `${predecessorPackage}.checkpoint`,
  `${predecessorPackage}.model-manifest`,
  `${predecessorPackage}.model-pointer`,
  `edgeconnect.${retired}.rules.v3`,
  `edgeconnect.${retired}.action-layout.nodes-only.v1`,
];

// Read-only collectors must recognize the immutable predecessor runtime. Keep
// exact quoted native/wire values and complete fixture literals scoped to their
// compatibility files; other identifiers, prose and symbols still fail.
const predecessorNative = `${retired}_native`;
const collectorCompatibilityLiterals = new Map([
  ['training/scripts/strength_freshness_cpu_collect_identity.py', [predecessorNative]],
  ['training/scripts/strength_freshness_cpu_collect_records.py', [`${predecessorPackage}.model-pointer`]],
  ['training/tests/test_strength_freshness_cpu_collect_identity.py', [
    predecessorPackage,
    predecessorNative,
    `/release/${predecessorNative}.so`,
    `/release/bin/python\\0-m\\0${predecessorPackage}.worker\\0`,
    `100-200 r-xp 0 08:01 10 /release/${predecessorNative}.so\\n`,
    `100-200 r-xp 0 08:01 99 /release/${predecessorNative}.so\\n`,
    `100-200 r-xp 0 08:01 10 /release/${predecessorNative}.so (deleted)\\n`,
    `100-200 r-xp 0 bad 10 /release/${predecessorNative}.so\\n`,
  ]],
  ['training/tests/test_strength_freshness_cpu_collect_records.py', [`${predecessorPackage}.model-pointer`]],
  ['training/tests/test_strength_freshness_cpu_readonly.py', [predecessorPackage]],
  ['training/scripts/strength_freshness_cpu_capture_cli.py', [predecessorPackage]],
  ['training/scripts/strength_freshness_cpu_outer_runtime.py', [predecessorPackage]],
  ['training/tests/test_strength_freshness_cpu_capture_cli.py', [predecessorPackage, predecessorNative]],
]);

function migrationBrandText(path, contents) {
  let inspected = contents;
  for (const literal of collectorCompatibilityLiterals.get(path) ?? []) {
    for (const quote of ['"', "'"]) {
      inspected = inspected.replaceAll(`${quote}${literal}${quote}`, `${quote}legacy-runtime-identity${quote}`);
    }
  }
  if (migrationBoundaryFiles.has(path)) {
    const identifiers = path === 'training/tests/test_champion_migration.py'
      ? [...predecessorIdentifiers, `${predecessorPackage}.`]
      : predecessorIdentifiers;
    for (const identifier of identifiers) {
      for (const quote of ['"', "'"]) {
        inspected = inspected.replaceAll(`${quote}${identifier}${quote}`, `${quote}legacy-wire-identity${quote}`);
      }
    }
  }
  if (path === 'training/deltreltrain/champion_migration.py') {
    inspected = inspected.replaceAll(
      `Supports the immediately preceding ${title}Train publication layout only.`,
      'Supports the immediately preceding publication layout only.',
    );
  }
  if (path === 'training/docs/deltrel-rebrand.md') {
    inspected = inspected.replaceAll(
      `from a frozen ${title}Train champion and a separately verified, conclusive promotion`,
      'from a frozen preceding champion and a separately verified, conclusive promotion',
    );
  }
  return inspected;
}

export function inspectBrand(path, contents = '') {
  const findings = [];
  if (names.test(path) || symbols.test(path)) findings.push('retired brand in path');
  const brandText = migrationBrandText(path, contents);
  if ((names.test(brandText) || symbols.test(brandText)) && !historicalDeploymentEvidence.has(path)) {
    findings.push('retired brand in text');
  }
  if (oldTerms.test(contents)) findings.push('retired scoring terminology');
  if (oldCoordinates.test(contents)) findings.push('retired coordinate notation');
  if (oldSymbols.test(contents)) findings.push('retired visual symbol');
  return findings;
}

export function auditBrand(root) {
  const files = new Set(execFileSync('git', [
    'ls-files', '--cached', '--others', '--exclude-standard', '-z',
  ], { cwd: root, encoding: 'utf8' }).split('\0').filter(Boolean));
  const failures = [];
  let checked = 0;
  for (const path of files) {
    const absolute = resolve(root, path);
    if (!existsSync(absolute)) continue;
    const bytes = readFileSync(absolute);
    const contents = bytes.includes(0) ? '' : bytes.toString('utf8');
    const findings = inspectBrand(path, contents);
    if (findings.length) failures.push(`${path}: ${findings.join(', ')}`);
    checked++;
  }
  return { checked, failures };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
  const { checked, failures } = auditBrand(root);
  if (failures.length) {
    console.error(failures.join('\n'));
    process.exitCode = 1;
  } else {
    console.log(`Deltrel brand audit passed: ${checked} project files, paths, notation, and text.`);
  }
}
