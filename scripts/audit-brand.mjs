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
// Letter/number map coordinates are current notation; only the retired symbol
// prefix identifies an obsolete coordinate unambiguously.
const oldCoordinates = /["'`]\\{0,2}\*\d{2}["'`]/;
const oldSymbols = /[\u2605\u2606\u2733-\u273c\u2726\u2727\u274b]/u;

// Historical rollout records name the actual pre-migration service and file
// identities. Preserve that evidence verbatim; this exception never covers
// shipped code, paths, scoring terminology, or visual symbols.
const historicalDeploymentEvidence = new Set([
  'training/docs/elo-efficiency-deployment-20260923.md',
  'training/docs/elo-efficiency-release-20260923.md',
  'training/docs/elo-efficiency-runtime-deployment-evidence-20260923.json',
]);

export function inspectBrand(path, contents = '') {
  const findings = [];
  if (names.test(path) || symbols.test(path)) findings.push('retired brand in path');
  if ((names.test(contents) || symbols.test(contents)) && !historicalDeploymentEvidence.has(path)) {
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
