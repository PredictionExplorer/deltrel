/**
 * Export deterministic Deltrel vectors for cross-language parity.
 *
 * Usage:
 *   node scripts/export-deltrel-conformance.mjs [output-path]
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { serializeDeltrelConformance } from '../src/lib/deltrel/conformance';
import {
  fnv1a64,
  DELTREL_RULES_CANONICAL,
  DELTREL_RULES_HASH,
  DELTREL_RULES_HASH_ALGORITHM,
} from '../src/lib/deltrel/rules';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(scriptDirectory, '..');
const outputPath = process.argv[2]
  ? resolve(process.cwd(), process.argv[2])
  : resolve(repositoryRoot, 'testdata/deltrel/conformance-v3.json');

const actualHash =
  `${DELTREL_RULES_HASH_ALGORITHM}:${fnv1a64(DELTREL_RULES_CANONICAL)}`;
if (actualHash !== DELTREL_RULES_HASH) {
  throw new Error(
    `stale DELTREL_RULES_HASH: expected ${actualHash}, found ${DELTREL_RULES_HASH}`,
  );
}

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, serializeDeltrelConformance(), 'utf8');
console.log(`Wrote ${outputPath}`);
