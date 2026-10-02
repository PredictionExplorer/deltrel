import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { register } from 'node:module';
import { runtimeDescriptor, verifyLegacyBrowserArtifacts, verifyQualifiedRuntime } from './browser-runtime.mjs';

register(new URL('./typescript-loader.mjs', import.meta.url));
const { parseDeltrelBrowserModelManifest } = await import('../src/lib/deltrel/ai/manifest.ts');

/** Fail a deployment before shipping a missing, stale, or truncated AI release. */
export async function verifyBrowserRelease(projectRoot) {
  const publicRoot = resolve(projectRoot, 'public');
  verifyLegacyBrowserArtifacts(projectRoot);
  const manifest = parseDeltrelBrowserModelManifest(JSON.parse(
    readFileSync(resolve(publicRoot, 'models/deltrel', runtimeDescriptor.manifest), 'utf8'),
  ));
  const model = readFileSync(resolve(publicRoot, manifest.model.url.slice(1)));
  if (model.byteLength !== manifest.model.bytes ||
      `sha256:${createHash('sha256').update(model).digest('hex')}` !== manifest.model.sha256) {
    throw new Error('Published browser model size or SHA-256 does not match its manifest.');
  }
  // Hash both execution artifacts before importing any of their code.
  await verifyQualifiedRuntime(projectRoot);
  return { modelVersion: manifest.modelVersion, precision: manifest.model.precision, bytes: model.byteLength, outputs: manifest.model.outputs.length };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const result = await verifyBrowserRelease(fileURLToPath(new URL('..', import.meta.url)));
  console.log(`Verified browser release ${result.modelVersion}: ${result.precision}, ${result.bytes} bytes, ${result.outputs} outputs.`);
}
