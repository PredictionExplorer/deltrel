import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { register } from 'node:module';

register(new URL('./typescript-loader.mjs', import.meta.url));
const { parseDeltrelBrowserModelManifest } = await import('../src/lib/deltrel/ai/manifest.ts');

/** Fail a deployment before shipping a missing, stale, or truncated AI release. */
export async function verifyBrowserRelease(projectRoot) {
  const publicRoot = resolve(projectRoot, 'public');
  const manifest = parseDeltrelBrowserModelManifest(JSON.parse(
    readFileSync(resolve(publicRoot, 'models/deltrel/manifest.json'), 'utf8'),
  ));
  const model = readFileSync(resolve(publicRoot, manifest.model.url.slice(1)));
  if (model.byteLength !== manifest.model.bytes ||
      `sha256:${createHash('sha256').update(model).digest('hex')}` !== manifest.model.sha256) {
    throw new Error('Published browser model size or SHA-256 does not match its manifest.');
  }
  const moduleSource = readFileSync(resolve(publicRoot, manifest.wasm.moduleUrl.slice(1)), 'utf8');
  const binary = readFileSync(resolve(publicRoot, manifest.wasm.binaryUrl.slice(1)));
  if (!moduleSource.includes('WasmState') || !WebAssembly.validate(binary)) {
    throw new Error('Published browser rules/search package is invalid.');
  }
  const wasm = await import(pathToFileURL(resolve(publicRoot, manifest.wasm.moduleUrl.slice(1))).href);
  await wasm.default({ module_or_path: binary });
  if (wasm.WasmState.rules_hash_tag() !== manifest.rulesHash || wasm.WasmState.rules_schema() !== manifest.rulesSchema) {
    throw new Error('Published browser rules and model contracts do not match.');
  }
  return { modelVersion: manifest.modelVersion, bytes: model.byteLength, outputs: manifest.model.outputs.length };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const result = await verifyBrowserRelease(fileURLToPath(new URL('..', import.meta.url)));
  console.log(`Verified browser release ${result.modelVersion}: ${result.bytes} bytes, ${result.outputs} outputs.`);
}
