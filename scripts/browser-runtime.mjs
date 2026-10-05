import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import descriptor from '../training/deltreltrain/browser_runtime.json' with { type: 'json' };

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const files = ['module', 'binary'].map(key => ({
  path: descriptor[key].path, sha256: descriptor[key].sha256, bytes: descriptor[key].bytes,
}));
if (hash(Buffer.from(JSON.stringify(files))) !== descriptor.runtime_content_sha256) {
  throw new Error('Browser runtime descriptor content hash is invalid.');
}
export const runtimeDescriptor = Object.freeze({
  ...descriptor,
  directory: `wasm-${descriptor.rules_hash.slice('fnv1a64:'.length)}-${descriptor.runtime_content_sha256}`,
  manifest: `manifest-runtime-${descriptor.runtime_content_sha256}.json`,
});

export function verifiedBytes(path, expected) {
  const bytes = readFileSync(path);
  if (bytes.byteLength !== expected.bytes || hash(bytes) !== expected.sha256) {
    throw new Error(`Browser artifact size/SHA-256 mismatch: ${path}`);
  }
  return bytes;
}

/** Existing deployed pages retain their own model and execution package. */
export function verifyLegacyBrowserArtifacts(projectRoot) {
  const base = resolve(projectRoot, 'public/models/deltrel');
  const { legacy } = descriptor;
  for (const key of ['manifest', 'model']) {
    verifiedBytes(resolve(base, legacy[key].path), legacy[key]);
  }
  for (const key of ['module', 'binary']) {
    verifiedBytes(resolve(base, legacy.directory, legacy[key].path), legacy[key]);
  }
  return true;
}

export async function verifyQualifiedRuntime(projectRoot) {
  const base = resolve(projectRoot, 'public/models/deltrel', runtimeDescriptor.directory);
  const modulePath = resolve(base, descriptor.module.path);
  const javascript = verifiedBytes(modulePath, descriptor.module);
  const binary = verifiedBytes(resolve(base, descriptor.binary.path), descriptor.binary);
  if (!WebAssembly.validate(binary)) throw new Error('Invalid qualified browser WASM.');
  // Execute the verified snapshot, not a second filesystem read that could race
  // a writer. Like the browser's Blob import, initialization supplies WASM bytes.
  const wasm = await import(`data:text/javascript;base64,${javascript.toString('base64')}`);
  await wasm.default({ module_or_path: binary });
  if (wasm.search_algorithm_id?.() !== descriptor.search_algorithm ||
      wasm.WasmState.rules_hash_tag() !== descriptor.rules_hash ||
      wasm.WasmState.rules_schema() !== descriptor.rules_schema ||
      wasm.search_execution_version?.() !== 1 ||
      wasm.derive_root_seed?.(0n, 0n, 0) !== 0xe220a8397b1dcdafn) {
    throw new Error('Qualified browser runtime has incompatible execution contracts.');
  }
  return { identity: descriptor.runtime_content_sha256, search: wasm.search_algorithm_id() };
}
