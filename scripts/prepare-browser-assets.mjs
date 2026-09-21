import { copyFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { register } from 'node:module';

register(new URL('./typescript-loader.mjs', import.meta.url));
const {
  DELTREL_ORT_VERSION,
  DELTREL_ORT_ASSET_PREFIX,
  DELTREL_ORT_ASSET_FILES,
} = await import('../src/lib/deltrel/ai/runtime-assets.ts');

/** Copy the exact installed runtime; visitors never need npm or a separate CDN. */
export function prepareBrowserAssets(projectRoot) {
  const packageRoot = resolve(projectRoot, 'node_modules/onnxruntime-web');
  const { version } = JSON.parse(readFileSync(resolve(packageRoot, 'package.json'), 'utf8'));
  if (version !== DELTREL_ORT_VERSION) {
    throw new Error(`ONNX Runtime ${version} does not match web asset version ${DELTREL_ORT_VERSION}. Update runtime-assets.ts when upgrading the dependency.`);
  }
  const files = DELTREL_ORT_ASSET_FILES.map((name) => ({
    name,
    source: resolve(packageRoot, 'dist', name),
  }));
  // Check the complete set before copying, so a partial package cannot publish.
  for (const { source } of files) {
    if (!existsSync(source)) throw new Error(`Required ONNX Runtime asset is missing: ${source}`);
  }
  const destination = resolve(projectRoot, 'public', DELTREL_ORT_ASSET_PREFIX.slice(1));
  mkdirSync(destination, { recursive: true });
  for (const { name, source } of files) {
    copyFileSync(source, resolve(destination, name));
  }
  return { version, destination, files: files.map(({ name }) => name) };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
  const result = prepareBrowserAssets(root);
  console.log(`Prepared ONNX Runtime ${result.version}: ${result.files.length} browser assets.`);
}
