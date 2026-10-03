import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, realpathSync } from 'node:fs';
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { verifyLegacyBrowserArtifacts, verifyQualifiedRuntime } from './browser-runtime.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');

/** Default builds verify shipped bytes; development compilation requires a new private directory. */
export async function buildDeltrelWasm(args, projectRoot = root, spawn = spawnSync) {
  if (args.length === 0) {
    verifyLegacyBrowserArtifacts(projectRoot);
    return verifyQualifiedRuntime(projectRoot);
  }
  if (args.length !== 2 || args[0] !== '--out-dir') {
    throw new Error('Usage: build-deltrel-wasm.mjs [--out-dir NEW_PRIVATE_DIRECTORY]');
  }
  const output = resolve(projectRoot, args[1]);
  const publicRoot = resolve(projectRoot, 'public');
  let ancestor = dirname(output);
  while (!existsSync(ancestor)) ancestor = dirname(ancestor);
  const physicalOutput = resolve(realpathSync(ancestor), relative(ancestor, output));
  const physicalPublic = existsSync(publicRoot) ? realpathSync(publicRoot) : publicRoot;
  const insidePublic = relative(physicalPublic, physicalOutput);
  const parentTraversal = insidePublic === '..' || insidePublic.startsWith(`..${sep}`);
  if (insidePublic === '' || (!parentTraversal && !isAbsolute(insidePublic))) {
    throw new Error('Development WASM output must be outside public; immutable assets are never rebuilt in place.');
  }
  if (existsSync(output)) throw new Error('Development WASM output must be a new directory.');
  const crate = resolve(projectRoot, 'training/crates/deltrel-wasm');
  if (!existsSync(resolve(crate, 'Cargo.toml'))) throw new Error('Deltrel WASM crate is missing.');
  mkdirSync(dirname(output), { recursive: true });
  const result = spawn('wasm-pack', ['build', '.', '--target', 'web', '--release',
    '--out-dir', relative(crate, output), '--out-name', 'deltrel_wasm', '--no-pack', '--', '--locked'], {
    cwd: crate, env: { ...process.env, SOURCE_DATE_EPOCH: process.env.SOURCE_DATE_EPOCH ?? '0' }, stdio: 'inherit',
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`wasm-pack exited with status ${String(result.status)}`);
  return { output, qualifiedForPublication: false };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  console.log(JSON.stringify(await buildDeltrelWasm(process.argv.slice(2))));
}
