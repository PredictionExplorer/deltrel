import { spawnSync } from 'node:child_process';
import {
  existsSync,
  mkdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { register } from 'node:module';

register(new URL('./typescript-loader.mjs', import.meta.url));
const { DELTREL_RULES_HASH, DELTREL_RULES_SCHEMA_ID } =
  await import('../src/lib/deltrel/rules.ts');

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const crate = resolve(root, 'training/crates/deltrel-wasm');
const wasmDirectory = `wasm-${DELTREL_RULES_HASH.split(':')[1]}-champion-v1`;
const output = resolve(root, `public/models/deltrel/${wasmDirectory}`);
const outputFromCrate = relative(crate, output);

if (!existsSync(resolve(crate, 'Cargo.toml'))) {
  throw new Error(`deltrel-wasm crate not found at ${crate}`);
}

mkdirSync(dirname(output), { recursive: true });
rmSync(output, { recursive: true, force: true });

const result = spawnSync(
  'wasm-pack',
  [
    'build',
    '.',
    '--target',
    'web',
    '--release',
    '--out-dir',
    outputFromCrate,
    '--out-name',
    'deltrel_wasm',
    '--no-pack',
    '--',
    '--locked',
  ],
  {
    cwd: crate,
    env: {
      ...process.env,
      SOURCE_DATE_EPOCH: process.env.SOURCE_DATE_EPOCH ?? '0',
    },
    stdio: 'inherit',
  },
);

if (result.error) throw result.error;
if (result.status !== 0) {
  throw new Error(`wasm-pack exited with status ${String(result.status)}`);
}

for (const filename of ['deltrel_wasm.js', 'deltrel_wasm_bg.wasm']) {
  if (!existsSync(resolve(output, filename))) {
    throw new Error(`wasm-pack did not produce ${filename}`);
  }
}

// Ship the generated rules/search package with the website. wasm-pack's
// blanket ignore would otherwise hide the files from Git-based deployments.
for (const filename of ['.gitignore', 'deltrel_wasm.d.ts', 'deltrel_wasm_bg.wasm.d.ts']) {
  rmSync(resolve(output, filename), { force: true });
}

writeFileSync(
  resolve(output, 'contract.json'),
  `${JSON.stringify(
    {
      schema: 'deltrel.browser-wasm-build.v3',
      rulesSchema: DELTREL_RULES_SCHEMA_ID,
      rulesHash: DELTREL_RULES_HASH,
      moduleUrl: `/models/deltrel/${wasmDirectory}/deltrel_wasm.js`,
      binaryUrl: `/models/deltrel/${wasmDirectory}/deltrel_wasm_bg.wasm`,
      modelManifestUrl: '/models/deltrel/manifest.json',
    },
    null,
    2,
  )}\n`,
);

console.log(`Built Deltrel WASM assets in public/models/deltrel/${wasmDirectory}`);
console.log('Browser model manifest convention: public/models/deltrel/manifest.json');
