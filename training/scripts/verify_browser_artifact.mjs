#!/usr/bin/env node
/** Real CPU/WASM browser inference against the published, checksum-verified model. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createReadStream, readFileSync, statSync} from 'node:fs';
import http from 'node:http';
import {register} from 'node:module';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {chromium} from '@playwright/test';

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
register(pathToFileURL(path.join(repository, 'scripts/typescript-loader.mjs')));
const {buildAiRequest} = await import(pathToFileURL(path.join(repository, 'src/lib/deltrel/ai/protocol.ts')));
const {encodeDeltrelFeatures, float32ToFloat16Array} = await import(pathToFileURL(path.join(repository, 'src/lib/deltrel/ai/features.ts')));
const {parseDeltrelBrowserModelManifest} = await import(pathToFileURL(path.join(repository, 'src/lib/deltrel/ai/manifest.ts')));
const manifestPath = path.resolve(process.argv[2] ?? path.join(repository, 'public/models/deltrel/manifest.json'));
const payload = JSON.parse(readFileSync(manifestPath, 'utf8'));
const manifest = parseDeltrelBrowserModelManifest(payload);
const checksum = manifest.model.sha256.replace(/^sha256:/, '');
const modelPath = path.join(path.dirname(manifestPath), payload.artifacts.onnx.file);
assert.equal(statSync(modelPath).size, manifest.model.bytes);
assert.equal(createHash('sha256').update(readFileSync(modelPath)).digest('hex'), checksum);
const fixture = JSON.parse(readFileSync(path.join(repository, 'testdata/deltrel/conformance-v3.json'), 'utf8'));
const cases = [];
for (const game of fixture.games) {
  for (const index of [1, 2, 3, game.states.length - 2]) {
    const request = buildAiRequest(game.config, game.actions.slice(0, index), `browser-${cases.length}`);
    const encoded = encodeDeltrelFeatures(request.state), n = encoded.nodeCount, d = encoded.maxDegree;
    const tensor = (type, data, dims) => ({type, data: Array.from(data, Number), dims});
    cases.push({name: `${game.id}:${index}`, rings: game.config.rings, nodes: n, legal: request.legalActions, feeds: {
      node_features: tensor('float16', float32ToFloat16Array(encoded.nodeFeatures), [1, n, 19]),
      global_features: tensor('float16', float32ToFloat16Array(encoded.globalFeatures), [1, 25]),
      neighbor_index: tensor('int64', encoded.neighborIndex, [1, n, d]),
      neighbor_mask: tensor('bool', encoded.neighborMask, [1, n, d]),
      neighbor_edge_type: tensor('int64', encoded.neighborEdgeType, [1, n, d]),
      node_mask: tensor('bool', encoded.nodeMask, [1, n]),
      legal_action_mask: tensor('bool', encoded.legalActionMask, [1, n]),
      rings: tensor('int64', encoded.rings, [1]),
    }});
  }
}
const ortDirectory = path.join(repository, 'node_modules/onnxruntime-web/dist');
const server = http.createServer((request, response) => {
  const url = new URL(request.url, 'http://localhost');
  response.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  if (url.pathname === '/') {
    response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html><title>Browser model verification</title>'); return;
  }
  if (url.pathname === '/cases.json') {
    response.setHeader('Content-Type', 'application/json'); response.end(JSON.stringify(cases)); return;
  }
  const runtimeName = path.basename(url.pathname);
  if (url.pathname !== '/model.onnx' && (!url.pathname.startsWith('/ort/') || !/^[A-Za-z0-9._-]+\.(?:mjs|js|wasm)$/.test(runtimeName))) {
    response.writeHead(404); response.end(); return;
  }
  const filename = url.pathname === '/model.onnx' ? modelPath : path.join(ortDirectory, runtimeName);
  let size;
  try { size = statSync(filename).size; } catch { response.writeHead(404); response.end(); return; }
  response.setHeader('Content-Length', size);
  response.setHeader('Content-Type', filename.endsWith('.wasm') ? 'application/wasm' : /\.(?:mjs|js)$/.test(filename) ? 'text/javascript' : 'application/octet-stream');
  createReadStream(filename).pipe(response);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch({headless: true, args: ['--disable-gpu']});
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const report = await page.evaluate(async ({expectedNames, expectedBytes, expectedHash}) => {
    const ort = await import('/ort/ort.webgpu.min.mjs');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = `${location.origin}/ort/`;
    const started = performance.now();
    const bytes = await (await fetch('/model.onnx')).arrayBuffer();
    const checksum = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
    if (bytes.byteLength !== expectedBytes || checksum !== expectedHash) throw new Error('browser model integrity mismatch');
    const session = await ort.InferenceSession.create(bytes, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
    const loadMilliseconds = performance.now() - started;
    const float16 = bits => {
      const sign = bits & 32768 ? -1 : 1, exponent = (bits >> 10) & 31, fraction = bits & 1023;
      return exponent === 0 ? sign * fraction * 2 ** -24 : exponent === 31 ? (fraction ? NaN : sign * Infinity) : sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
    };
    const records = await (await fetch('/cases.json')).json();
    const reports = [];
    for (const row of records) {
      const feeds = {};
      for (const [name, input] of Object.entries(row.feeds)) {
        const data = input.type === 'int64' ? new BigInt64Array(input.data.map(BigInt)) : input.type === 'bool' ? new Uint8Array(input.data) : new Uint16Array(input.data);
        feeds[name] = new ort.Tensor(input.type, data, input.dims);
      }
      const start = performance.now();
      const outputs = await session.run(feeds);
      if (Object.keys(outputs).sort().join('|') !== [...expectedNames].sort().join('|')) throw new Error(`head set mismatch: ${row.name}`);
      for (const [name, tensor] of Object.entries(outputs)) {
        if (tensor.type !== 'float16' || !Array.from(tensor.data, tensor.data instanceof Uint16Array ? float16 : Number).every(Number.isFinite)) throw new Error(`invalid ${name}: ${row.name}; ${tensor.data.constructor.name}; ${Array.from(tensor.data).slice(0, 5)}`);
      }
      const policy = Array.from(outputs.policy_logits.data, outputs.policy_logits.data instanceof Uint16Array ? float16 : Number);
      const selected = row.legal.reduce((best, node) => policy[node] > policy[best] ? node : best, row.legal[0]);
      if (!Number.isInteger(selected) || selected < 0 || selected >= row.nodes) throw new Error('invalid model placement');
      reports.push({case: row.name, rings: row.rings, milliseconds: performance.now() - start, legalMove: selected, heads: expectedNames.length});
      for (const tensor of [...Object.values(feeds), ...Object.values(outputs)]) tensor.dispose();
    }
    await session.release();
    return {provider: 'wasm', loadMilliseconds, positions: reports.length, reports};
  }, {expectedNames: manifest.model.outputs, expectedBytes: manifest.model.bytes, expectedHash: checksum});
  console.log(JSON.stringify({modelVersion: manifest.modelVersion, bytes: manifest.model.bytes, ...report}, null, 2));
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
