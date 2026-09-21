#!/usr/bin/env node
/** Measure actual bundled Workers; no app/debug routes or substitute search code. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createReadStream, readFileSync, readdirSync, statSync, writeFileSync} from 'node:fs';
import http from 'node:http';
import {register} from 'node:module';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {chromium} from '@playwright/test';

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const args = process.argv.slice(2);
function option(name, fallback) { const i = args.indexOf(name); return i < 0 ? fallback : args[i + 1]; }
const baseline = path.resolve(option('--baseline', repository));
const candidateArg = option('--candidate', '');
const candidate = candidateArg ? path.resolve(candidateArg) : null;
const repetitions = Number(option('--repetitions', '3'));
assert(Number.isInteger(repetitions) && repetitions >= 1 && repetitions <= 10);
const budgets = option('--budgets', '8,32').split(',').map(Number);
assert(budgets.every(n => Number.isInteger(n) && n >= 1 && n <= 64));
const providers = option('--providers', 'wasm,webgpu').split(',');
assert(providers.every(p => ['wasm', 'webgpu'].includes(p)));
const isolation = option('--isolation', 'none');
assert(['none', 'coi'].includes(isolation));
const outputPath = path.resolve(option('--output', '/tmp/deltrel-browser-search-benchmark.json'));
register(pathToFileURL(path.join(repository, 'scripts/typescript-loader.mjs')));
const {buildAiRequest} = await import(pathToFileURL(path.join(repository, 'src/lib/deltrel/ai/protocol.ts')));
const {parseDeltrelAiDecision} = await import(pathToFileURL(path.join(repository, 'src/lib/deltrel/ai/decision.ts')));
const fixture = JSON.parse(readFileSync(path.join(repository, 'testdata/deltrel/conformance-v3.json'), 'utf8'));
const cases = [];
for (const rings of [4, 10]) {
  const game = fixture.games.find(g => g.id === `rings-${rings}-board-full`);
  const index = Math.floor(game.states[0].stones.length * 0.3);
  for (const simulations of budgets) {
    const request = buildAiRequest(game.config, game.actions.slice(0, index), `benchmark-${rings}-${simulations}`);
    cases.push({id: `rings-${rings}-moves-${index}-visits-${simulations}`, request,
      search: {simulations, maxConsidered: simulations <= 8 ? 4 : 8}});
  }
}
function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(k => [k, canonical(value[k])]));
  return value;
}
function semanticDigest(decision) {
  const response = {...decision.response};
  const analysis = {...decision.analysis};
  delete response.requestId;
  delete analysis.timingMs;
  return createHash('sha256').update(JSON.stringify(canonical({response, analysis}))).digest('hex');
}
function startServer(root) {
  const chunks = path.join(root, '.next/static/chunks');
  const workers = readdirSync(chunks).filter(name => /^deltrel-local-ai\..+\.js$/.test(name));
  assert.equal(workers.length, 1, `build ${root} before benchmarking`);
  const workerUrl = `/_next/static/chunks/${workers[0]}`;
  const manifest = JSON.parse(readFileSync(path.join(root, 'public/models/deltrel/manifest.json'), 'utf8'));
  const onnx = path.join(root, 'public/models/deltrel', manifest.artifacts.onnx.file);
  assert.equal(statSync(onnx).size, manifest.artifacts.onnx.bytes);
  assert.equal(createHash('sha256').update(readFileSync(onnx)).digest('hex'), manifest.artifacts.onnx.sha256);
  const server = http.createServer((request, response) => {
    const url = new URL(request.url, 'http://localhost');
    if (isolation === 'coi') {
      response.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    }
    if (url.pathname === '/') { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html><title>Deltrel worker benchmark</title>'); return; }
    if (url.pathname === '/benchmark-worker.js') {
      response.setHeader('Content-Type', 'text/javascript');
      const forceWasm = url.searchParams.get('backend') === 'wasm';
      // Test-only provider selection; the shipped worker bundle is untouched.
      response.end(`${forceWasm ? "const originalNavigator = navigator; Object.defineProperty(globalThis, 'navigator', {value: new Proxy(originalNavigator, {has: (target, key) => key === 'gpu' ? false : Reflect.has(target, key), get: (target, key) => key === 'gpu' ? undefined : Reflect.get(target, key, target)})});\n" : ''}importScripts(${JSON.stringify(workerUrl)});`);
      return;
    }
    const decoded = decodeURIComponent(url.pathname);
    const directory = decoded.startsWith('/_next/static/') ? path.join(root, '.next/static') : path.join(root, 'public');
    const relative = decoded.startsWith('/_next/static/') ? decoded.slice('/_next/static/'.length) : decoded.slice(1);
    const filename = path.resolve(directory, relative);
    if (!filename.startsWith(`${directory}${path.sep}`)) { response.writeHead(403); response.end(); return; }
    let stat;
    try { stat = statSync(filename); } catch { response.writeHead(404); response.end(); return; }
    if (!stat.isFile()) { response.writeHead(404); response.end(); return; }
    response.setHeader('Content-Length', stat.size);
    response.setHeader('Content-Type', /\.(?:js|mjs)$/.test(filename) ? 'text/javascript' : filename.endsWith('.wasm') ? 'application/wasm' : filename.endsWith('.json') ? 'application/json' : 'application/octet-stream');
    createReadStream(filename).pipe(response);
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve({root, manifest, worker: workers[0],
    workerSha256: createHash('sha256').update(readFileSync(path.join(chunks, workers[0]))).digest('hex'),
    origin: `http://127.0.0.1:${server.address().port}`, close: () => new Promise(done => server.close(done))})));
}
async function sample(page, backend, item, sequence) {
  const result = await page.evaluate(async ({backend, item, sequence}) => {
    const worker = new Worker(`/benchmark-worker.js?backend=${backend}`, {name: `benchmark-${backend}`});
    const waiters = new Map(); let readyResolve, readyReject;
    const ready = new Promise((resolve, reject) => {readyResolve = resolve; readyReject = reject;});
    worker.onerror = event => {const error = new Error(event.message); readyReject(error); for (const w of waiters.values()) w.reject(error);};
    worker.onmessage = ({data}) => {
      if (data.type === 'ready') {readyResolve(); return;}
      const waiter = waiters.get(data.taskId);
      if (!waiter) return;
      if (data.type === 'error') {waiters.delete(data.taskId); waiter.reject(new Error(`${data.error.code}: ${data.error.message}`));}
      else if (data.type === waiter.expected) {waiters.delete(data.taskId); waiter.resolve(data);}
    };
    function send(command, expected) {
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {waiters.delete(command.taskId); reject(new Error(`worker timed out: ${expected}`));}, 120_000);
        waiters.set(command.taskId, {expected, resolve: x => {clearTimeout(timer); resolve(x);}, reject: e => {clearTimeout(timer); reject(e);}});
        worker.postMessage(command);
      });
    }
    try {
      let handshakeTimer;
      try {
        await Promise.race([ready, new Promise((_, reject) => { handshakeTimer = setTimeout(() => reject(new Error('worker handshake timeout')), 10_000); })]);
      } finally { clearTimeout(handshakeTimer); }
      const start = performance.now();
      const prepared = await send({type: 'prepare', taskId: `prepare-${sequence}`}, 'prepared');
      const preparationMs = performance.now() - start;
      if (prepared.info.backend !== backend) return {skipped: true, reason: `requested ${backend}, runtime selected ${prepared.info.backend}`, ready: prepared.info, preparationMs};
      const measured = [];
      for (const workload of ['fresh-runtime', 'same-root-repeat']) {
        const taskId = `${item.id}-${sequence}-${workload}`;
        const request = {...item.request, requestId: taskId};
        const started = performance.now();
        const response = await send({type: 'choose', taskId, request, search: item.search}, 'result');
        measured.push({workload, wallMs: performance.now() - started, decision: response.decision, request});
      }
      return {ready: prepared.info, preparationMs, measured};
    } finally {worker.terminate();}
  }, {backend, item, sequence});
  if (result.skipped) return result;
  const measured = result.measured.map(row => {
    const decision = parseDeltrelAiDecision(row.request, row.decision);
    const visits = decision.analysis.rootVisits.reduce((sum, x) => sum + x, 0);
    assert.equal(visits, item.search.simulations);
    assert.equal(decision.analysis.maxConsidered, item.search.maxConsidered);
    return {workload: row.workload, wallMs: row.wallMs, inferenceSearchMs: decision.analysis.timingMs.inferenceSearch,
      modelLoadMs: decision.analysis.timingMs.modelLoad, visits, selectedAction: decision.response.action,
      outcome: decision.analysis.outcome, expectedMargin: decision.analysis.expectedMargin,
      semanticSha256: semanticDigest(decision), heads: Object.keys(decision.analysis.networkOutput?.heads ?? {}).length};
  });
  assert.equal(measured[0].semanticSha256, measured[1].semanticSha256, 'same-root cache reuse changed semantics');
  return {...result, measured};
}
function median(values) { const sorted = [...values].sort((a,b) => a-b); return (sorted[Math.floor((sorted.length-1)/2)] + sorted[Math.floor(sorted.length/2)]) / 2; }
const builds = [{label: 'baseline', root: baseline}];
if (candidate) builds.push({label: 'candidate', root: candidate});
const servers = [];
const report = {schemaVersion: 1, complete: false, createdAt: new Date().toISOString(), hardware: {platform: os.platform(), architecture: os.arch(), cpu: os.cpus()[0].model, logicalCpus: os.cpus().length},
  isolation, repetitions, cases: cases.map(({id, search, request}) => ({id, search, stateHash: request.stateHash, rings: request.state.rings, occupied: request.state.stones.filter(x=>x!==-1).length})), builds: [], samples: [], summaries: [], providerNotes: []};
try {
  for (const build of builds) {const server = await startServer(build.root); servers.push({...build, ...server}); report.builds.push({label: build.label, revision: build.label === 'baseline' ? option('--baseline-revision', 'unspecified') : option('--candidate-revision', 'working-tree'), sourceWorkerSha256: createHash('sha256').update(readFileSync(path.join(build.root, 'src/workers/deltrel-ai.worker.ts'))).digest('hex'), worker: server.worker, workerSha256: server.workerSha256,
    modelSha256: server.manifest.artifacts.onnx.sha256, modelVersion: server.manifest.model_version});}
  if (servers.length === 2) assert.equal(servers[0].manifest.artifacts.onnx.sha256, servers[1].manifest.artifacts.onnx.sha256, 'model changes invalidate speed comparison');
  for (const backend of providers) {
    const browser = await chromium.launch({headless: true, args: backend === 'wasm' ? ['--disable-gpu'] : []});
    try {
      const context = await browser.newContext();
      const pages = [];
      for (const server of servers) {const page = await context.newPage(); await page.goto(server.origin); pages.push(page);}
      report.browserVersion = browser.version();
      if (backend === 'webgpu') {
        const capability = await pages[0].evaluate(async () => {
          try {
          if (!navigator.gpu) return {available: false, reason: 'navigator.gpu unavailable'};
          const adapter = await navigator.gpu.requestAdapter();
          if (!adapter) return {available: false, reason: 'no WebGPU adapter'};
          const info = adapter.info;
          return {available: true, shaderF16: adapter.features.has('shader-f16'), vendor: info?.vendor, architecture: info?.architecture, device: info?.device, description: info?.description, fallback: info?.isFallbackAdapter ?? adapter.isFallbackAdapter};
          } catch (error) { return {available: false, reason: String(error)}; }
        });
        report.providerNotes.push({backend, capability});
        if (!capability.available) continue;
      }
      let providerSkipped = false;
      for (const item of cases) {
        for (let repetition = 0; repetition < repetitions; repetition++) {
          for (let order = 0; order < servers.length; order++) {
            const i = (order + repetition) % servers.length;
            const data = await sample(pages[i], backend, item, `${repetition}-${i}`);
            report.samples.push({build: servers[i].label, backend, case: item.id, repetition, ...data});
            process.stderr.write(`${servers[i].label} ${backend} ${item.id} #${repetition + 1}: ${data.skipped ? data.reason : data.measured.map(x => `${x.workload}=${x.wallMs.toFixed(1)}ms`).join(', ')}\n`);
            writeFileSync(outputPath, JSON.stringify(report, null, 2));
            if (data.skipped) {providerSkipped = true; break;}
          }
          if (providerSkipped) break;
        }
        if (providerSkipped) break;
      }
      await context.close();
    } finally {await browser.close();}
  }
  for (const build of builds) for (const backend of providers) for (const item of cases) for (const workload of ['fresh-runtime', 'same-root-repeat']) {
    const rows = report.samples.filter(x => x.build === build.label && x.backend === backend && x.case === item.id && !x.skipped).flatMap(x => x.measured.filter(y => y.workload === workload));
    if (!rows.length) continue;
    const hashes = [...new Set(rows.map(x => x.semanticSha256))];
    assert.equal(hashes.length, 1, 'same fixed search produced inconsistent semantics');
    report.summaries.push({build: build.label, backend, case: item.id, workload, samples: rows.length, medianWallMs: median(rows.map(x => x.wallMs)), medianInferenceSearchMs: median(rows.map(x => x.inferenceSearchMs)), semanticSha256: hashes[0], visits: item.search.simulations});
  }
  if (candidate) {
    for (const row of report.summaries.filter(x => x.build === 'candidate')) {
      const prior = report.summaries.find(x => x.build === 'baseline' && x.backend === row.backend && x.case === row.case && x.workload === row.workload);
      if (!prior) continue;
      assert.equal(row.semanticSha256, prior.semanticSha256, 'optimization changed decisions, search statistics, or network outputs');
      row.speedup = prior.medianWallMs / row.medianWallMs;
      row.reductionPercent = 100 * (1-row.medianWallMs/prior.medianWallMs);
    }
  }
  report.complete = true;
  writeFileSync(outputPath, JSON.stringify(report, null, 2));
  console.log(JSON.stringify({output: outputPath, summaries: report.summaries, providers: report.providerNotes}, null, 2));
} finally {for (const server of servers) await server.close();}
