import { beforeAll, describe, expect, it, vi } from 'vitest';
import * as ort from 'onnxruntime-web';
import { buildAiRequest, codeToAction, type DeltrelAiRequest } from '../protocol';
import { encodeDeltrelFeatures, float16ToFloat32Array, float32ToFloat16Array } from '../features';
import { validateNetworkOutputState } from '../network-output';
import type { LocalAiSearchProgress } from '../worker-protocol';

let worker: typeof import('@/workers/deltrel-ai.worker');
beforeAll(async () => {
  vi.stubGlobal('postMessage', vi.fn());
  vi.stubGlobal('addEventListener', vi.fn());
  worker = await import('@/workers/deltrel-ai.worker');
});
const config = { rings: 4, mode: 'double' as const, pieRule: false, playerNames: ['A', 'B'] as [string, string] };
const request = () => buildAiRequest(config, []);
const nextRequest = (root: DeltrelAiRequest, node: number) => buildAiRequest(config,
  [...root.actionLog.map((action) => codeToAction(action, 50)), { type: 'place', node }]);

function state(root: DeltrelAiRequest) {
  const bits = (nodes: number[]) => {
    const words = new BigUint64Array(5);
    nodes.forEach((node) => { words[Math.floor(node / 64)] |= BigInt(1) << BigInt(node % 64); });
    return words;
  };
  const semantic = root.state;
  return {
    request: root, to_move: semantic.toMove, moves_left: semantic.movesLeft,
    opening: semantic.opening, terminal: semantic.terminal, mode: semantic.mode,
    handicap: semantic.handicap, pie: semantic.pie, swap_available: semantic.swapAvailable,
    swapped: semantic.swapped,
    zero_bits: () => bits(semantic.stones.flatMap((owner, node) => owner === 0 ? [node] : [])),
    one_bits: () => bits(semantic.stones.flatMap((owner, node) => owner === 1 ? [node] : [])),
    current_turn_bits: () => bits(semantic.history.currentTurn),
    previous_turn_bits: () => bits(semantic.history.previousTurn),
    own_previous_turn_bits: () => bits(semantic.history.ownPreviousTurn),
    handicap_bits: () => bits(semantic.history.handicapStones),
    legal_actions: () => Int32Array.from(root.legalActions),
    hash64: () => BigInt(`0x${root.stateHash.slice('zobrist64:'.length)}`),
    apply: vi.fn(), swap: vi.fn(), free: vi.fn(),
  };
}

function fixture(auxiliary = false) {
  const tensors: Array<ReturnType<typeof vi.spyOn>> = [];
  const outputDisposals = vi.fn();
  const forwards: Array<Record<string, { dims: readonly number[]; data: ArrayLike<number | bigint> }>> = [];
  const run = vi.fn(async (feeds: Record<string, ort.Tensor>) => {
    const captured: Record<string, { dims: readonly number[]; data: ArrayLike<number | bigint> }> = {};
    for (const [name, tensor] of Object.entries(feeds)) {
      captured[name] = { dims: [...tensor.dims], data: (tensor.data as Uint16Array).slice() };
      tensors.push(vi.spyOn(tensor, 'dispose'));
    }
    forwards.push(captured);
    const batch = feeds.node_features.dims[0];
    const nodes = feeds.node_features.dims[1];
    const policy = new Float32Array(batch * nodes);
    const outcome = new Float32Array(batch * 2);
    const legal = feeds.legal_action_mask.data as Uint8Array;
    for (let row = 0; row < batch; row++) {
      const firstStone = Array.from(legal.slice(row * nodes, (row + 1) * nodes)).indexOf(0);
      const marker = firstStone + 2;
      for (let node = 0; node < nodes; node++) policy[row * nodes + node] = marker * 10 + node;
      outcome[row * 2 + 1] = marker;
    }
    const output = (values: Float32Array, dims: number[]) => {
      const data = float32ToFloat16Array(values);
      return { data, dims, dispose: () => { data.fill(0); outputDisposals(); } };
    };
    const base = {
      policy_logits: output(policy, [batch, nodes]), outcome_logits: output(outcome, [batch, 2]),
      score_margin_logits: output(new Float32Array(batch * 303), [batch, 303]),
      ownership_logits: output(new Float32Array(batch * nodes * 3), [batch, nodes, 3]),
      alive_logits: output(new Float32Array(batch * nodes), [batch, nodes]),
      soft_policy_logits: output(policy.slice(), [batch, nodes]),
    };
    return auxiliary ? { ...base,
      opponent_reply_logits: output(new Float32Array(batch * (nodes + 1)), [batch, nodes + 1]),
      second_stone_logits: output(new Float32Array(batch * nodes), [batch, nodes]),
      final_shores_logits: output(new Float32Array(batch * 102), [batch, 2, 51]),
      final_networks_logits: output(new Float32Array(batch * 52), [batch, 2, 26]),
      final_capes_logits: output(new Float32Array(batch * 12), [batch, 2, 6]),
    } : base;
  });
  const sessions: FakeSession[] = [];
  class FakeSession {
    root: ReturnType<typeof state>;
    budget: number;
    finished = false;
    token = BigInt(100);
    inherited = 0;
    free = vi.fn();
    restarts = vi.fn();
    submissions = vi.fn();
    pendingStates: Array<ReturnType<typeof state>> = [];
    constructor(root: ReturnType<typeof state>, simulations: number) {
      this.root = root; this.budget = simulations; sessions.push(this);
    }
    restart(root: ReturnType<typeof state>, simulations: number) {
      this.restarts(root, simulations);
      this.inherited = this.root.hash64() !== root.hash64() ? 3 : 0;
      this.root = root; this.budget = simulations; this.finished = false; this.token += BigInt(10);
    }
    root_actions = () => this.root.legal_actions();
    root_token = () => this.token;
    initialize_root = vi.fn();
    next_requests = () => 2;
    pending_tokens = () => BigUint64Array.from([this.token + BigInt(1), this.token + BigInt(2)]);
    pending_state = (row: number) => {
      const pending = state(nextRequest(this.root.request, this.root.request.legalActions[row]));
      this.pendingStates.push(pending);
      return pending;
    };
    pending_actions = (row: number) => Int32Array.from(nextRequest(this.root.request, this.root.request.legalActions[row]).legalActions);
    submit(tokens: BigUint64Array, values: Float32Array, offsets: Uint32Array, logits: Float32Array) {
      expect(Array.from(tokens)).toEqual(Array.from(this.pending_tokens()));
      expect(offsets.length).toBe(3);
      expect(values.length).toBe(2);
      expect(offsets[2]).toBe(logits.length);
      this.submissions(tokens, values, offsets, logits); this.finished = true;
    }
    done = () => this.finished;
    simulations = () => this.finished ? this.budget : 0;
    unique_nodes = () => 4;
    complete = vi.fn();
    selected_action = () => this.root.request.legalActions[0];
    selected_action_value = () => 0.5;
    root_value = () => 0.5;
    actions = () => this.root_actions();
    visits = () => Uint32Array.from(this.root.request.legalActions, (_, index) => index === 0 ? this.budget : 0);
    inherited_visits = () => Uint32Array.from(this.root.request.legalActions, (_, index) => index === 0 ? this.inherited : 0);
    total_visits = () => Uint32Array.from(this.visits(), (value, index) => value + this.inherited_visits()[index]);
    q_values = () => Float32Array.from(this.root.request.legalActions, (_, index) => index === 0 ? 0.5 : 0);
    policy_target = () => Float32Array.from(this.root.request.legalActions, (_, index) => index === 0 ? 1 : 0);
    reused_visits = () => this.inherited;
    reused_nodes = () => this.inherited ? 2 : 0;
  }
  const runtime = {
    manifest: { model: { sha256: 'model-a' }, featureSchemaHash: 'features-a',
      auxiliaryStatus: (auxiliary ? 'ready' : 'absent') as 'ready' | 'untrained' | 'absent',
      search: { firstVisitBatchSize: 2, subtreeReuse: true, subtreeReuseMaxNodes: 4096,
        cVisit: 50, cScale: 1, swapDeadZone: 0.02 } },
    ort, session: { run }, predictions: new worker.PredictionCache(),
    wasm: { search_execution_version: () => 1, WasmSearchSession: FakeSession },
    completedSearch: undefined as { session: FakeSession; context: string } | undefined,
  };
  const search = (root: DeltrelAiRequest, check = () => {}, onProgress?: (progress: LocalAiSearchProgress) => void) => worker.runSessionSearch(
    runtime as never, state(root), root.state, { simulations: 2, maxConsidered: 2 }, check, async () => {}, onProgress,
  );
  return { runtime, run, forwards, tensors, outputDisposals, sessions, search };
}

describe('batched browser prediction execution', () => {
  it('stacks real ONNX Tensor inputs and routes deduplicated rows into owned cache entries', async () => {
    const f = fixture();
    const roots = [nextRequest(request(), 0), nextRequest(request(), 1)];
    const rows = roots.map((root) => ({ semantic: root.state, legalActions: Int32Array.from(root.legalActions) }));
    const actual = await worker.evaluateBatch(f.runtime as never, [rows[1], rows[0], rows[1]]);
    expect(f.run).toHaveBeenCalledTimes(1);
    const feeds = f.forwards[0];
    const encoded = rows.map((row) => encodeDeltrelFeatures(row.semantic));
    const degree = encoded[0].maxDegree;
    expect(feeds.node_features.dims).toEqual([2, 50, 19]);
    expect(feeds.global_features.dims).toEqual([2, 25]);
    for (const name of ['neighbor_index', 'neighbor_mask', 'neighbor_edge_type']) expect(feeds[name].dims).toEqual([2, 50, degree]);
    expect(feeds.rings.dims).toEqual([2]);
    expect(Array.from(feeds.rings.data)).toEqual([BigInt(4), BigInt(4)]);
    expect(Array.from(feeds.legal_action_mask.data).slice(0, 50)).toEqual(Array.from(encoded[1].legalActionMask));
    expect(Array.from(feeds.legal_action_mask.data).slice(50)).toEqual(Array.from(encoded[0].legalActionMask));
    expect(Array.from(float16ToFloat32Array(feeds.node_features.data as Uint16Array).slice(0, 50 * 19)))
      .toEqual(Array.from(float16ToFloat32Array(float32ToFloat16Array(encoded[1].nodeFeatures))));
    expect(actual[0].logits[0]).toBe(30);
    expect(actual[1].logits[0]).toBe(21);
    expect(actual[0].outcome.win).toBeGreaterThan(actual[1].outcome.win);
    actual[0].logits.fill(-1);
    expect(actual[2].logits[0]).toBe(30);
    const cached = await worker.evaluateBatch(f.runtime as never, rows);
    expect(cached[1].logits[0]).toBe(30);
    expect(f.run).toHaveBeenCalledTimes(1);
    expect(f.tensors).toHaveLength(8);
    f.tensors.forEach((dispose) => expect(dispose).toHaveBeenCalledOnce());
    expect(f.outputDisposals).toHaveBeenCalledTimes(6);
  });

  it.each(['policy_logits', 'ownership_logits', 'alive_logits'] as const)('publishes no partial cache batch for malformed %s and disposes every output', async (head) => {
    const f = fixture();
    const normal = f.run.getMockImplementation()!;
    f.run.mockImplementationOnce(async (feeds) => {
      const outputs = await normal(feeds);
      outputs[head].data[50] = 0x7e00;
      return outputs;
    });
    const rows = [0, 1].map((node) => ({ semantic: nextRequest(request(), node).state, legalActions: Int32Array.from([2, 3]) }));
    await expect(worker.evaluateBatch(f.runtime as never, rows)).rejects.toThrow(/non-finite/);
    await worker.evaluateBatch(f.runtime as never, rows);
    expect(f.forwards.map((feeds) => feeds.rings.dims)).toEqual([[2], [2]]);
    expect(f.outputDisposals).toHaveBeenCalledTimes(12);
  });

  it('cleans up partially constructed input tensors and rejects mixed boards before inference', async () => {
    const f = fixture();
    const dispose = vi.fn();
    let allocations = 0;
    class FailingTensor { dispose = dispose; constructor() { if (++allocations === 3) throw new Error('allocation failed'); } }
    expect(() => worker.batchTensorFeeds({ ...f.runtime, ort: { Tensor: FailingTensor } } as never,
      [request().state, nextRequest(request(), 0).state])).toThrow('allocation failed');
    expect(dispose).toHaveBeenCalledTimes(2);
    const bigger = buildAiRequest({ ...config, rings: 6 }, []).state;
    await expect(worker.evaluateBatch(f.runtime as never,
      [{ semantic: request().state, legalActions: Int32Array.from([0]) },
        { semantic: bigger, legalActions: Int32Array.from([0]) }])).rejects.toThrow(/mixes boards/);
    expect(f.run).not.toHaveBeenCalled();
  });
});

describe('completed browser session ownership', () => {
  it('gates optional execution while leaving default manifests on the original path', () => {
    expect(worker.usesExperimentalSearch({ search: {} } as never)).toBe(false);
    expect(worker.usesExperimentalSearch({ search: { firstVisitBatchSize: 1, subtreeReuse: false } } as never)).toBe(false);
    expect(worker.usesExperimentalSearch({ search: { firstVisitBatchSize: 2 } } as never)).toBe(true);
    expect(worker.hasExpectedWasmExecution({ search_execution_version: () => 1 })).toBe(false);
    expect(worker.hasExpectedWasmExecution({ search_execution_version: () => { throw new Error(); } })).toBe(false);
  });

  it('detaches completed ownership before restart and reports fresh/inherited work separately', async () => {
    const f = fixture();
    const first = await f.search(request());
    const cached = f.runtime.completedSearch!.session;
    expect(first.rootVisits.reduce((a, b) => a + b)).toBe(2);
    cached.restarts.mockImplementation(() => expect(f.runtime.completedSearch).toBeUndefined());
    const progress = vi.fn();
    const second = await f.search(nextRequest(request(), first.actionCode), () => {}, progress);
    expect(f.sessions).toHaveLength(1);
    expect(second.reusedVisits).toBe(3);
    expect(second.rootVisits.reduce((a, b) => a + b)).toBe(2);
    expect(second.totalVisits.reduce((a, b) => a + b)).toBe(5);
    expect(progress).toHaveBeenCalledExactlyOnceWith({ completedSimulations: 2, totalSimulations: 2 });
    expect(cached.free).not.toHaveBeenCalled();
    cached.pendingStates.forEach((pending) => expect(pending.free).toHaveBeenCalledOnce());
    f.runtime.completedSearch!.session.free();
  });

  it('invalidates changed model context and caps retained trees', async () => {
    const f = fixture();
    await f.search(request());
    const first = f.sessions[0];
    f.runtime.manifest.model.sha256 = 'model-b';
    await f.search(nextRequest(request(), 0));
    expect(first.free).toHaveBeenCalledOnce();
    expect(first.restarts).not.toHaveBeenCalled();
    expect(f.sessions).toHaveLength(2);
    f.runtime.manifest.featureSchemaHash = 'features-b';
    await f.search(nextRequest(request(), 1));
    expect(f.sessions[1].free).toHaveBeenCalledOnce();
    expect(f.sessions).toHaveLength(3);
    f.runtime.manifest.search.subtreeReuseMaxNodes = 1;
    await f.search(nextRequest(request(), 1));
    expect(f.runtime.completedSearch).toBeUndefined();
    expect(f.sessions[2].free).toHaveBeenCalledOnce();
  });

  it('frees a detached reused session on cancellation and a fresh one on inference failure', async () => {
    const f = fixture();
    await f.search(request());
    let checks = 0;
    await expect(f.search(nextRequest(request(), 0), () => {
      if (++checks === 2) throw new Error('cancelled');
    })).rejects.toThrow('cancelled');
    expect(f.runtime.completedSearch).toBeUndefined();
    expect(f.sessions[0].free).toHaveBeenCalledOnce();
    const failing = fixture();
    failing.run.mockRejectedValueOnce(new Error('model failed'));
    await expect(failing.search(request())).rejects.toThrow('model failed');
    expect(failing.runtime.completedSearch).toBeUndefined();
    expect(failing.sessions[0].free).toHaveBeenCalledOnce();
  });
});


describe('root-only browser network reporting', () => {
  it.each([4, 6, 8, 10])('exposes all trained heads with board-specific masks on %i rings', async rings => {
    const f = fixture();
    const normal = f.run.getMockImplementation()!;
    const root = buildAiRequest({ ...config, rings }, [{ type: 'place', node: 0 }]);
    const nodes = root.state.stones.length;
    f.run.mockImplementationOnce(async feeds => {
      const outputs = await normal(feeds);
      const head = (dims: number[]) => ({ dims, data: new Uint16Array(dims.reduce((a, b) => a * b, 1)), dispose: f.outputDisposals });
      return { ...outputs,
        opponent_reply_logits: head([1, nodes + 1]), second_stone_logits: head([1, nodes]),
        final_shores_logits: head([1, 2, 51]), final_networks_logits: head([1, 2, 26]), final_capes_logits: head([1, 2, 6]),
      };
    });
    const output = await worker.captureRootNetworkOutput({ ...f.runtime,
      manifest: { ...f.runtime.manifest, auxiliaryStatus: 'ready' },
    } as never, root.state, () => {});
    expect(output.auxiliaryStatus).toBe('ready');
    expect(Object.values(output.heads).every(Boolean)).toBe(true);
    expect(output.heads.secondStone?.applicable).toBe(true);
    expect(output.heads.opponentReply?.probabilities[0]).toBe(0);
    expect(output.heads.opponentReply?.mask[nodes]).toBe(true);
    expect(output.heads.finalShores?.mask.slice(0, 51).filter(Boolean)).toHaveLength(5 * rings + 1);
    expect(output.heads.finalNetworks?.mask.slice(0, 26).filter(Boolean)).toHaveLength(Math.floor(5 * rings / 2) + 1);
    expect(() => validateNetworkOutputState(output, root, false)).not.toThrow();
    expect(f.outputDisposals).toHaveBeenCalledTimes(11);
  });

  it('reports six decoded FP16 heads, exact root perspective and row-wise activations', async () => {
    const f = fixture();
    const root = nextRequest(request(), 0);
    const normal = f.run.getMockImplementation()!;
    f.run.mockImplementationOnce(async (feeds) => {
      const outputs = await normal(feeds);
      outputs.ownership_logits.data.set(float32ToFloat16Array(new Float32Array([1, 0, -1, 0, 2, 0])));
      outputs.alive_logits.data.set(float32ToFloat16Array(new Float32Array([-2, 0, 2])));
      outputs.soft_policy_logits.data.fill(0);
      outputs.soft_policy_logits.data[2] = float32ToFloat16Array(new Float32Array([3]))[0];
      outputs.score_margin_logits.data[302] = float32ToFloat16Array(new Float32Array([0.1]))[0];
      return outputs;
    });
    const output = await worker.captureRootNetworkOutput(f.runtime as never, root.state, () => {});
    expect(output.perspective).toBe(1);
    expect(output.nodeCount).toBe(50);
    expect(output.auxiliaryStatus).toBe('absent');
    expect(Object.keys(output.heads)).toHaveLength(11);
    expect(output.heads.policy?.shape).toEqual([50]);
    expect(output.heads.outcome?.shape).toEqual([2]);
    expect(output.heads.scoreMargin?.shape).toEqual([303]);
    expect(output.heads.ownership?.shape).toEqual([50, 3]);
    expect(output.heads.alive?.shape).toEqual([50]);
    expect(output.heads.softPolicy?.shape).toEqual([50]);
    for (const name of ['opponentReply', 'secondStone', 'finalShores', 'finalNetworks', 'finalCapes'] as const) {
      expect(output.heads[name]).toBeNull();
    }
    for (const name of ['policy', 'softPolicy'] as const) {
      const head = output.heads[name]!;
      expect(head.mask).toEqual(root.state.stones.map(stone => stone === -1));
      expect(head.logits[0]).toBeNull();
      expect(head.probabilities[0]).toBe(0);
      expect(head.probabilities.reduce((sum, p) => sum + p, 0)).toBeCloseTo(1, 12);
    }
    expect(output.heads.policy?.logits[1]).toBe(21);
    expect(output.heads.softPolicy?.logits[2]).toBe(3);
    expect(output.heads.softPolicy!.probabilities[2]).toBeGreaterThan(output.heads.softPolicy!.probabilities[1]);
    expect(output.heads.outcome?.logits).toEqual([0, 2]);
    expect(output.heads.outcome?.probabilities[1]).toBeCloseTo(1 / (1 + Math.exp(-2)), 12);
    expect(output.heads.scoreMargin?.logits[302]).toBe(0.0999755859375);
    expect(output.heads.scoreMargin!.probabilities.reduce((sum, p) => sum + p, 0)).toBeCloseTo(1, 12);
    expect(output.heads.ownership!.probabilities[0]).toBeCloseTo(Math.exp(1) / (Math.exp(1) + 1 + Math.exp(-1)), 12);
    for (let node = 0; node < 50; node++) {
      expect(output.heads.ownership!.probabilities.slice(node * 3, node * 3 + 3).reduce((sum, p) => sum + p, 0)).toBeCloseTo(1, 12);
    }
    expect(output.heads.alive?.activation).toBe('sigmoid');
    expect(output.heads.alive?.probabilities.slice(0, 3)).toEqual([
      1 / (1 + Math.exp(2)), 0.5, 1 / (1 + Math.exp(-2)),
    ]);
    expect(() => validateNetworkOutputState(output, root, false)).not.toThrow();
    expect(f.outputDisposals).toHaveBeenCalledTimes(6);
    f.tensors.forEach(dispose => expect(dispose).toHaveBeenCalledOnce());
  });

  it('uses actual board dimensions and preserves perspective after a pie swap', async () => {
    const f = fixture();
    const root = buildAiRequest({ ...config, rings: 10, pieRule: true }, [
      { type: 'place', node: 274 }, { type: 'swap' },
    ]);
    const output = await worker.captureRootNetworkOutput(f.runtime as never, root.state, () => {});
    expect(output.perspective).toBe(0);
    expect(output.nodeCount).toBe(275);
    expect(output.heads.ownership?.shape).toEqual([275, 3]);
    expect(output.heads.policy?.mask[274]).toBe(false);
    expect(output.heads.policy?.logits[274]).toBeNull();
    expect(output.heads.alive?.probabilities).toHaveLength(275);
    expect(() => validateNetworkOutputState(output, root, false)).not.toThrow();
  });

  it('reports cached roots without adding full vectors to the leaf prediction cache', async () => {
    const f = fixture();
    const root = request();
    const legal = Int32Array.from(root.legalActions);
    const prediction = await worker.evaluate(f.runtime as never, root.state, legal);
    const get = vi.spyOn(f.runtime.predictions, 'get');
    const set = vi.spyOn(f.runtime.predictions, 'set');
    const report = await worker.captureRootNetworkOutput(f.runtime as never, root.state, () => {});
    expect(get).toHaveBeenCalledOnce();
    expect(set).not.toHaveBeenCalled();
    expect(f.run).toHaveBeenCalledTimes(2);
    expect(report.heads.policy?.logits).toHaveLength(50);
    const cached = await worker.evaluate(f.runtime as never, root.state, legal);
    expect(f.run).toHaveBeenCalledTimes(2);
    expect(Object.keys(cached).sort()).toEqual(['expectedMargin', 'logits', 'outcome', 'value']);
    expect(cached).toEqual(prediction);
  });

  it('does not allocate or infer when capture was already cancelled', async () => {
    const f = fixture();
    await expect(worker.captureRootNetworkOutput(f.runtime as never, request().state, () => {
      throw new Error('cancelled');
    })).rejects.toThrow('cancelled');
    expect(f.run).not.toHaveBeenCalled();
    expect(f.tensors).toHaveLength(0);
    expect(f.outputDisposals).not.toHaveBeenCalled();
  });

  it('releases all root tensors when cancellation arrives during the root forward', async () => {
    const f = fixture();
    const normal = f.run.getMockImplementation()!;
    let cancelled = false;
    f.run.mockImplementationOnce(async feeds => {
      const outputs = await normal(feeds);
      cancelled = true;
      return outputs;
    });
    await expect(worker.captureRootNetworkOutput(f.runtime as never, request().state, () => {
      if (cancelled) throw new Error('cancelled');
    })).rejects.toThrow('cancelled');
    expect(f.tensors).toHaveLength(8);
    f.tensors.forEach(dispose => expect(dispose).toHaveBeenCalledOnce());
    expect(f.outputDisposals).toHaveBeenCalledTimes(6);
  });

  it.each(['shape', 'length', 'non-finite'] as const)('rejects %s root reports and still releases every tensor', async fault => {
    const f = fixture();
    const normal = f.run.getMockImplementation()!;
    f.run.mockImplementationOnce(async feeds => {
      const outputs = await normal(feeds);
      if (fault === 'shape') outputs.ownership_logits.dims = [1, 50, 2];
      if (fault === 'length') outputs.alive_logits.data = new Uint16Array(49);
      if (fault === 'non-finite') outputs.soft_policy_logits.data[2] = 0x7e00;
      return outputs;
    });
    await expect(worker.captureRootNetworkOutput(f.runtime as never, request().state, () => {})).rejects.toThrow(
      fault === 'shape' ? /wrong shape/ : fault === 'length' ? /wrong data length/ : /non-finite/,
    );
    expect(f.outputDisposals).toHaveBeenCalledTimes(6);
    f.tensors.forEach(dispose => expect(dispose).toHaveBeenCalledOnce());
  });

  it('cleans up a failed root inference and partially allocated inputs', async () => {
    const f = fixture();
    const normal = f.run.getMockImplementation()!;
    f.run.mockImplementationOnce(async feeds => {
      // Record disposals as the real runtime would, then fail before outputs exist.
      Object.values(feeds).forEach(tensor => { f.tensors.push(vi.spyOn(tensor, 'dispose')); });
      throw new Error('root inference failed');
    });
    await expect(worker.captureRootNetworkOutput(f.runtime as never, request().state, () => {})).rejects.toThrow('root inference failed');
    expect(f.outputDisposals).not.toHaveBeenCalled();
    f.tensors.forEach(dispose => expect(dispose).toHaveBeenCalledOnce());
    f.run.mockImplementation(normal);
    const dispose = vi.fn();
    let allocations = 0;
    class FailingTensor {
      dispose = dispose;
      constructor() { if (++allocations === 3) throw new Error('allocation failed'); }
    }
    await expect(worker.captureRootNetworkOutput({ ...f.runtime, ort: { Tensor: FailingTensor } } as never,
      request().state, () => {})).rejects.toThrow('allocation failed');
    expect(dispose).toHaveBeenCalledTimes(2);
  });
});


describe('single-pass root evaluation and bounded report reuse', () => {
  it.each([false, true])('combines search evaluation and the full report in one forward (auxiliary=%s)', async auxiliary => {
    const f = fixture(auxiliary);
    const baseline = fixture(auxiliary);
    const root = nextRequest(request(), 0);
    const legal = Int32Array.from(root.legalActions);
    const expectedEvaluation = await worker.evaluate(baseline.runtime as never, root.state, legal);
    const expectedReport = await worker.captureRootNetworkOutput(baseline.runtime as never, root.state, () => {});
    expect(baseline.run).toHaveBeenCalledTimes(2);

    const result = await worker.evaluateRoot(f.runtime as never, root.state, legal, () => {});
    expect(f.run).toHaveBeenCalledOnce();
    expect(result.evaluation).toEqual(expectedEvaluation);
    expect(result.networkOutput).toEqual(expectedReport);
    expect(Object.values(result.networkOutput.heads).filter(Boolean)).toHaveLength(auxiliary ? 11 : 6);
    expect(await worker.evaluate(f.runtime as never, root.state, legal)).toEqual(expectedEvaluation);
    expect(f.run).toHaveBeenCalledOnce();
    expect(f.tensors).toHaveLength(8);
    f.tensors.forEach(dispose => expect(dispose).toHaveBeenCalledOnce());
    expect(f.outputDisposals).toHaveBeenCalledTimes(auxiliary ? 11 : 6);
  });

  it('retains only one complete root and isolates caller mutations from both caches', async () => {
    const f = fixture(true);
    const root = nextRequest(request(), 0);
    const legal = Int32Array.from(root.legalActions);
    const first = await worker.evaluateRoot(f.runtime as never, root.state, legal, () => {});
    const expected = structuredClone(first);
    first.evaluation.logits.fill(-100);
    first.evaluation.outcome.win = 0;
    first.networkOutput.heads.policy!.probabilities.fill(0);
    first.networkOutput.heads.finalShores!.logits.fill(100);
    first.networkOutput.heads.ownership!.shape[0] = 1;
    first.networkOutput.heads.secondStone!.applicable = false;
    const repeated = await worker.evaluateRoot(f.runtime as never, root.state, legal, () => {});
    expect(repeated).toEqual(expected);
    expect(f.run).toHaveBeenCalledOnce();
    const compact = await worker.evaluate(f.runtime as never, root.state, legal);
    expect(Object.keys(compact).sort()).toEqual(['expectedMargin', 'logits', 'outcome', 'value']);
    expect(compact).toEqual(expected.evaluation);

    const other = nextRequest(request(), 1);
    await worker.evaluateRoot(f.runtime as never, other.state, Int32Array.from(other.legalActions), () => {});
    await worker.evaluateRoot(f.runtime as never, root.state, legal, () => {});
    expect(f.run).toHaveBeenCalledTimes(3); // The prior full report was evicted.
  });

  it('preserves an existing leaf evaluation exactly while obtaining missing raw heads', async () => {
    const f = fixture(true);
    const root = request();
    const legal = Int32Array.from(root.legalActions);
    const cached = await worker.evaluate(f.runtime as never, root.state, legal);
    const normal = f.run.getMockImplementation()!;
    f.run.mockImplementationOnce(async feeds => {
      const outputs = await normal(feeds);
      // Model execution in a differently shaped batch can round differently.
      outputs.outcome_logits.data[1] = float32ToFloat16Array(new Float32Array([1.01]))[0];
      outputs.policy_logits.data[0] = float32ToFloat16Array(new Float32Array([10.01]))[0];
      return outputs;
    });
    const result = await worker.evaluateRoot(f.runtime as never, root.state, legal, () => {});
    expect(result.evaluation).toEqual(cached);
    expect(result.networkOutput.heads.outcome!.probabilities[1]).not.toBe(cached.outcome.win);
    expect((await worker.evaluate(f.runtime as never, root.state, legal))).toEqual(cached);
    expect(f.run).toHaveBeenCalledTimes(2);
  });

  it('invalidates reports on model, features, auxiliary readiness, history, and action-order changes', async () => {
    const f = fixture(true);
    const root = request();
    const legal = Int32Array.from(root.legalActions);
    const evaluate = (semantic = root.state, actions = legal) => worker.evaluateRoot(f.runtime as never, semantic, actions, () => {});
    await evaluate();
    f.runtime.manifest.model.sha256 = 'model-b';
    await evaluate();
    f.runtime.manifest.featureSchemaHash = 'features-b';
    await evaluate();
    f.runtime.manifest.auxiliaryStatus = 'untrained';
    expect((await evaluate()).networkOutput.auxiliaryStatus).toBe('untrained');
    const differentHistory = { ...root.state, history: { ...root.state.history, previousTurn: [1] } };
    await evaluate(differentHistory);
    const reversed = await evaluate(differentHistory, legal.slice().reverse());
    expect(reversed.evaluation.logits[0]).toBe(59);
    expect(f.run).toHaveBeenCalledTimes(6);
  });

  it('changes swap applicability after search without modifying the reusable report', async () => {
    const f = fixture(true);
    const root = buildAiRequest({ ...config, pieRule: true }, [{ type: 'place', node: 0 }]);
    const swap = await worker.captureRootNetworkOutput(f.runtime as never, root.state, () => {}, true);
    expect(swap.heads.secondStone?.applicable).toBe(false);
    const keep = await worker.captureRootNetworkOutput(f.runtime as never, root.state, () => {}, false);
    expect(keep.heads.secondStone?.applicable).toBe(true);
    expect(f.run).toHaveBeenCalledOnce();
    expect(() => validateNetworkOutputState(swap, root, true)).not.toThrow();
    expect(() => validateNetworkOutputState(keep, root, false)).not.toThrow();
  });

  it('publishes neither cache until every head is valid and cancellation checks pass', async () => {
    const f = fixture(true);
    const root = request();
    const legal = Int32Array.from(root.legalActions);
    const set = vi.spyOn(f.runtime.predictions, 'set');
    let checks = 0;
    await expect(worker.evaluateRoot(f.runtime as never, root.state, legal, () => {
      if (++checks === 4) throw new Error('cancelled after decode');
    })).rejects.toThrow('cancelled after decode');
    expect(set).not.toHaveBeenCalled();
    expect(f.runtime).not.toHaveProperty('rootPrediction');
    expect(f.outputDisposals).toHaveBeenCalledTimes(11);
    f.tensors.forEach(dispose => expect(dispose).toHaveBeenCalledOnce());
    await worker.evaluateRoot(f.runtime as never, root.state, legal, () => {});
    expect(f.run).toHaveBeenCalledTimes(2);
    await expect(worker.evaluateRoot(f.runtime as never, root.state, legal, () => {
      throw new Error('cancelled cache hit');
    })).rejects.toThrow('cancelled cache hit');
    expect(f.run).toHaveBeenCalledTimes(2);
  });

  it('returns identical experimental search statistics while removing the reporting forward', async () => {
    const f = fixture(true);
    const root = request();
    const first = await f.search(root);
    expect(f.run).toHaveBeenCalledTimes(2); // One complete root plus one leaf batch.
    expect(first.networkOutput.auxiliaryStatus).toBe('ready');
    expect(first.rootVisits.reduce((sum, visits) => sum + visits, 0)).toBe(2);
    const repeated = await f.search(root);
    expect(repeated).toEqual(first);
    expect(f.run).toHaveBeenCalledTimes(2); // Both prediction caches served the repeat.
    expect(f.sessions).toHaveLength(1);
    f.runtime.completedSearch!.session.free();
  });
});
