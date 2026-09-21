import { describe, expect, it } from 'vitest';
import {
  DELTREL_MODEL_INPUT_NAMES,
  DELTREL_MODEL_OUTPUT_NAMES,
} from '../features';
import {
  DELTREL_BROWSER_MODEL_MANIFEST_SCHEMA_ID,
  DELTREL_WASM_BINARY_PATH,
  DELTREL_WASM_MODULE_PATH,
  parseDeltrelBrowserModelManifest,
} from '../manifest';
import {
  DELTREL_FEATURE_SCHEMA_HASH,
  buildAiRequest,
  makeAiResponse,
} from '../protocol';
import { parseWorkerCommand, parseWorkerEvent } from '../worker-protocol';

const request = buildAiRequest(
  {
    rings: 4,
    mode: 'double',
    pieRule: false,
    playerNames: ['A', 'B'],
  },
  [],
  'worker-task',
);

function workerDecision() {
  return {
    response: makeAiResponse(request, { type: 'place' as const, node: 0 }),
    analysis: {
      perspective: request.state.toMove,
      stateHash: request.stateHash,
      outcome: { loss: 0.4, win: 0.6 },
      modelValue: 0.2,
      searchValue: 0.2,
      rootValue: 0.15,
      swapRecommended: false,
      expectedMargin: 1.5,
      rootActions: [
        { type: 'place' as const, node: 0 },
        { type: 'place' as const, node: 1 },
      ],
      rootPolicy: [0.75, 0.25],
      rootQ: [0.2, -0.1],
      rootVisits: [3, 1],
      modelVersion: 'browser-smoke-v2',
      modelStep: null,
      modelIdentity: 'browser-smoke-v2',
      simulations: 4,
      maxConsidered: 2,
      timingMs: {
        queue: 0,
        modelLoad: 1,
        inferenceSearch: 2,
        total: 3,
      },
    },
  };
}

const manifest = {
  format: DELTREL_BROWSER_MODEL_MANIFEST_SCHEMA_ID,
  schema_version: 3,
  model_version: 'browser-smoke-v3',
  created_ns: 1_700_000_000_000_000_000,
  weights: 'ema',
  rules: {
    schema_id: 'deltrel.rules.v3',
    hash: 'fnv1a64:46e4fbcff4e17fd3',
    mode: 'double',
    pie_rule: false,
    handicap: 1,
    rings: [4, 6, 8, 10],
    variants: {
      modes: ['classic', 'double'],
      handicap_min: 1,
      handicap_max: 9,
      pie_allowed: true,
    },
  },
  features: {
    schema_id: 'deltrel.model-features.external.v3',
    version: 4,
    hash: DELTREL_FEATURE_SCHEMA_HASH,
    node_feature_count: 19,
    global_feature_count: 25,
  },
  actions: {
    schema_id: 'deltrel.action-layout.nodes-only.v1',
    types: ['place', 'swap'],
  },
  outcome: {
    classes: ['loss', 'win'],
    value: 'P(win)-P(loss)',
  },
  architecture: {
    name: 'GraphResTNet',
    schema_version: 3,
    all_size: true,
    parameter_count: 12_345,
    config: {
      node_feature_dim: 19,
      global_feature_dim: 25,
      feature_schema_version: 4,
      width: 64,
      rrt_groups: 5,
      attention_heads: 8,
      kv_heads: 2,
      bottleneck_ratio: 0.5,
      ff_multiplier: 2,
      dropout: 0,
      rms_norm_eps: 0.000001,
      score_margin_min: -151,
      score_margin_max: 151,
      soft_policy_temperature: 4,
      local_operator: 'mean',
      local_blocks_per_group: 2,
      relational_bias: true,
      adaln_hidden: 32,
    },
  },
  precision: 'float16',
  artifacts: {
    onnx: {
      file: 'browser-smoke-v3.fp16.onnx',
      sha256: 'a'.repeat(64),
      bytes: 123_456,
      opset: 18,
    },
    checkpoint: {
      file: 'browser-smoke-v3.pt',
      sha256: 'b'.repeat(64),
      bytes: 234_567,
    },
  },
  tensors: {
    inputs: {
      node_features: { dtype: 'float16', shape: ['batch', 'nodes', 19] },
      global_features: { dtype: 'float16', shape: ['batch', 25] },
      neighbor_index: { dtype: 'int64', shape: ['batch', 'nodes', 'degree'] },
      neighbor_mask: { dtype: 'bool', shape: ['batch', 'nodes', 'degree'] },
      neighbor_edge_type: { dtype: 'int64', shape: ['batch', 'nodes', 'degree'] },
      node_mask: { dtype: 'bool', shape: ['batch', 'nodes'] },
      legal_action_mask: { dtype: 'bool', shape: ['batch', 'nodes'] },
      rings: { dtype: 'int64', shape: ['batch'] },
    },
    outputs: {
      policy_logits: { dtype: 'float16', shape: ['batch', 'nodes'] },
      outcome_logits: { dtype: 'float16', shape: ['batch', 2] },
      score_margin_logits: { dtype: 'float16', shape: ['batch', 303] },
      ownership_logits: { dtype: 'float16', shape: ['batch', 'nodes', 3] },
      alive_logits: { dtype: 'float16', shape: ['batch', 'nodes'] },
      soft_policy_logits: { dtype: 'float16', shape: ['batch', 'nodes'] },
    },
  },
  recommended_local_search: {
    simulations: 64,
    max_considered: 16,
    c_visit: 50,
    c_scale: 1,
    swap_dead_zone: 0.02,
  },
  training: {
    steps: 10_000,
    replay_samples: 1_000_000,
    teacher_model_version: 'teacher-v2',
    teacher_logit_kl: true,
  },
};

describe('local worker protocol', () => {
  it('accepts only bounded, completed simulation progress with no arbitrary heartbeat fields', () => {
    const event = { type: 'search-progress', taskId: request.requestId,
      progress: { completedSimulations: 1, totalSimulations: 64 } };
    expect(parseWorkerEvent(event)).toEqual(event);
    expect(parseWorkerEvent({ ...event, progress: { completedSimulations: 64, totalSimulations: 64 } })).toMatchObject({
      progress: { completedSimulations: 64, totalSimulations: 64 },
    });
    for (const changes of [
      { completedSimulations: 0 }, { completedSimulations: -1 }, { completedSimulations: 65 },
      { completedSimulations: 0.5 }, { completedSimulations: NaN }, { completedSimulations: '1' },
      { totalSimulations: 0 }, { totalSimulations: 1025 }, { totalSimulations: Infinity },
      { totalSimulations: 64.5 }, { totalSimulations: '64' }, { elapsedMs: 10 },
    ]) {
      expect(() => parseWorkerEvent({ ...event, progress: { ...event.progress, ...changes } })).toThrow(/invalid event/);
    }
    expect(() => parseWorkerEvent({ ...event, heartbeat: true })).toThrow(/invalid event/);
    expect(() => parseWorkerEvent({ ...event, taskId: '' })).toThrow(/invalid message/);
    expect(() => parseWorkerEvent({ type: 'ready', protocolVersion: 3 })).toThrow(/invalid message/);
  });

  it('round-trips a typed choose command and cancellation', () => {
    expect(
      parseWorkerCommand({
        type: 'choose',
        taskId: request.requestId,
        request,
        search: { simulations: 32, maxConsidered: 8 },
      }),
    ).toMatchObject({
      type: 'choose',
      taskId: 'worker-task',
      search: { simulations: 32, maxConsidered: 8 },
    });
    expect(parseWorkerCommand({ type: 'cancel', taskId: 'worker-task' })).toEqual({
      type: 'cancel',
      taskId: 'worker-task',
    });
  });

  it('rejects a semantic payload whose state hash was altered', () => {
    expect(() =>
      parseWorkerCommand({
        type: 'choose',
        taskId: 'worker-task',
        request: { ...request, stateHash: 'zobrist64:0000000000000000' },
        search: null,
      }),
    ).toThrow(/state hash/i);
  });

  it('rejects removed state fields and negative action codes', () => {
    expect(() =>
      parseWorkerCommand({
        type: 'choose',
        taskId: 'worker-task',
        request: {
          ...request,
          state: { ...request.state, passStreak: 0 },
        },
        search: null,
      }),
    ).toThrow(/invalid semantic state/i);
    expect(() =>
      parseWorkerCommand({
        type: 'choose',
        taskId: 'worker-task',
        request: { ...request, legalActions: [...request.legalActions, -1] },
        search: null,
      }),
    ).toThrow(/incompatible AI request/i);
  });

  it('rejects malformed or out-of-range per-request browser budgets', () => {
    for (const search of [
      { simulations: 0, maxConsidered: 8 },
      { simulations: 1_025, maxConsidered: 8 },
      { simulations: 32, maxConsidered: 129 },
      { simulations: 32, maxConsidered: 8, extra: true },
    ]) {
      expect(() =>
        parseWorkerCommand({
          type: 'choose',
          taskId: request.requestId,
          request,
          search,
        }),
      ).toThrow(/search budget/i);
    }
  });

  it('parses structured worker errors without trusting arbitrary codes', () => {
    expect(parseWorkerEvent({ type: 'ready', protocolVersion: 4 })).toEqual({
      type: 'ready',
      protocolVersion: 4,
    });
    expect(() => parseWorkerEvent({ type: 'ready', protocolVersion: 2 })).toThrow(
      /invalid message/,
    );
    expect(
      parseWorkerEvent({
        type: 'error',
        taskId: 'worker-task',
        error: { code: 'unavailable', message: 'missing', retryable: false },
      }),
    ).toMatchObject({ type: 'error', error: { code: 'unavailable' } });
    expect(
      parseWorkerEvent({
        type: 'result',
        taskId: request.requestId,
        decision: workerDecision(),
      }),
    ).toMatchObject({
      type: 'result',
      decision: {
        analysis: {
          outcome: { loss: 0.4, win: 0.6 },
          rootVisits: [3, 1],
          modelIdentity: 'browser-smoke-v2',
        },
      },
    });
    expect(() =>
      parseWorkerEvent({
        type: 'result',
        taskId: request.requestId,
        decision: {
          ...workerDecision(),
          analysis: { ...workerDecision().analysis, rootQ: [Number.NaN, 0] },
        },
      }),
    ).toThrow(/root Q/i);
    expect(() =>
      parseWorkerEvent({
        type: 'result',
        taskId: 'stale-task',
        decision: workerDecision(),
      }),
    ).toThrow(/identity/i);
    expect(() =>
      parseWorkerEvent({
        type: 'error',
        taskId: 'worker-task',
        error: { code: 'anything', message: 'bad', retryable: false },
      }),
    ).toThrow(/invalid event/i);
  });

  it('accepts only fully pinned browser model manifests', () => {
    expect(parseDeltrelBrowserModelManifest(manifest)).toMatchObject({
      rulesHash: 'fnv1a64:46e4fbcff4e17fd3',
      featureSchemaHash: DELTREL_FEATURE_SCHEMA_HASH,
      weights: 'ema',
      wasm: {
        moduleUrl: DELTREL_WASM_MODULE_PATH,
        binaryUrl: DELTREL_WASM_BINARY_PATH,
      },
      model: {
        precision: 'float16',
        url: '/models/deltrel/browser-smoke-v3.fp16.onnx',
        sha256: `sha256:${'a'.repeat(64)}`,
        inputs: DELTREL_MODEL_INPUT_NAMES,
        outputs: DELTREL_MODEL_OUTPUT_NAMES,
      },
      search: {
        simulations: 64,
        maxConsidered: 16,
        maximumSimulations: 1_024,
        maximumMaxConsidered: 128,
        swapDeadZone: 0.02,
      },
    });
    expect(() =>
      parseDeltrelBrowserModelManifest({
        ...manifest,
        rules: { ...manifest.rules, hash: 'fnv1a64:0000000000000000' },
      }),
    ).toThrow(/do not match/i);
    expect(() =>
      parseDeltrelBrowserModelManifest({
        ...manifest,
        artifacts: {
          ...manifest.artifacts,
          onnx: { ...manifest.artifacts.onnx, sha256: 'unverified' },
        },
      }),
    ).toThrow(/fields are invalid/i);
  });

  it('validates download stages and readiness without confusing them with move results', () => {
    expect(parseWorkerCommand({ type: 'prepare', taskId: 'download' })).toEqual({ type: 'prepare', taskId: 'download' });
    const progress = { phase: 'downloading', loadedBytes: 40, totalBytes: 100, modelVersion: 'champion-v1', cached: false };
    expect(parseWorkerEvent({ type: 'progress', taskId: 'download', progress })).toMatchObject({ progress });
    for (const changes of [{ loadedBytes: -1 }, { loadedBytes: 101 }, { totalBytes: 0 }, { phase: 'ready' }, { cached: 'yes' }]) {
      expect(() => parseWorkerEvent({ type: 'progress', taskId: 'download', progress: { ...progress, ...changes } })).toThrow();
    }
    const info = { modelVersion: 'champion-v1', bytes: 100, backend: 'wasm', cached: true };
    expect(parseWorkerEvent({ type: 'prepared', taskId: 'download', info })).toMatchObject({ info });
    expect(() => parseWorkerEvent({ type: 'prepared', taskId: 'download', info: { ...info, backend: 'unknown' } })).toThrow();
  });

  it('accepts the complete champion export without treating unsupervised heads as trained', () => {
    const expanded = {
      ...manifest,
      tensors: { ...manifest.tensors, outputs: { ...manifest.tensors.outputs,
        opponent_reply_logits: { dtype: 'float16', shape: ['batch', 'nodes+1'] },
        second_stone_logits: { dtype: 'float16', shape: ['batch', 'nodes'] },
        final_shores_logits: { dtype: 'float16', shape: ['batch', 2, 51] },
        final_networks_logits: { dtype: 'float16', shape: ['batch', 2, 26] },
        final_capes_logits: { dtype: 'float16', shape: ['batch', 2, 6] },
      } },
      training: { ...manifest.training, auxiliary_predictions_ready: true },
    };
    expect(parseDeltrelBrowserModelManifest(expanded)).toMatchObject({ auxiliaryStatus: 'ready' });
    expect(parseDeltrelBrowserModelManifest(expanded).model.outputs).toHaveLength(11);
    expect(parseDeltrelBrowserModelManifest({ ...expanded, training: {} }).auxiliaryStatus).toBe('untrained');
    expect(parseDeltrelBrowserModelManifest(manifest).auxiliaryStatus).toBe('absent');
    expect(() => parseDeltrelBrowserModelManifest({ ...expanded, tensors: { ...expanded.tensors,
      outputs: { ...expanded.tensors.outputs, final_networks_logits: { dtype: 'float16', shape: ['batch', 2, 25] } },
    } })).toThrow();
  });

  it('accepts bounded optional execution fields without adding them to old manifests', () => {
    expect(parseDeltrelBrowserModelManifest(manifest).search).not.toHaveProperty('subtreeReuse');
    expect(parseDeltrelBrowserModelManifest({ ...manifest, recommended_local_search: {
      ...manifest.recommended_local_search, first_visit_batch_size: 4,
      subtree_reuse: true, subtree_reuse_max_nodes: 2048,
    } }).search).toMatchObject({ firstVisitBatchSize: 4, subtreeReuse: true, subtreeReuseMaxNodes: 2048 });
    for (const change of [{ first_visit_batch_size: 0 }, { first_visit_batch_size: 65 },
      { first_visit_batch_size: true }, { subtree_reuse: 1 }, { subtree_reuse_max_nodes: 65537 },
      { subtree_reuse_max_nodes: 0 }, { unknown: true }]) {
      expect(() => parseDeltrelBrowserModelManifest({ ...manifest,
        recommended_local_search: { ...manifest.recommended_local_search, ...change },
      })).toThrow(/fields are invalid/i);
    }
  });
});
