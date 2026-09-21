import {
  DELTREL_ACTION_LAYOUT_SCHEMA_ID,
  DELTREL_FEATURE_SCHEMA_ID,
  DELTREL_RULES_HASH,
  DELTREL_RULES_SCHEMA_ID,
} from '../rules';
import { SUPPORTED_RINGS } from '../board';
import {
  DELTREL_ACTION_LAYOUT_VERSION,
  DELTREL_FEATURE_SCHEMA_HASH,
  DELTREL_FEATURE_SCHEMA_VERSION,
} from './protocol';
import {
  DELTREL_GLOBAL_FEATURE_DIM,
  DELTREL_MODEL_INPUT_NAMES,
  DELTREL_MODEL_OUTPUT_NAMES,
  DELTREL_NODE_FEATURE_DIM,
} from './features';
import { DeltrelAiError } from './errors';

export const DELTREL_BROWSER_MODEL_MANIFEST_SCHEMA_ID =
  'deltreltrain.browser-model' as const;
export const DELTREL_BROWSER_MODEL_MANIFEST_VERSION = 3 as const;
export const DELTREL_BROWSER_MODEL_ARCHITECTURE_VERSION = 3 as const;
export const DELTREL_BROWSER_MODEL_PRECISION = 'float16' as const;
export const MAX_BROWSER_AI_SIMULATIONS = 1_024;
export const MAX_BROWSER_AI_MAX_CONSIDERED = 128;
export const MAX_BROWSER_AI_FIRST_VISIT_BATCH_SIZE = 64;
export const DEFAULT_BROWSER_AI_SUBTREE_REUSE_MAX_NODES = 4_096;
export const MAX_BROWSER_AI_SUBTREE_REUSE_NODES = 65_536;

/** Deployment convention; intentionally absent until a trained model is published. */
export const DELTREL_BROWSER_MODEL_MANIFEST_PATH = '/models/deltrel/manifest.json' as const;
/** The WASM package directory is keyed by the rules hash it was built from. */
export const DELTREL_WASM_MODULE_PATH =
  '/models/deltrel/wasm-46e4fbcff4e17fd3/deltrel_wasm.js' as const;
export const DELTREL_WASM_BINARY_PATH =
  '/models/deltrel/wasm-46e4fbcff4e17fd3/deltrel_wasm_bg.wasm' as const;

export interface DeltrelBrowserModelManifest {
  format: typeof DELTREL_BROWSER_MODEL_MANIFEST_SCHEMA_ID;
  schemaVersion: typeof DELTREL_BROWSER_MODEL_MANIFEST_VERSION;
  modelVersion: string;
  weights: 'ema';
  rulesSchema: typeof DELTREL_RULES_SCHEMA_ID;
  rulesHash: typeof DELTREL_RULES_HASH;
  featureSchema: typeof DELTREL_FEATURE_SCHEMA_ID;
  featureSchemaVersion: typeof DELTREL_FEATURE_SCHEMA_VERSION;
  featureSchemaHash: string;
  actionLayout: typeof DELTREL_ACTION_LAYOUT_SCHEMA_ID;
  actionLayoutVersion: typeof DELTREL_ACTION_LAYOUT_VERSION;
  outcome: {
    classes: readonly ['loss', 'win'];
    value: 'P(win)-P(loss)';
  };
  wasm: {
    moduleUrl: typeof DELTREL_WASM_MODULE_PATH;
    binaryUrl: typeof DELTREL_WASM_BINARY_PATH;
  };
  model: {
    format: 'onnx';
    precision: typeof DELTREL_BROWSER_MODEL_PRECISION;
    url: string;
    sha256: string;
    bytes: number;
    opset: number;
    inputs: readonly string[];
    outputs: readonly string[];
  };
  search: {
    simulations: number;
    maxConsidered: number;
    maximumSimulations: typeof MAX_BROWSER_AI_SIMULATIONS;
    maximumMaxConsidered: typeof MAX_BROWSER_AI_MAX_CONSIDERED;
    cVisit: number;
    cScale: number;
    /** A pie responder swaps when the selected keep continuation is below -deadZone. */
    swapDeadZone: number;
    firstVisitBatchSize?: number;
    subtreeReuse?: boolean;
    subtreeReuseMaxNodes?: number;
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  return (
    actual.length === sortedExpected.length &&
    actual.every((key, index) => key === sortedExpected[index])
  );
}

function positiveInteger(value: unknown, maximum: number): value is number {
  return (
    typeof value === 'number' &&
    Number.isInteger(value) &&
    value > 0 &&
    value <= maximum
  );
}

function positiveFinite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0;
}

function isSafeArtifactFile(value: unknown, extension: string): value is string {
  return (
    typeof value === 'string' &&
    new RegExp(`^[A-Za-z0-9][A-Za-z0-9._-]*\\.${extension}$`).test(value) &&
    !value.includes('..')
  );
}

function isChecksum(value: unknown): value is string {
  return typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
}

function sameShape(value: unknown, expected: readonly (string | number)[]): boolean {
  return (
    Array.isArray(value) &&
    value.length === expected.length &&
    value.every((item, index) => item === expected[index])
  );
}

function validateTensorEntry(
  value: unknown,
  dtype: 'float16' | 'int64' | 'bool',
  shape: readonly (string | number)[],
): boolean {
  return (
    isRecord(value) &&
    hasExactKeys(value, ['dtype', 'shape']) &&
    value.dtype === dtype &&
    sameShape(value.shape, shape)
  );
}

function validateTensorMap(
  value: unknown,
  expectations: ReadonlyArray<
    readonly [string, 'float16' | 'int64' | 'bool', readonly (string | number)[]]
  >,
): boolean {
  if (!isRecord(value) || !hasExactKeys(value, expectations.map(([name]) => name))) {
    return false;
  }
  return expectations.every(([name, dtype, shape]) =>
    validateTensorEntry(value[name], dtype, shape),
  );
}

function validateArtifact(
  value: unknown,
  extension: 'onnx' | 'pt',
  withOpset: boolean,
): value is Record<string, unknown> {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, withOpset ? ['file', 'sha256', 'bytes', 'opset'] : ['file', 'sha256', 'bytes']) ||
    !isSafeArtifactFile(value.file, extension) ||
    !isChecksum(value.sha256) ||
    !positiveInteger(value.bytes, Number.MAX_SAFE_INTEGER)
  ) {
    return false;
  }
  return (
    !withOpset ||
    (typeof value.opset === 'number' && Number.isInteger(value.opset) && value.opset >= 18)
  );
}

export function parseDeltrelBrowserModelManifest(payload: unknown): DeltrelBrowserModelManifest {
  if (!isRecord(payload)) {
    throw new DeltrelAiError('unavailable', 'Local AI model manifest is invalid.');
  }

  const topLevelKeys = [
    'format',
    'schema_version',
    'model_version',
    'created_ns',
    'rules',
    'features',
    'actions',
    'outcome',
    'architecture',
    'precision',
    'weights',
    'artifacts',
    'tensors',
    'recommended_local_search',
    'training',
  ] as const;
  const rules = payload.rules;
  const features = payload.features;
  const actions = payload.actions;
  const outcome = payload.outcome;
  const architecture = payload.architecture;
  const artifacts = payload.artifacts;
  const tensors = payload.tensors;
  const search = payload.recommended_local_search;

  if (
    !hasExactKeys(payload, topLevelKeys) ||
    payload.format !== DELTREL_BROWSER_MODEL_MANIFEST_SCHEMA_ID ||
    payload.schema_version !== DELTREL_BROWSER_MODEL_MANIFEST_VERSION ||
    payload.precision !== DELTREL_BROWSER_MODEL_PRECISION ||
    payload.weights !== 'ema' ||
    typeof payload.created_ns !== 'number' ||
    !Number.isFinite(payload.created_ns) ||
    payload.created_ns <= 0 ||
    !isRecord(rules) ||
    !hasExactKeys(rules, [
      'schema_id',
      'hash',
      'mode',
      'pie_rule',
      'handicap',
      'rings',
      'variants',
    ]) ||
    rules.schema_id !== DELTREL_RULES_SCHEMA_ID ||
    rules.hash !== DELTREL_RULES_HASH ||
    rules.mode !== 'double' ||
    rules.pie_rule !== false ||
    rules.handicap !== 1 ||
    !isRecord(rules.variants) ||
    !hasExactKeys(rules.variants, ['modes', 'handicap_min', 'handicap_max', 'pie_allowed']) ||
    !sameShape(rules.variants.modes, ['classic', 'double']) ||
    rules.variants.handicap_min !== 1 ||
    rules.variants.handicap_max !== 9 ||
    rules.variants.pie_allowed !== true ||
    !Array.isArray(rules.rings) ||
    rules.rings.length !== SUPPORTED_RINGS.length ||
    !rules.rings.every(
      (rings, index) => rings === SUPPORTED_RINGS[index],
    ) ||
    !isRecord(features) ||
    !hasExactKeys(features, [
      'schema_id',
      'version',
      'hash',
      'node_feature_count',
      'global_feature_count',
    ]) ||
    features.schema_id !== DELTREL_FEATURE_SCHEMA_ID ||
    features.version !== DELTREL_FEATURE_SCHEMA_VERSION ||
    features.hash !== DELTREL_FEATURE_SCHEMA_HASH ||
    features.node_feature_count !== DELTREL_NODE_FEATURE_DIM ||
    features.global_feature_count !== DELTREL_GLOBAL_FEATURE_DIM ||
    !isRecord(actions) ||
    !hasExactKeys(actions, ['schema_id', 'types']) ||
    actions.schema_id !== DELTREL_ACTION_LAYOUT_SCHEMA_ID ||
    !sameShape(actions.types, ['place', 'swap']) ||
    !isRecord(outcome) ||
    !hasExactKeys(outcome, ['classes', 'value']) ||
    !sameShape(outcome.classes, ['loss', 'win']) ||
    outcome.value !== 'P(win)-P(loss)'
  ) {
    throw new DeltrelAiError(
      'unavailable',
      'Local AI assets do not match this game build.',
    );
  }
  if (
    typeof payload.model_version !== 'string' ||
    !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(payload.model_version)
  ) {
    throw new DeltrelAiError('unavailable', 'Local AI model identity is invalid.');
  }

  if (
    !isRecord(architecture) ||
    !hasExactKeys(architecture, [
      'name',
      'schema_version',
      'all_size',
      'parameter_count',
      'config',
    ]) ||
    architecture.name !== 'GraphResTNet' ||
    architecture.schema_version !== DELTREL_BROWSER_MODEL_ARCHITECTURE_VERSION ||
    architecture.all_size !== true ||
    !positiveInteger(architecture.parameter_count, Number.MAX_SAFE_INTEGER) ||
    !isRecord(architecture.config) ||
    architecture.config.node_feature_dim !== DELTREL_NODE_FEATURE_DIM ||
    architecture.config.global_feature_dim !== DELTREL_GLOBAL_FEATURE_DIM ||
    architecture.config.feature_schema_version !== DELTREL_FEATURE_SCHEMA_VERSION ||
    !isRecord(artifacts) ||
    !hasExactKeys(artifacts, ['onnx', 'checkpoint']) ||
    !validateArtifact(artifacts.onnx, 'onnx', true) ||
    !validateArtifact(artifacts.checkpoint, 'pt', false) ||
    !isRecord(tensors) ||
    !hasExactKeys(tensors, ['inputs', 'outputs']) ||
    !validateTensorMap(tensors.inputs, [
      ['node_features', 'float16', ['batch', 'nodes', DELTREL_NODE_FEATURE_DIM]],
      ['global_features', 'float16', ['batch', DELTREL_GLOBAL_FEATURE_DIM]],
      ['neighbor_index', 'int64', ['batch', 'nodes', 'degree']],
      ['neighbor_mask', 'bool', ['batch', 'nodes', 'degree']],
      ['neighbor_edge_type', 'int64', ['batch', 'nodes', 'degree']],
      ['node_mask', 'bool', ['batch', 'nodes']],
      ['legal_action_mask', 'bool', ['batch', 'nodes']],
      ['rings', 'int64', ['batch']],
    ]) ||
    !validateTensorMap(tensors.outputs, [
      ['policy_logits', 'float16', ['batch', 'nodes']],
      ['outcome_logits', 'float16', ['batch', 2]],
      ['score_margin_logits', 'float16', ['batch', 303]],
      ['ownership_logits', 'float16', ['batch', 'nodes', 3]],
      ['alive_logits', 'float16', ['batch', 'nodes']],
      ['soft_policy_logits', 'float16', ['batch', 'nodes']],
    ]) ||
    !isRecord(search) ||
    !hasExactKeys(search, [
      'simulations',
      'max_considered',
      'c_visit',
      'c_scale',
      'swap_dead_zone',
      ...['first_visit_batch_size', 'subtree_reuse', 'subtree_reuse_max_nodes']
        .filter((key) => Object.hasOwn(search, key)),
    ]) ||
    !positiveInteger(search.simulations, MAX_BROWSER_AI_SIMULATIONS) ||
    !positiveInteger(search.max_considered, MAX_BROWSER_AI_MAX_CONSIDERED) ||
    !positiveFinite(search.c_visit) ||
    !positiveFinite(search.c_scale) ||
    typeof search.swap_dead_zone !== 'number' ||
    !Number.isFinite(search.swap_dead_zone) ||
    search.swap_dead_zone < 0 ||
    search.swap_dead_zone >= 1 ||
    (search.first_visit_batch_size !== undefined &&
      !positiveInteger(search.first_visit_batch_size, MAX_BROWSER_AI_FIRST_VISIT_BATCH_SIZE)) ||
    (search.subtree_reuse !== undefined && typeof search.subtree_reuse !== 'boolean') ||
    (search.subtree_reuse_max_nodes !== undefined &&
      !positiveInteger(search.subtree_reuse_max_nodes, MAX_BROWSER_AI_SUBTREE_REUSE_NODES)) ||
    !isRecord(payload.training)
  ) {
    throw new DeltrelAiError('unavailable', 'Local AI model manifest fields are invalid.');
  }

  const onnx = artifacts.onnx;
  return {
    format: DELTREL_BROWSER_MODEL_MANIFEST_SCHEMA_ID,
    schemaVersion: DELTREL_BROWSER_MODEL_MANIFEST_VERSION,
    modelVersion: payload.model_version as string,
    weights: 'ema',
    rulesSchema: DELTREL_RULES_SCHEMA_ID,
    rulesHash: DELTREL_RULES_HASH,
    featureSchema: DELTREL_FEATURE_SCHEMA_ID,
    featureSchemaVersion: DELTREL_FEATURE_SCHEMA_VERSION,
    featureSchemaHash: DELTREL_FEATURE_SCHEMA_HASH,
    actionLayout: DELTREL_ACTION_LAYOUT_SCHEMA_ID,
    actionLayoutVersion: DELTREL_ACTION_LAYOUT_VERSION,
    outcome: {
      classes: ['loss', 'win'],
      value: 'P(win)-P(loss)',
    },
    wasm: {
      moduleUrl: DELTREL_WASM_MODULE_PATH,
      binaryUrl: DELTREL_WASM_BINARY_PATH,
    },
    model: {
      format: 'onnx',
      precision: DELTREL_BROWSER_MODEL_PRECISION,
      url: `/models/deltrel/${onnx.file as string}`,
      sha256: `sha256:${onnx.sha256 as string}`,
      bytes: onnx.bytes as number,
      opset: onnx.opset as number,
      inputs: [...DELTREL_MODEL_INPUT_NAMES],
      outputs: [...DELTREL_MODEL_OUTPUT_NAMES],
    },
    search: {
      simulations: search.simulations,
      maxConsidered: search.max_considered,
      maximumSimulations: MAX_BROWSER_AI_SIMULATIONS,
      maximumMaxConsidered: MAX_BROWSER_AI_MAX_CONSIDERED,
      cVisit: search.c_visit,
      cScale: search.c_scale,
      swapDeadZone: search.swap_dead_zone,
      ...(search.first_visit_batch_size !== undefined
        ? { firstVisitBatchSize: search.first_visit_batch_size as number } : {}),
      ...(search.subtree_reuse !== undefined ? { subtreeReuse: search.subtree_reuse as boolean } : {}),
      ...(search.subtree_reuse_max_nodes !== undefined
        ? { subtreeReuseMaxNodes: search.subtree_reuse_max_nodes as number } : {}),
    },
  };
}
