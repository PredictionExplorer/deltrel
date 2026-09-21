import {
  DELTREL_ACTION_LAYOUT_SCHEMA_ID,
  DELTREL_FEATURE_SCHEMA_ID,
  DELTREL_RULES_HASH,
  DELTREL_RULES_SCHEMA_ID,
} from '../rules';
import type { ControllerType } from './controllers';
import type { DeltrelAiSearchBudget } from './decision';
import {
  DELTREL_FEATURE_SCHEMA_HASH,
  DELTREL_FEATURE_SCHEMA_VERSION,
} from './protocol';
import { configuredServerHealthUrl } from './server-client';

export type AiCapability =
  | { status: 'checking'; label: string }
  | {
      status: 'available';
      label: string;
      search?: AiSearchCapability;
      device?: string;
      champion?: AiChampionCapability;
      browserModel?: { modelVersion: string; bytes: number; sha256: string };
    }
  | {
      status: 'unavailable';
      label: string;
      code: string;
      reason: string;
      retryable: boolean;
    };

export interface AiSearchCapability {
  default: DeltrelAiSearchBudget;
  maximum: DeltrelAiSearchBudget;
  presets: Readonly<Record<string, DeltrelAiSearchBudget>>;
}

export interface AiChampionCapability {
  role: 'champion';
  modelVersion: string;
  modelStep: number;
  modelIdentity: string;
}

export interface AiCapabilities {
  server: AiCapability;
  local: AiCapability;
}

export const INITIAL_AI_CAPABILITIES: AiCapabilities = {
  server: { status: 'checking', label: 'Server AI' },
  local: { status: 'checking', label: 'Local AI' },
};

const LOCAL_MANIFEST_PATH = '/models/deltrel/manifest.json';
const CAPABILITY_TIMEOUT_MS = 5_000;
const CAPABILITY_BODY_BYTES = 256 * 1024;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return (
    actual.length === expected.length &&
    actual.every((key, index) => key === expected[index])
  );
}

function parseHealthBudget(value: unknown): DeltrelAiSearchBudget {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ['simulations', 'max_considered']) ||
    typeof value.simulations !== 'number' ||
    !Number.isSafeInteger(value.simulations) ||
    value.simulations <= 0 ||
    typeof value.max_considered !== 'number' ||
    !Number.isSafeInteger(value.max_considered) ||
    value.max_considered <= 0
  ) {
    throw new Error('invalid search budget');
  }
  return {
    simulations: value.simulations,
    maxConsidered: value.max_considered,
  };
}

function parseHealthSearch(value: unknown): AiSearchCapability {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ['defaults', 'maximums', 'presets']) ||
    !isRecord(value.presets) ||
    Object.keys(value.presets).length === 0
  ) {
    throw new Error('invalid search metadata');
  }
  const defaultBudget = parseHealthBudget(value.defaults);
  const maximum = parseHealthBudget(value.maximums);
  if (
    defaultBudget.simulations > maximum.simulations ||
    defaultBudget.maxConsidered > maximum.maxConsidered
  ) {
    throw new Error('invalid search defaults');
  }
  const presets: Record<string, DeltrelAiSearchBudget> = {};
  for (const [name, rawBudget] of Object.entries(value.presets)) {
    if (!/^[a-z][a-z0-9_-]{0,31}$/.test(name)) {
      throw new Error('invalid search preset name');
    }
    const budget = parseHealthBudget(rawBudget);
    if (
      budget.simulations > maximum.simulations ||
      budget.maxConsidered > maximum.maxConsidered
    ) {
      throw new Error('invalid search preset');
    }
    presets[name] = budget;
  }
  return { default: defaultBudget, maximum, presets };
}

function parseChampionCapability(model: Record<string, unknown>): AiChampionCapability | undefined {
  if (!('role' in model) && !('model_identity' in model)) return undefined;
  if (
    model.role !== 'champion' ||
    typeof model.model_version !== 'string' ||
    model.model_version.length === 0 ||
    model.model_version.length > 256 ||
    typeof model.model_step !== 'number' ||
    !Number.isSafeInteger(model.model_step) ||
    model.model_step < 0 ||
    typeof model.model_identity !== 'string' ||
    model.model_identity.length === 0 ||
    model.model_identity.length > 256
  ) {
    throw new Error('invalid champion metadata');
  }
  return {
    role: 'champion',
    modelVersion: model.model_version,
    modelStep: model.model_step,
    modelIdentity: model.model_identity,
  };
}

function unavailable(
  label: string,
  code: string,
  reason: string,
  retryable: boolean,
): AiCapability {
  return { status: 'unavailable', label, code, reason, retryable };
}

function linkedTimeout(signal?: AbortSignal): {
  signal: AbortSignal;
  dispose: () => void;
  timedOut: () => boolean;
} {
  const controller = new AbortController();
  let didTimeOut = false;
  const abort = () => controller.abort(signal?.reason);
  signal?.addEventListener('abort', abort, { once: true });
  if (signal?.aborted) abort();
  const timeout = setTimeout(() => {
    didTimeOut = true;
    controller.abort();
  }, CAPABILITY_TIMEOUT_MS);
  return {
    signal: controller.signal,
    timedOut: () => didTimeOut,
    dispose: () => {
      clearTimeout(timeout);
      signal?.removeEventListener('abort', abort);
    },
  };
}

async function readJsonResponse(response: Response): Promise<unknown> {
  const contentType = response.headers.get('Content-Type')?.split(';', 1)[0].trim().toLowerCase();
  if (contentType !== 'application/json') {
    await response.body?.cancel();
    throw new Error('response is not JSON');
  }
  const declared = response.headers.get('Content-Length');
  if (declared !== null && (!Number.isSafeInteger(Number(declared)) ||
      Number(declared) < 0 || Number(declared) > CAPABILITY_BODY_BYTES)) {
    await response.body?.cancel();
    throw new Error('response is too large');
  }
  if (!response.body) throw new Error('response is empty');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let bytes = 0;
  let text = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > CAPABILITY_BODY_BYTES) {
        await reader.cancel();
        throw new Error('response is too large');
      }
      text += decoder.decode(value, { stream: true });
    }
    text += decoder.decode();
  } finally {
    reader.releaseLock();
  }
  return JSON.parse(text);
}

export function localBrowserCapabilityIssue(
  scope: typeof globalThis = globalThis,
): string | null {
  if (
    typeof scope.fetch !== 'function' ||
    typeof scope.AbortController !== 'function' ||
    typeof scope.TextEncoder !== 'function'
  ) {
    return 'Required browser networking APIs are not supported.';
  }
  if (typeof scope.Worker !== 'function') return 'Web Workers are not supported.';
  if (typeof scope.WebAssembly !== 'object') return 'WebAssembly is not supported.';
  if (typeof scope.BigInt !== 'function') return 'BigInt is not supported.';
  if (
    typeof scope.BigInt64Array !== 'function' ||
    typeof scope.BigUint64Array !== 'function'
  ) {
    return 'BigInt typed arrays are not supported.';
  }
  if (!scope.crypto?.subtle) return 'Web Crypto is not supported.';
  return null;
}

export async function checkServerAiCapability(
  signal?: AbortSignal,
): Promise<AiCapability> {
  if (typeof fetch !== 'function' || typeof AbortController !== 'function') {
    return unavailable(
      'Server AI',
      'browser_unsupported',
      'Required browser networking APIs are not supported.',
      false,
    );
  }
  if (typeof BigInt !== 'function') {
    return unavailable(
      'Server AI',
      'browser_unsupported',
      'AI controllers require BigInt browser support.',
      false,
    );
  }
  const timeout = linkedTimeout(signal);
  try {
    timeout.signal.throwIfAborted();
    const requestId =
      typeof crypto.randomUUID === 'function'
        ? crypto.randomUUID()
        : `health-${Date.now().toString(36)}`;
    const response = await fetch(configuredServerHealthUrl(), {
      cache: 'no-store',
      headers: { Accept: 'application/json', 'X-Request-ID': requestId },
      signal: timeout.signal,
    });
    const payload = await readJsonResponse(response);
    if (!response.ok) {
      const reason =
        isRecord(payload) &&
        isRecord(payload.error) &&
        typeof payload.error.message === 'string'
          ? payload.error.message
          : 'Server AI is not ready.';
      return unavailable('Server AI', 'server_unavailable', reason, response.status >= 500);
    }
    if (
      !isRecord(payload) ||
      payload.api_schema_version !== 3 ||
      (payload.status !== 'ok' && payload.status !== 'degraded') ||
      !isRecord(payload.rules) ||
      payload.rules.schema_id !== DELTREL_RULES_SCHEMA_ID ||
      payload.rules.hash !== DELTREL_RULES_HASH ||
      !isRecord(payload.features) ||
      payload.features.schema_id !== DELTREL_FEATURE_SCHEMA_ID ||
      payload.features.version !== DELTREL_FEATURE_SCHEMA_VERSION ||
      payload.features.hash !== DELTREL_FEATURE_SCHEMA_HASH ||
      !isRecord(payload.actions) ||
      payload.actions.schema_id !== DELTREL_ACTION_LAYOUT_SCHEMA_ID ||
      !isRecord(payload.model) ||
      payload.model.ready !== true
    ) {
      return unavailable(
        'Server AI',
        'server_incompatible',
        'Server AI is incompatible with this game build.',
        false,
      );
    }
    try {
      let device: string | undefined;
      if (payload.device !== undefined) {
        if (
          typeof payload.device !== 'string' ||
          payload.device.length === 0 ||
          payload.device.length > 128
        ) {
          throw new Error('invalid device metadata');
        }
        device = payload.device;
      }
      const search =
        payload.search === undefined ? undefined : parseHealthSearch(payload.search);
      const champion = parseChampionCapability(payload.model);
      return {
        status: 'available',
        label: 'Server AI',
        ...(device === undefined ? {} : { device }),
        ...(search === undefined ? {} : { search }),
        ...(champion === undefined ? {} : { champion }),
      };
    } catch {
      return unavailable(
        'Server AI',
        'server_incompatible',
        'Server AI capability metadata is invalid.',
        false,
      );
    }
  } catch {
    return unavailable(
      'Server AI',
      timeout.timedOut() ? 'server_timeout' : 'server_unavailable',
      timeout.timedOut() ? 'Server AI health check timed out.' : 'Server AI is unavailable.',
      true,
    );
  } finally {
    timeout.dispose();
  }
}

async function assetExists(url: string, signal: AbortSignal): Promise<boolean> {
  const response = await fetch(url, {
    method: 'HEAD',
    cache: 'no-store',
    signal,
  });
  return response.ok;
}

export async function checkLocalAiCapability(
  signal?: AbortSignal,
): Promise<AiCapability> {
  const issue = localBrowserCapabilityIssue();
  if (issue) return unavailable('Local AI', 'browser_unsupported', issue, false);

  const timeout = linkedTimeout(signal);
  try {
    timeout.signal.throwIfAborted();
    const response = await fetch(LOCAL_MANIFEST_PATH, {
      cache: 'no-store',
      signal: timeout.signal,
    });
    if (!response.ok) {
      return unavailable(
        'Local AI',
        'local_assets_missing',
        'The browser AI model is not available yet.',
        false,
      );
    }
    const payload = await readJsonResponse(response);
    const { parseDeltrelBrowserModelManifest } = await import('./manifest');
    const manifest = parseDeltrelBrowserModelManifest(payload);
    const assets = await Promise.all([
      assetExists(manifest.wasm.moduleUrl, timeout.signal),
      assetExists(manifest.wasm.binaryUrl, timeout.signal),
      assetExists(manifest.model.url, timeout.signal),
    ]);
    if (assets.some((exists) => !exists)) {
      return unavailable(
        'Local AI',
        'local_assets_missing',
        'The browser AI download is temporarily unavailable.',
        false,
      );
    }
    const defaultBudget = {
      simulations: manifest.search.simulations,
      maxConsidered: manifest.search.maxConsidered,
    };
    const maximum = {
      simulations: manifest.search.maximumSimulations,
      maxConsidered: manifest.search.maximumMaxConsidered,
    };
    return {
      status: 'available',
      label: 'Local AI',
      browserModel: {
        modelVersion: manifest.modelVersion,
        bytes: manifest.model.bytes,
        sha256: manifest.model.sha256,
      },
      search: {
        default: defaultBudget,
        maximum,
        presets: {
          quick: {
            simulations: Math.min(128, defaultBudget.simulations),
            maxConsidered: Math.min(8, defaultBudget.maxConsidered),
          },
          strong: defaultBudget,
          maximum,
        },
      },
    };
  } catch {
    return unavailable(
      'Local AI',
      timeout.timedOut() ? 'local_timeout' : 'local_unavailable',
      timeout.timedOut() ? 'Local AI preflight timed out.' : 'Local AI is unavailable.',
      timeout.timedOut(),
    );
  } finally {
    timeout.dispose();
  }
}

export async function checkAiCapabilities(signal?: AbortSignal): Promise<AiCapabilities> {
  const [server, local] = await Promise.all([
    checkServerAiCapability(signal),
    checkLocalAiCapability(signal),
  ]);
  return { server, local };
}

export function capabilityForController(
  capabilities: AiCapabilities,
  controller: ControllerType,
): AiCapability {
  if (controller === 'human') return { status: 'available', label: 'Human' };
  return capabilities[controller];
}
