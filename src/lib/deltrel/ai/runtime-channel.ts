import descriptor from '../../../../training/deltreltrain/browser_runtime.json' with { type: 'json' };

/** Trusted, build-pinned execution channel; model JSON cannot select executable URLs. */
export const DELTREL_RUNTIME_ID = descriptor.runtime_content_sha256;
export const DELTREL_RUNTIME_SEARCH_ALGORITHM = descriptor.search_algorithm;
export const DELTREL_RUNTIME_CHANNEL_PATH =
  `/models/deltrel/manifest-runtime-${DELTREL_RUNTIME_ID}.json`;
export const DELTREL_RUNTIME_DIRECTORY =
  `wasm-${descriptor.rules_hash.slice('fnv1a64:'.length)}-${DELTREL_RUNTIME_ID}`;

function artifact(value: typeof descriptor.module) {
  return Object.freeze({
    url: `/models/deltrel/${DELTREL_RUNTIME_DIRECTORY}/${value.path}`,
    sha256: `sha256:${value.sha256}`,
    bytes: value.bytes,
  });
}

export const DELTREL_RUNTIME_ARTIFACTS = Object.freeze({
  module: artifact(descriptor.module),
  binary: artifact(descriptor.binary),
});
