/** Must match the installed ONNX Runtime package and its copied web assets. */
export const DELTREL_ORT_VERSION = '1.27.0' as const;
export const DELTREL_ORT_ASSET_PREFIX = `/onnxruntime/${DELTREL_ORT_VERSION}/` as const;
/** The webgpu entry point in this release uses the Asyncify runtime. */
export const DELTREL_ORT_ASSET_FILES = [
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
] as const;
