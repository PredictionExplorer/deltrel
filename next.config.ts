import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  webpack(config) {
    // Runtime files live at explicit same-origin URLs; avoid embedding a second
    // copy of the large ONNX WASM binary in the generated JavaScript assets.
    config.resolve.conditionNames = [
      'onnxruntime-web-use-extern-wasm',
      ...(config.resolve.conditionNames ?? []),
    ];
    return config;
  },
  headers() {
    return [
      {
        source: '/onnxruntime/:version/:file',
        headers: [{ key: 'Cache-Control', value: 'public, max-age=31536000, immutable' }],
      },
      {
        source: '/models/deltrel/:file',
        headers: [{ key: 'Cache-Control', value: 'public, max-age=31536000, immutable' }],
      },
      {
        // The pointer is mutable; the model files it names are immutable.
        source: '/models/deltrel/manifest.json',
        headers: [{ key: 'Cache-Control', value: 'public, max-age=0, must-revalidate' }],
      },
      {
        // The directory keys the rules, while search implementation can evolve.
        source: '/models/deltrel/wasm-:revision/:file',
        headers: [{ key: 'Cache-Control', value: 'public, max-age=0, must-revalidate' }],
      },
    ];
  },
};

export default nextConfig;
