import type { MetadataRoute } from 'next';

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: 'Deltrel — A game of connection',
    short_name: 'Deltrel',
    description: 'Reach the shore. Join your networks. A thoughtful strategy game for two.',
    start_url: '/',
    display: 'standalone',
    background_color: '#062e32',
    theme_color: '#0b393c',
    icons: [{ src: '/icon.svg', sizes: 'any', type: 'image/svg+xml', purpose: 'any' }],
  };
}
