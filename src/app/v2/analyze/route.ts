import {
  DELTREL_AI_PROXY_ANALYZE_PATH,
  proxyDeltrelAiRequest,
} from '@/lib/deltrel/ai/server-proxy';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 65;

export async function POST(request: Request): Promise<Response> {
  return proxyDeltrelAiRequest(request, DELTREL_AI_PROXY_ANALYZE_PATH);
}
