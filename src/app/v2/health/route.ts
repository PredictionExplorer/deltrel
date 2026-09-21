import {
  DELTREL_AI_PROXY_HEALTH_PATH,
  proxyDeltrelAiRequest,
} from '@/lib/deltrel/ai/server-proxy';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 10;

export async function GET(request: Request): Promise<Response> {
  return proxyDeltrelAiRequest(request, DELTREL_AI_PROXY_HEALTH_PATH);
}
