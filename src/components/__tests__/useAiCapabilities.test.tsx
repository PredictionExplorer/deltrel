import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { checkAiCapabilities, checkServerAiCapability, type AiCapabilities } from '@/lib/deltrel/ai/capabilities';
import { useAiCapabilities } from '../useAiCapabilities';

vi.mock('@/lib/deltrel/ai/capabilities', async () => ({
  ...await vi.importActual<typeof import('@/lib/deltrel/ai/capabilities')>('@/lib/deltrel/ai/capabilities'),
  checkAiCapabilities: vi.fn(), checkServerAiCapability: vi.fn(),
}));
const offline: AiCapabilities = {
  local: { status: 'available', label: 'AI' },
  server: { status: 'unavailable', label: 'Cloud AI', code: 'offline', reason: 'Offline', retryable: true },
};

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(checkAiCapabilities).mockResolvedValue(offline);
  vi.mocked(checkServerAiCapability).mockResolvedValue({ status: 'available', label: 'Cloud AI' });
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.clearAllMocks(); });

it('recovers offline availability through bounded health probes and stops probing after recovery', async () => {
  const { result, unmount } = renderHook(() => useAiCapabilities(null));
  await act(async () => {});
  expect(result.current.capabilities.server.status).toBe('unavailable');
  await act(async () => { await vi.advanceTimersByTimeAsync(30_000); });
  expect(result.current.capabilities.server.status).toBe('available');
  expect(checkServerAiCapability).toHaveBeenCalledOnce();
  await act(async () => { await vi.advanceTimersByTimeAsync(90_000); });
  expect(checkServerAiCapability).toHaveBeenCalledOnce();
  expect(checkAiCapabilities).toHaveBeenCalledOnce();
  unmount();
  expect(vi.mocked(checkServerAiCapability).mock.calls[0][0]?.aborted).toBe(true);
});

it('skips health polls in a hidden tab and checks when it becomes visible', async () => {
  const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden');
  renderHook(() => useAiCapabilities(null));
  await act(async () => { await vi.advanceTimersByTimeAsync(30_000); });
  expect(checkServerAiCapability).not.toHaveBeenCalled();
  visibility.mockReturnValue('visible');
  await act(async () => { document.dispatchEvent(new Event('visibilitychange')); });
  expect(checkServerAiCapability).toHaveBeenCalledOnce();
  visibility.mockRestore();
});
