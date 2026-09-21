import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { prepareLocalAi } from '@/lib/deltrel/ai/local-client';
import { publishLocalAiStatus } from '@/lib/deltrel/ai/local-ai-status';
import { useBrowserAiPreparation } from '../useBrowserAiPreparation';

vi.mock('@/lib/deltrel/ai/local-client', async () => ({
  ...await vi.importActual<typeof import('@/lib/deltrel/ai/local-client')>('@/lib/deltrel/ai/local-client'),
  prepareLocalAi: vi.fn(),
}));
const ready = { modelVersion: 'browser-v3', bytes: 37_577_312, backend: 'wasm' as const, cached: true };

beforeEach(() => {
  publishLocalAiStatus({ phase: 'idle' });
  vi.mocked(prepareLocalAi).mockReset();
});
afterEach(cleanup);

describe('useBrowserAiPreparation', () => {
  it('does not prepare on mount and retains authorization during later worker initialization', async () => {
    const { result } = renderHook(useBrowserAiPreparation);
    expect(prepareLocalAi).not.toHaveBeenCalled();
    expect(result.current.authorized).toBe(false);
    vi.mocked(prepareLocalAi).mockImplementation(async () => {
      publishLocalAiStatus({ phase: 'ready', info: ready });
      return ready;
    });
    await act(async () => { expect(await result.current.prepare()).toBe(true); });
    expect(result.current.authorized).toBe(true);
    act(() => publishLocalAiStatus({ phase: 'initializing', loadedBytes: ready.bytes, totalBytes: ready.bytes, modelVersion: ready.modelVersion, cached: true }));
    expect(result.current.authorized).toBe(true);
    expect(result.current.status.phase).toBe('initializing');
  });

  it('cancels a pending preparation and ignores a late completion', async () => {
    let finish!: (value: typeof ready) => void;
    vi.mocked(prepareLocalAi).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    const { result } = renderHook(useBrowserAiPreparation);
    let pending!: Promise<boolean>;
    act(() => { pending = result.current.prepare(); });
    await waitFor(() => expect(prepareLocalAi).toHaveBeenCalledOnce());
    const signal = vi.mocked(prepareLocalAi).mock.calls[0][0]!.signal!;
    act(() => result.current.cancel());
    expect(signal.aborted).toBe(true);
    await act(async () => { finish(ready); expect(await pending).toBe(false); });
    expect(result.current.authorized).toBe(false);
    expect(result.current.notice).toMatch(/cancelled/i);
  });

  it('starts only one preparation and aborts it when the screen unmounts', async () => {
    let finish!: (value: typeof ready) => void;
    vi.mocked(prepareLocalAi).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    const { result, unmount } = renderHook(useBrowserAiPreparation);
    let pending!: Promise<boolean>;
    act(() => { pending = result.current.prepare(); });
    await act(async () => { expect(await result.current.prepare()).toBe(false); });
    await waitFor(() => expect(prepareLocalAi).toHaveBeenCalledOnce());
    const signal = vi.mocked(prepareLocalAi).mock.calls[0][0]!.signal!;
    unmount();
    expect(signal.aborted).toBe(true);
    finish(ready);
    expect(await pending).toBe(false);
  });

  it('accepts already prepared state without a network operation and gives a retryable UI notice for import/runtime failure', async () => {
    publishLocalAiStatus({ phase: 'ready', info: ready });
    const { result } = renderHook(useBrowserAiPreparation);
    expect(result.current.authorized).toBe(true);
    expect(prepareLocalAi).not.toHaveBeenCalled();
    vi.mocked(prepareLocalAi).mockRejectedValue(new Error('worker failed'));
    await act(async () => { expect(await result.current.prepare()).toBe(false); });
    expect(result.current.notice).toMatch(/could not be prepared/i);
  });
});
