'use client';

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';
import {
  getLocalAiStatus,
  getServerLocalAiStatus,
  subscribeLocalAiStatus,
} from '@/lib/deltrel/ai/local-ai-status';

/** Prepare the published champion on a fresh visit; a ready session survives screen changes. */
export function useBrowserAiPreparation() {
  const status = useSyncExternalStore(subscribeLocalAiStatus, getLocalAiStatus, getServerLocalAiStatus);
  const [authorized, setAuthorized] = useState(() => getLocalAiStatus().phase === 'ready');
  const [notice, setNotice] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  const prepare = useCallback(async (): Promise<boolean> => {
    if (controllerRef.current) return false;
    const controller = new AbortController();
    controllerRef.current = controller;
    setNotice(null);
    try {
      const { prepareLocalAi } = await import('@/lib/deltrel/ai/local-client');
      if (controller.signal.aborted) return false;
      await prepareLocalAi({ signal: controller.signal });
      if (controller.signal.aborted) return false;
      // Keep this authorization through later worker/cache initialization stages.
      setAuthorized(true);
      return true;
    } catch {
      if (!controller.signal.aborted && getLocalAiStatus().phase !== 'error') {
        setNotice('AI could not be prepared. Please try again.');
      }
      return false;
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    // React Strict Mode replays mount effects. Start only after its discarded
    // effect has cleaned up, so it cannot cancel the actual download.
    queueMicrotask(() => {
      if (mounted && getLocalAiStatus().phase === 'idle') void prepare();
    });
    return () => {
      mounted = false;
      controllerRef.current?.abort();
      controllerRef.current = null;
    };
  }, [prepare]);

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setAuthorized(false);
    setNotice('Preparation cancelled. You can try again whenever you’re ready.');
  }, []);

  return { status, authorized, notice, prepare, cancel };
}
