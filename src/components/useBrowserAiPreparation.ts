'use client';

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';
import {
  getLocalAiStatus,
  getServerLocalAiStatus,
  subscribeLocalAiStatus,
} from '@/lib/deltrel/ai/local-ai-status';

/** Subscribing reads status only; model preparation always begins with a user action. */
export function useBrowserAiPreparation() {
  const status = useSyncExternalStore(subscribeLocalAiStatus, getLocalAiStatus, getServerLocalAiStatus);
  const [authorized, setAuthorized] = useState(() => getLocalAiStatus().phase === 'ready');
  const [notice, setNotice] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => () => controllerRef.current?.abort(), []);

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
        setNotice('Browser AI could not be prepared. Please try again.');
      }
      return false;
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
    }
  }, []);

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setAuthorized(false);
    setNotice('Preparation cancelled. You can try again whenever you’re ready.');
  }, []);

  return { status, authorized, notice, prepare, cancel };
}
