'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  INITIAL_AI_CAPABILITIES,
  checkAiCapabilities,
  checkServerAiCapability,
  type AiCapabilities,
} from '@/lib/deltrel/ai/capabilities';

/** Availability probes are cheap and never submit searches or prepare a model. */
export function useAiCapabilities(preparedModelVersion: string | null) {
  const [capabilities, setCapabilities] = useState<AiCapabilities>(INITIAL_AI_CAPABILITIES);
  const [refresh, setRefresh] = useState(0);
  const latest = useRef(capabilities);
  useEffect(() => { latest.current = capabilities; }, [capabilities]);

  useEffect(() => {
    const controller = new AbortController();
    let probing = true;
    let lastProbe = Date.now();
    void checkAiCapabilities(controller.signal).then((result) => {
      if (!controller.signal.aborted) setCapabilities(result);
    }).finally(() => { probing = false; });
    const probeCloud = () => {
      if (document.visibilityState === 'hidden' || probing ||
          latest.current.server.status !== 'unavailable' || Date.now() - lastProbe < 15_000) return;
      probing = true;
      lastProbe = Date.now();
      void checkServerAiCapability(controller.signal).then((server) => {
        if (!controller.signal.aborted) setCapabilities((previous) => ({ ...previous, server }));
      }).finally(() => { probing = false; });
    };
    const timer = window.setInterval(probeCloud, 30_000);
    window.addEventListener('online', probeCloud);
    document.addEventListener('visibilitychange', probeCloud);
    return () => {
      controller.abort();
      window.clearInterval(timer);
      window.removeEventListener('online', probeCloud);
      document.removeEventListener('visibilitychange', probeCloud);
    };
  }, [refresh, preparedModelVersion]);

  const checkAgain = useCallback(() => setRefresh((value) => value + 1), []);
  const markCloudUnavailable = useCallback(() => {
    setCapabilities((previous) => ({ ...previous, server: {
      status: 'unavailable', label: 'Cloud AI', code: 'server_unavailable',
      reason: 'Cloud AI is currently offline or busy. Your game is saved.', retryable: true,
    } }));
  }, []);
  return { capabilities, checkAgain, markCloudUnavailable };
}
