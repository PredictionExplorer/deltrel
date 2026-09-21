'use client';

import { useId } from 'react';
import { Download, Check } from 'lucide-react';
import type { AiCapability } from '@/lib/deltrel/ai/capabilities';
import type { LocalAiStatus } from '@/lib/deltrel/ai/local-ai-status';
import styles from './BrowserAiPreparation.module.css';

export function browserAiIsPreparing(status: LocalAiStatus): boolean {
  return status.phase === 'checking' || status.phase === 'downloading' || status.phase === 'verifying' || status.phase === 'initializing';
}

export function formatModelBytes(bytes: number): string {
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  if (bytes >= 1_000) return `${(bytes / 1_000).toFixed(0)} KB`;
  return `${bytes} bytes`;
}

export interface BrowserAiPreparationProps {
  status: LocalAiStatus;
  authorized: boolean;
  capability?: AiCapability;
  selected?: boolean;
  notice?: string | null;
  inGame?: boolean;
  onPrepare: () => void;
  onCancel: () => void;
  onSelect?: () => void;
  onCheck?: () => void;
}

export function BrowserAiPreparation({ status, authorized, capability, selected = false, notice, inGame = false, onPrepare, onCancel, onSelect, onCheck }: BrowserAiPreparationProps) {
  const titleId = useId();
  const detailId = useId();
  const preparing = browserAiIsPreparing(status);
  const published = capability?.status === 'available' ? capability.browserModel : undefined;
  const modelBytes = status.phase === 'ready' ? status.info.bytes : 'totalBytes' in status ? status.totalBytes ?? published?.bytes : published?.bytes;
  const modelVersion = status.phase === 'ready' ? status.info.modelVersion : 'modelVersion' in status ? status.modelVersion ?? published?.modelVersion : published?.modelVersion;
  const available = capability?.status !== 'unavailable' && capability?.status !== 'checking';
  const ready = status.phase === 'ready' && authorized && available;
  const progress = status.phase === 'downloading' && status.totalBytes !== null && status.totalBytes > 0
    ? Math.min(100, Math.max(0, status.loadedBytes / status.totalBytes * 100)) : null;
  const stage = status.phase === 'checking' ? 'Checking the latest model…'
    : status.phase === 'downloading' ? status.cached ? 'Loading the saved model…' : 'Downloading browser AI…'
      : status.phase === 'verifying' ? 'Checking the download…'
        : status.phase === 'initializing' ? 'Preparing the engine…'
          : ready ? 'Ready on this device'
            : capability?.status === 'checking' ? 'Checking availability…'
              : capability?.status === 'unavailable' ? 'Not available'
                : status.phase === 'error' ? 'Preparation interrupted' : 'Ready to prepare';

  return (
    <section className={styles.panel} aria-labelledby={titleId} aria-busy={preparing}>
      <header className={styles.header}>
        <h2 id={titleId}>{ready ? <Check size={17} aria-hidden /> : <Download size={17} aria-hidden />} Browser AI</h2>
        <span className={styles.badge}>{ready ? 'On-device' : 'No installation'}</span>
      </header>
      <p className={styles.description}>Play the trained champion right in your browser. Once prepared, moves are calculated on this device.</p>
      <p id={detailId} className={styles.size}>
        {ready ? status.info.cached ? 'Loaded from this browser’s saved model.' : 'The engine is ready to play.'
          : modelBytes ? `${formatModelBytes(modelBytes)} for the model. Saved copies are reused when available.` : 'A model download may be needed. Saved copies are reused when available.'}
      </p>
      <div role="status" aria-live="polite" className={styles.stage}>{stage}</div>
      {preparing && (
        <div className={styles.progressArea}>
          <progress aria-label="Browser AI preparation" aria-describedby={detailId} max={100} value={progress ?? undefined} />
          {status.phase === 'downloading' && <div className={styles.progressText}>
            <span>{formatModelBytes(status.loadedBytes)}{status.totalBytes !== null ? ` of ${formatModelBytes(status.totalBytes)}` : ' received'}</span>
            {progress !== null && <strong>{Math.floor(progress)}%</strong>}
          </div>}
          <p className={styles.hint}>{status.phase === 'initializing' ? 'The first preparation can take a moment. No other software is needed.' : status.phase === 'verifying' ? 'Verifying that the complete model arrived correctly.' : 'You can cancel without changing the game.'}</p>
        </div>
      )}
      {status.phase === 'error' && <p role="alert" className={styles.error}>{status.message}</p>}
      {capability?.status === 'unavailable' && !preparing && !ready && <p className={styles.hint}>{capability.reason}</p>}
      {notice && <p className={styles.hint}>{notice}</p>}
      <div className={styles.actions}>
        {preparing ? <button type="button" onClick={onCancel} aria-label="Cancel browser AI preparation">Cancel</button>
          : ready ? onSelect ? <button type="button" onClick={onSelect} aria-pressed={selected}>{selected ? 'Browser AI selected' : 'Play against browser AI'}</button> : null
            : available ? <button type="button" onClick={onPrepare}>
              {status.phase === 'error' || notice ? 'Retry browser AI' : inGame ? 'Prepare browser AI' : 'Download browser AI'}
            </button> : onCheck ? <button type="button" onClick={onCheck} disabled={capability?.status === 'checking'}>Check browser AI availability</button> : null}
      </div>
      {inGame && preparing && <p className={styles.hint}>Canceling preparation pauses AI play.</p>}
      {modelVersion && <details className={styles.details}><summary>Browser AI details</summary><dl><div><dt>Published model</dt><dd>{modelVersion}</dd></div>{modelBytes && <div><dt>Model size</dt><dd>{formatModelBytes(modelBytes)}</dd></div>}</dl></details>}
    </section>
  );
}
