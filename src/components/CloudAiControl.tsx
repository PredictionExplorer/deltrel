'use client';

import { Cloud } from 'lucide-react';
import type { AiCapability } from '@/lib/deltrel/ai/capabilities';
import styles from './BrowserAiPreparation.module.css';

export function CloudAiControl({ capability, selected, onSelect, onCheck, onUseDevice }: {
  capability: AiCapability;
  selected: boolean;
  onSelect?: () => void;
  onCheck: () => void;
  onUseDevice?: () => void;
}) {
  return <section className={styles.panel} aria-label="Cloud AI">
    <header className={styles.header}>
      <h2><Cloud size={17} aria-hidden /> Cloud AI</h2>
      <span className={styles.badge}>No model download</span>
    </header>
    <p className={styles.description}>Let our server calculate each move. A good choice for slower devices. Requires an internet connection.</p>
    <p role="status" className={styles.stage}>{capability.status === 'checking'
      ? 'Checking cloud availability…'
      : capability.status === 'available' ? 'Cloud AI is ready'
        : 'Cloud AI is currently unavailable'}</p>
    {capability.status === 'unavailable' && <p className={styles.hint}>The server may be switched off. Your game stays saved; retry later or choose AI on this device.</p>}
    <div className={styles.actions}>
      {onSelect && <button type="button" onClick={onSelect} aria-pressed={selected} disabled={capability.status !== 'available'}>
        {selected ? 'Cloud AI selected' : 'Play against Cloud AI'}
      </button>}
      {capability.status === 'unavailable' && <button type="button" onClick={onCheck}>Check cloud availability</button>}
      {selected && capability.status === 'unavailable' && onUseDevice && <button type="button" onClick={onUseDevice}>Use AI on this device</button>}
    </div>
  </section>;
}
