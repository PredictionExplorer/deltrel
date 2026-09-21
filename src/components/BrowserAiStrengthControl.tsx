'use client';

import { useId } from 'react';
import type { AiCapability } from '@/lib/deltrel/ai/capabilities';
import type { DeltrelAiSearchBudget } from '@/lib/deltrel/ai/decision';
import { browserStrengthOptions, browserStrengthSelection } from '@/lib/deltrel/ai/browser-strength';
import styles from './BrowserAiStrengthControl.module.css';

export interface BrowserAiStrengthControlProps {
  budget: DeltrelAiSearchBudget;
  capability: AiCapability;
  onChange: (budget: DeltrelAiSearchBudget) => void;
  inGame?: boolean;
}

export function BrowserAiStrengthControl({ budget, capability, onChange, inGame = false }: BrowserAiStrengthControlProps) {
  const headingId = useId();
  const hintId = useId();
  const search = capability.status === 'available' ? capability.search : undefined;
  if (!search) return null;
  const options = browserStrengthOptions(search.maximum);
  const selected = browserStrengthSelection(budget, search.maximum);
  const exceedsLimits = budget.simulations > search.maximum.simulations || budget.maxConsidered > search.maximum.maxConsidered;

  return (
    <section className={styles.panel} aria-labelledby={headingId}>
      <header className={styles.header}>
        <h2 id={headingId}>Browser AI strength</h2>
        <span className={styles.selected} aria-label={`Selected strength: ${selected?.label ?? 'Custom'}`}>{selected?.label ?? 'Custom'}</span>
      </header>
      <div role="group" aria-label="Browser AI playing strength" aria-describedby={hintId} className={styles.options} style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}>
        {options.map((option) => <button
          key={option.id}
          type="button"
          aria-label={`${option.label} browser AI strength`}
          aria-pressed={selected?.id === option.id}
          onClick={() => { if (selected?.id !== option.id) onChange({ ...option.budget }); }}
        >
          <strong>{option.label}</strong>
          <span>{option.description}</span>
        </button>)}
      </div>
      <p id={hintId} className={styles.hint}>All levels use the same trained model. Deeper searches take longer.</p>
      {!selected && <p className={styles.custom}>Custom setting: {budget.simulations.toLocaleString('en-US')} simulations, up to {budget.maxConsidered.toLocaleString('en-US')} candidate moves.</p>}
      {exceedsLimits && <p className={styles.custom}>{inGame
        ? `This model limits each search to ${search.maximum.simulations.toLocaleString('en-US')} simulations and ${search.maximum.maxConsidered.toLocaleString('en-US')} candidate moves.`
        : 'This saved setting exceeds the current model’s limits. Choose a supported level above.'}</p>}
      <p className={styles.timing}>{inGame
        ? 'Applies to the next search. A search already running keeps its current setting.'
        : 'Your choice is saved for future games.'}</p>
      <details className={styles.details}>
        <summary>Search details</summary>
        <p>{selected ? `${selected.budget.simulations.toLocaleString('en-US')} simulations · up to ${selected.budget.maxConsidered.toLocaleString('en-US')} candidate moves` : 'Your custom search budget is retained until you choose another level.'}</p>
      </details>
    </section>
  );
}
