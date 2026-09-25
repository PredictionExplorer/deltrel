'use client';

import { useId } from 'react';
import type { AiCapability, AiSearchCapability } from '@/lib/deltrel/ai/capabilities';
import type { DeltrelAiSearchBudget } from '@/lib/deltrel/ai/decision';
import { browserStrengthOptions, browserStrengthSelection, normalizeBrowserStrengthBudget } from '@/lib/deltrel/ai/browser-strength';
import styles from './BrowserAiStrengthControl.module.css';

export function budgetFitsCapability(
  budget: DeltrelAiSearchBudget,
  search: AiSearchCapability | undefined,
): boolean {
  return (
    search === undefined ||
    (Number.isSafeInteger(budget.simulations) &&
      budget.simulations > 0 &&
      budget.simulations <= search.maximum.simulations &&
      Number.isSafeInteger(budget.maxConsidered) &&
      budget.maxConsidered > 0 &&
      budget.maxConsidered <= search.maximum.maxConsidered)
  );
}

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
  const supportedBudget = normalizeBrowserStrengthBudget(budget, search.maximum);
  const selected = browserStrengthSelection(supportedBudget, search.maximum)!;

  return (
    <section className={styles.panel} aria-labelledby={headingId}>
      <header className={styles.header}>
        <h2 id={headingId}>AI strength</h2>
        <span className={styles.selected} aria-label={`Selected strength: ${selected.label}`}>{selected.label}</span>
      </header>
      <div role="group" aria-label="AI playing strength" aria-describedby={hintId} className={styles.options} style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}>
        {options.map((option) => <button
          key={option.id}
          type="button"
          aria-label={`${option.label} AI strength`}
          aria-pressed={selected.id === option.id}
          onClick={() => {
            if (budget.simulations !== option.budget.simulations || budget.maxConsidered !== option.budget.maxConsidered) {
              onChange({ ...option.budget });
            }
          }}
        >
          <strong>{option.label}</strong>
          <span>{option.description}</span>
        </button>)}
      </div>
      <p id={hintId} className={styles.hint}>Both levels use the same trained model. Deeper searches take longer.</p>
      <p className={styles.timing}>{inGame
        ? 'Applies to the next search. A search already running keeps its current setting.'
        : 'Your choice is saved for future games.'}</p>
      <details className={styles.details}>
        <summary>Search details</summary>
        <p>{selected.budget.simulations.toLocaleString('en-US')} simulations · up to {selected.budget.maxConsidered.toLocaleString('en-US')} candidate moves</p>
      </details>
    </section>
  );
}
