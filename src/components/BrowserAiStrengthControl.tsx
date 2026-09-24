'use client';

import { useId, useState } from 'react';
import type { AiCapability, AiSearchCapability } from '@/lib/deltrel/ai/capabilities';
import type { DeltrelAiSearchBudget } from '@/lib/deltrel/ai/decision';
import { browserStrengthOptions, browserStrengthSelection } from '@/lib/deltrel/ai/browser-strength';
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

function parseCustomCount(value: string, maximum: number): number | null {
  if (!/^[0-9]+$/.test(value)) return null;
  const count = Number(value);
  return Number.isSafeInteger(count) && count > 0 && count <= maximum ? count : null;
}

export function BrowserAiStrengthControl({ budget, capability, onChange, inGame = false }: BrowserAiStrengthControlProps) {
  const headingId = useId();
  const hintId = useId();
  const customId = useId();
  const [draft, setDraft] = useState({ simulations: String(budget.simulations), maxConsidered: String(budget.maxConsidered) });
  const [observedBudget, setObservedBudget] = useState(budget);
  if (observedBudget.simulations !== budget.simulations || observedBudget.maxConsidered !== budget.maxConsidered) {
    setObservedBudget(budget);
    setDraft({ simulations: String(budget.simulations), maxConsidered: String(budget.maxConsidered) });
  }
  const search = capability.status === 'available' ? capability.search : undefined;
  if (!search) return null;
  const options = browserStrengthOptions(search.maximum);
  const selected = browserStrengthSelection(budget, search.maximum);
  const exceedsLimits = budget.simulations > search.maximum.simulations || budget.maxConsidered > search.maximum.maxConsidered;
  const simulations = parseCustomCount(draft.simulations, search.maximum.simulations);
  const maxConsidered = parseCustomCount(draft.maxConsidered, search.maximum.maxConsidered);
  const changed = simulations !== budget.simulations || maxConsidered !== budget.maxConsidered;

  return (
    <section className={styles.panel} aria-labelledby={headingId}>
      <header className={styles.header}>
        <h2 id={headingId}>AI strength</h2>
        <span className={styles.selected} aria-label={`Selected strength: ${selected?.label ?? 'Custom'}`}>{selected?.label ?? 'Custom'}</span>
      </header>
      <div role="group" aria-label="AI playing strength" aria-describedby={hintId} className={styles.options} style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}>
        {options.map((option) => <button
          key={option.id}
          type="button"
          aria-label={`${option.label} AI strength`}
          aria-pressed={selected?.id === option.id}
          onClick={() => {
            setDraft({ simulations: String(option.budget.simulations), maxConsidered: String(option.budget.maxConsidered) });
            if (selected?.id !== option.id) onChange({ ...option.budget });
          }}
        >
          <strong>{option.label}</strong>
          <span>{option.description}</span>
        </button>)}
      </div>
      <p id={hintId} className={styles.hint}>All levels use the same trained model. Deeper searches take longer.</p>
      {!selected && <p className={styles.custom}>Custom setting: {budget.simulations.toLocaleString('en-US')} simulations, up to {budget.maxConsidered.toLocaleString('en-US')} candidate moves.</p>}
      {exceedsLimits && <p className={styles.custom}>{inGame
        ? `This engine limits each search to ${search.maximum.simulations.toLocaleString('en-US')} simulations and ${search.maximum.maxConsidered.toLocaleString('en-US')} candidate moves.`
        : 'This saved setting exceeds the engine’s limits. Choose a supported level or custom budget.'}</p>}
      <details className={styles.customBudget}>
        <summary>Custom search budget</summary>
        <p className={styles.hint}>Go beyond Deep with your own values. Larger searches take longer; use Pause AI to stop a search.</p>
        <div className={styles.fields}>
          {([
            ['simulations', 'Simulations', 'How much search to perform per move.', simulations],
            ['maxConsidered', 'Candidate moves', 'How many possible moves to compare, up to the available moves.', maxConsidered],
          ] as const).map(([field, label, description, value]) => <div key={field}>
            <label htmlFor={`${customId}-${field}`}>{label}</label>
            <input
              id={`${customId}-${field}`}
              type="number"
              inputMode="numeric"
              min={1}
              max={search.maximum[field]}
              step={1}
              value={draft[field]}
              aria-invalid={value === null}
              aria-describedby={`${customId}-${field}-hint${value === null ? ` ${customId}-${field}-error` : ''}`}
              onChange={(event) => setDraft({ ...draft, [field]: event.target.value })}
            />
            <p id={`${customId}-${field}-hint`}>{description}</p>
            {value === null && <p id={`${customId}-${field}-error`} role="alert" className={styles.error}>
              Enter a whole number from 1 to {search.maximum[field].toLocaleString('en-US')}.
            </p>}
          </div>)}
        </div>
        <button
          type="button"
          className={styles.apply}
          disabled={simulations === null || maxConsidered === null || !changed}
          onClick={() => {
            if (simulations !== null && maxConsidered !== null) onChange({ simulations, maxConsidered });
          }}
        >Apply custom budget</button>
      </details>
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
