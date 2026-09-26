'use client';

import { useId, useState } from 'react';
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
  runtime?: 'local' | 'server';
}

export function BrowserAiStrengthControl({ budget, capability, onChange, inGame = false, runtime = 'local' }: BrowserAiStrengthControlProps) {
  const headingId = useId();
  const hintId = useId();
  const search = capability.status === 'available' ? capability.search : undefined;
  if (!search) return null;
  const options = browserStrengthOptions(search.maximum);
  const supportedBudget = runtime === 'server' ? budget : normalizeBrowserStrengthBudget(budget, search.maximum);
  const selected = browserStrengthSelection(supportedBudget, search.maximum);
  const label = runtime === 'server' ? 'Cloud AI' : 'AI';

  return (
    <section className={styles.panel} aria-labelledby={headingId}>
      <header className={styles.header}>
        <h2 id={headingId}>{label} strength</h2>
        <span className={styles.selected} aria-label={`Selected strength: ${selected?.label ?? 'Custom'}`}>{selected?.label ?? 'Custom'}</span>
      </header>
      <div role="group" aria-label={`${label} playing strength`} aria-describedby={hintId} className={styles.options} style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}>
        {options.map((option) => <button
          key={option.id}
          type="button"
          aria-label={`${option.label} ${label} strength`}
          aria-pressed={selected?.id === option.id}
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
        <p>{supportedBudget.simulations.toLocaleString('en-US')} simulations · up to {supportedBudget.maxConsidered.toLocaleString('en-US')} candidate moves</p>
      </details>
      {runtime === 'server' && <CloudCustomBudget key={`${budget.simulations}:${budget.maxConsidered}`} budget={budget} search={search} onChange={onChange} />}
    </section>
  );
}

function CloudCustomBudget({ budget, search, onChange }: {
  budget: DeltrelAiSearchBudget;
  search: AiSearchCapability;
  onChange: (budget: DeltrelAiSearchBudget) => void;
}) {
  const [simulations, setSimulations] = useState(String(budget.simulations));
  const [candidates, setCandidates] = useState(String(budget.maxConsidered));
  const value = { simulations: Number(simulations), maxConsidered: Number(candidates) };
  const valid = budgetFitsCapability(value, search);
  return <details className={styles.details}>
    <summary>Custom cloud search</summary>
    <form className={styles.custom} onSubmit={(event) => { event.preventDefault(); if (valid) onChange(value); }}>
      <label>Simulations<input type="number" min={1} max={search.maximum.simulations} step={1} value={simulations} onChange={(event) => setSimulations(event.target.value)} /></label>
      <label>Candidate moves<input type="number" min={1} max={search.maximum.maxConsidered} step={1} value={candidates} onChange={(event) => setCandidates(event.target.value)} /></label>
      <p>Up to {search.maximum.simulations.toLocaleString('en-US')} simulations and {search.maximum.maxConsidered.toLocaleString('en-US')} candidates.</p>
      <button type="submit" disabled={!valid}>Apply custom search</button>
    </form>
  </details>;
}
