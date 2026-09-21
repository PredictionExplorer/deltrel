import type { DeltrelAiSearchBudget } from './decision';

export interface BrowserStrengthOption {
  id: 'quick' | 'balanced' | 'deep';
  label: string;
  description: string;
  budget: DeltrelAiSearchBudget;
}

const LEVELS: readonly BrowserStrengthOption[] = [
  { id: 'quick', label: 'Quick', description: 'Faster replies', budget: { simulations: 8, maxConsidered: 4 } },
  { id: 'balanced', label: 'Balanced', description: 'More thinking', budget: { simulations: 32, maxConsidered: 8 } },
  { id: 'deep', label: 'Deep', description: 'Deepest search', budget: { simulations: 64, maxConsidered: 8 } },
];

function validMaximum(maximum: DeltrelAiSearchBudget): void {
  if (!Number.isSafeInteger(maximum.simulations) || maximum.simulations < 1 ||
      !Number.isSafeInteger(maximum.maxConsidered) || maximum.maxConsidered < 1) {
    throw new Error('Browser strength limits must be positive whole numbers.');
  }
}

/** Public levels are bounded by the currently published model, without duplicate choices. */
export function browserStrengthOptions(maximum: DeltrelAiSearchBudget): BrowserStrengthOption[] {
  validMaximum(maximum);
  const seen = new Set<string>();
  const options: BrowserStrengthOption[] = [];
  for (const level of LEVELS) {
    const simulations = Math.min(level.budget.simulations, maximum.simulations);
    const maxConsidered = Math.min(level.budget.maxConsidered, maximum.maxConsidered, simulations);
    const key = `${simulations}:${maxConsidered}`;
    if (seen.has(key)) continue;
    seen.add(key);
    options.push({ ...level, budget: { simulations, maxConsidered } });
  }
  return options;
}

/** Only an exact match is a preset; custom and out-of-range saved choices remain custom. */
export function browserStrengthSelection(
  budget: DeltrelAiSearchBudget,
  maximum: DeltrelAiSearchBudget,
): BrowserStrengthOption | null {
  return browserStrengthOptions(maximum).find((option) =>
    option.budget.simulations === budget.simulations && option.budget.maxConsidered === budget.maxConsidered,
  ) ?? null;
}
