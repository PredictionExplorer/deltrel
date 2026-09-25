import type { DeltrelAiSearchBudget } from './decision';

export interface BrowserStrengthOption {
  id: 'balanced' | 'deep';
  label: string;
  description: string;
  budget: DeltrelAiSearchBudget;
}

const LEVELS: readonly BrowserStrengthOption[] = [
  { id: 'balanced', label: 'Standard', description: 'Strong everyday play', budget: { simulations: 544, maxConsidered: 16 } },
  { id: 'deep', label: 'Deep', description: 'Deeper search', budget: { simulations: 4_096, maxConsidered: 64 } },
];

function validMaximum(maximum: DeltrelAiSearchBudget): void {
  if (!Number.isSafeInteger(maximum.simulations) || maximum.simulations < 1 ||
      !Number.isSafeInteger(maximum.maxConsidered) || maximum.maxConsidered < 1) {
    throw new Error('Browser strength limits must be positive whole numbers.');
  }
}

/** Offer only Standard and Deep, bounded by the current engine's limits. */
export function browserStrengthOptions(maximum: DeltrelAiSearchBudget): BrowserStrengthOption[] {
  validMaximum(maximum);
  const seen = new Set<string>();
  const options: BrowserStrengthOption[] = [];
  for (const level of LEVELS) {
    const budget = level.budget;
    const simulations = Math.min(budget.simulations, maximum.simulations);
    const maxConsidered = Math.min(budget.maxConsidered, maximum.maxConsidered, simulations);
    const key = `${simulations}:${maxConsidered}`;
    if (seen.has(key)) continue;
    seen.add(key);
    options.push({ ...level, budget: { simulations, maxConsidered } });
  }
  return options;
}

/** An exact match identifies a supported preset. */
export function browserStrengthSelection(
  budget: DeltrelAiSearchBudget,
  maximum: DeltrelAiSearchBudget,
): BrowserStrengthOption | null {
  return browserStrengthOptions(maximum).find((option) =>
    option.budget.simulations === budget.simulations && option.budget.maxConsidered === budget.maxConsidered,
  ) ?? null;
}

/** Upgrade older Quick/Standard settings and replace custom budgets with a supported preset. */
export function normalizeBrowserStrengthBudget(
  budget: DeltrelAiSearchBudget,
  maximum: DeltrelAiSearchBudget = LEVELS[1].budget,
): DeltrelAiSearchBudget {
  const options = browserStrengthOptions(maximum);
  const deep = options.find((option) => option.id === 'deep');
  const exact = options.find((option) => option.budget.simulations === budget.simulations && option.budget.maxConsidered === budget.maxConsidered);
  const selection = exact ?? (Number.isFinite(budget.simulations) && budget.simulations >= LEVELS[1].budget.simulations && deep
    ? deep
    : options[0]);
  return { ...selection.budget };
}
