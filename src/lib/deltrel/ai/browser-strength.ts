import type { DeltrelAiSearchBudget } from './decision';

export interface BrowserStrengthOption {
  id: 'quick' | 'balanced' | 'deep';
  label: string;
  description: string;
  budget: DeltrelAiSearchBudget;
}

const LEVELS: readonly BrowserStrengthOption[] = [
  { id: 'quick', label: 'Quick', description: 'Faster replies', budget: { simulations: 128, maxConsidered: 8 } },
  { id: 'balanced', label: 'Standard', description: 'Champion search settings', budget: { simulations: 512, maxConsidered: 16 } },
  { id: 'deep', label: 'Deep', description: 'Deeper search', budget: { simulations: 4_096, maxConsidered: 64 } },
];

function validMaximum(maximum: DeltrelAiSearchBudget): void {
  if (!Number.isSafeInteger(maximum.simulations) || maximum.simulations < 1 ||
      !Number.isSafeInteger(maximum.maxConsidered) || maximum.maxConsidered < 1) {
    throw new Error('Browser strength limits must be positive whole numbers.');
  }
}

/** Familiar presets stay stable as custom runtime limits increase, without duplicate choices. */
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

/** Only an exact match is a preset; custom and out-of-range saved choices remain custom. */
export function browserStrengthSelection(
  budget: DeltrelAiSearchBudget,
  maximum: DeltrelAiSearchBudget,
): BrowserStrengthOption | null {
  return browserStrengthOptions(maximum).find((option) =>
    option.budget.simulations === budget.simulations && option.budget.maxConsidered === budget.maxConsidered,
  ) ?? null;
}
