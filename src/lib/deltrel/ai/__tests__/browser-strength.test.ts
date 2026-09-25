import { describe, expect, it } from 'vitest';
import { browserStrengthOptions, browserStrengthSelection, normalizeBrowserStrengthBudget } from '../browser-strength';

const maximum = { simulations: 4_096, maxConsidered: 64 };
const standard = { simulations: 544, maxConsidered: 16 };

describe('browser playing strength', () => {
  it('offers only stronger Standard and Deep presets', () => {
    expect(browserStrengthOptions(maximum).map(({ id, label, budget }) => ({ id, label, budget }))).toEqual([
      { id: 'balanced', label: 'Standard', budget: standard },
      { id: 'deep', label: 'Deep', budget: maximum },
    ]);
    expect(browserStrengthSelection(standard, maximum)?.id).toBe('balanced');
    expect(standard.simulations / 512).toBe(1.0625);
  });

  it.each([
    [{ simulations: 1_024, maxConsidered: 32 }, [standard, { simulations: 1_024, maxConsidered: 32 }]],
    [{ simulations: 200, maxConsidered: 12 }, [{ simulations: 200, maxConsidered: 12 }]],
    [{ simulations: 8, maxConsidered: 4 }, [{ simulations: 8, maxConsidered: 4 }]],
    [{ simulations: 3, maxConsidered: 10 }, [{ simulations: 3, maxConsidered: 3 }]],
  ])('bounds and deduplicates levels for limits %j', (limits, expected) => {
    const options = browserStrengthOptions(limits);
    expect(options.map((option) => option.budget)).toEqual(expected);
    expect(new Set(options.map((option) => JSON.stringify(option.budget))).size).toBe(options.length);
    for (const option of options) expect(normalizeBrowserStrengthBudget(option.budget, limits)).toEqual(option.budget);
  });

  it.each([
    { simulations: 8, maxConsidered: 4 },
    { simulations: 128, maxConsidered: 8 },
    { simulations: 512, maxConsidered: 16 },
    { simulations: 2_048, maxConsidered: 32 },
    { simulations: NaN, maxConsidered: 16 },
  ])('upgrades a previous low or custom budget %j to Standard', (budget) => {
    expect(normalizeBrowserStrengthBudget(budget)).toEqual(standard);
  });

  it.each([
    maximum,
    { simulations: 4_096, maxConsidered: 256 },
    { simulations: 8_192, maxConsidered: 16 },
  ])('retains deeper search intent for saved budget %j', (budget) => {
    expect(normalizeBrowserStrengthBudget(budget)).toEqual(maximum);
  });

  it('uses exact matches for preset selection without mutating saved choices', () => {
    const custom = Object.freeze({ simulations: 24, maxConsidered: 6 });
    expect(browserStrengthSelection(custom, maximum)).toBeNull();
    expect(browserStrengthSelection({ simulations: 8_192, maxConsidered: 128 }, maximum)).toBeNull();
    expect(normalizeBrowserStrengthBudget(custom)).toEqual(standard);
    expect(custom).toEqual({ simulations: 24, maxConsidered: 6 });
  });

  it('keeps Deep practical when engine limits permit much deeper search', () => {
    const limits = { simulations: 536_870_911, maxConsidered: 4_294_967_295 };
    expect(browserStrengthOptions(limits).at(-1)).toMatchObject({ id: 'deep', budget: maximum });
    expect(normalizeBrowserStrengthBudget(limits, limits)).toEqual(maximum);
    expect(browserStrengthSelection(maximum, limits)?.id).toBe('deep');
  });

  it('returns independent budget objects and rejects invalid capability limits', () => {
    const options = browserStrengthOptions(Object.freeze(maximum));
    options[0].budget.simulations = 999;
    expect(browserStrengthOptions(maximum)[0].budget.simulations).toBe(544);
    const normalized = normalizeBrowserStrengthBudget(standard);
    normalized.simulations = 1;
    expect(normalizeBrowserStrengthBudget(standard)).toEqual(standard);
    for (const value of [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => browserStrengthOptions({ simulations: value, maxConsidered: 8 })).toThrow(/positive whole numbers/);
      expect(() => browserStrengthOptions({ simulations: 64, maxConsidered: value })).toThrow(/positive whole numbers/);
    }
  });
});
