import { describe, expect, it } from 'vitest';
import { browserStrengthOptions, browserStrengthSelection } from '../browser-strength';

const maximum = { simulations: 4_096, maxConsidered: 64 };

describe('browser playing strength', () => {
  it('offers champion-equivalent Standard and an explicitly deeper preset', () => {
    expect(browserStrengthOptions(maximum).map(({ id, label, budget }) => ({ id, label, budget }))).toEqual([
      { id: 'quick', label: 'Quick', budget: { simulations: 128, maxConsidered: 8 } },
      { id: 'balanced', label: 'Standard', budget: { simulations: 512, maxConsidered: 16 } },
      { id: 'deep', label: 'Deep', budget: { simulations: 4_096, maxConsidered: 64 } },
    ]);
    expect(browserStrengthSelection({ simulations: 512, maxConsidered: 16 }, maximum)?.id).toBe('balanced');
  });

  it.each([
    [{ simulations: 200, maxConsidered: 12 }, [{ simulations: 128, maxConsidered: 8 }, { simulations: 200, maxConsidered: 12 }]],
    [{ simulations: 8, maxConsidered: 4 }, [{ simulations: 8, maxConsidered: 4 }]],
    [{ simulations: 3, maxConsidered: 10 }, [{ simulations: 3, maxConsidered: 3 }]],
  ])('bounds and deduplicates levels for limits %j', (limits, expected) => {
    const options = browserStrengthOptions(limits);
    expect(options.map((option) => option.budget)).toEqual(expected);
    expect(new Set(options.map((option) => JSON.stringify(option.budget))).size).toBe(options.length);
  });

  it('preserves exact custom settings and never classifies a clamped request as a selected preset', () => {
    const custom = Object.freeze({ simulations: 24, maxConsidered: 6 });
    expect(browserStrengthSelection(custom, maximum)).toBeNull();
    expect(browserStrengthSelection({ simulations: 8_192, maxConsidered: 128 }, maximum)).toBeNull();
    expect(custom).toEqual({ simulations: 24, maxConsidered: 6 });
  });

  it('keeps Deep practical when custom limits permit much deeper search', () => {
    const limits = { simulations: 536_870_911, maxConsidered: 4_294_967_295 };
    expect(browserStrengthOptions(limits).at(-1)).toMatchObject({ id: 'deep', budget: maximum });
    expect(browserStrengthSelection(limits, limits)).toBeNull();
    expect(browserStrengthSelection(maximum, limits)?.id).toBe('deep');
  });

  it('returns independent budget objects and rejects invalid capability limits', () => {
    const options = browserStrengthOptions(Object.freeze(maximum));
    options[0].budget.simulations = 999;
    expect(browserStrengthOptions(maximum)[0].budget.simulations).toBe(128);
    for (const value of [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => browserStrengthOptions({ simulations: value, maxConsidered: 8 })).toThrow(/positive whole numbers/);
      expect(() => browserStrengthOptions({ simulations: 64, maxConsidered: value })).toThrow(/positive whole numbers/);
    }
  });
});
