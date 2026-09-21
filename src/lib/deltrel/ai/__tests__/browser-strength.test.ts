import { describe, expect, it } from 'vitest';
import { browserStrengthOptions, browserStrengthSelection } from '../browser-strength';

describe('browser playing strength', () => {
  it('offers three distinct public levels for the published champion', () => {
    const maximum = { simulations: 64, maxConsidered: 8 };
    expect(browserStrengthOptions(maximum).map(({ id, budget }) => ({ id, budget }))).toEqual([
      { id: 'quick', budget: { simulations: 8, maxConsidered: 4 } },
      { id: 'balanced', budget: { simulations: 32, maxConsidered: 8 } },
      { id: 'deep', budget: { simulations: 64, maxConsidered: 8 } },
    ]);
    expect(browserStrengthSelection({ simulations: 32, maxConsidered: 8 }, maximum)?.id).toBe('balanced');
  });

  it.each([
    [{ simulations: 10, maxConsidered: 6 }, [{ simulations: 8, maxConsidered: 4 }, { simulations: 10, maxConsidered: 6 }]],
    [{ simulations: 8, maxConsidered: 4 }, [{ simulations: 8, maxConsidered: 4 }]],
    [{ simulations: 3, maxConsidered: 10 }, [{ simulations: 3, maxConsidered: 3 }]],
  ])('bounds and deduplicates levels for limits %j', (maximum, expected) => {
    const options = browserStrengthOptions(maximum);
    expect(options.map((option) => option.budget)).toEqual(expected);
    expect(new Set(options.map((option) => JSON.stringify(option.budget))).size).toBe(options.length);
  });

  it('preserves exact custom settings and never classifies a clamped request as a selected preset', () => {
    const maximum = { simulations: 64, maxConsidered: 8 };
    const custom = Object.freeze({ simulations: 24, maxConsidered: 6 });
    expect(browserStrengthSelection(custom, maximum)).toBeNull();
    expect(browserStrengthSelection({ simulations: 123, maxConsidered: 9 }, maximum)).toBeNull();
    expect(custom).toEqual({ simulations: 24, maxConsidered: 6 });
  });

  it('returns independent budget objects and rejects invalid capability limits', () => {
    const maximum = Object.freeze({ simulations: 64, maxConsidered: 8 });
    const options = browserStrengthOptions(maximum);
    options[0].budget.simulations = 999;
    expect(browserStrengthOptions(maximum)[0].budget.simulations).toBe(8);
    for (const value of [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => browserStrengthOptions({ simulations: value, maxConsidered: 8 })).toThrow(/positive whole numbers/);
      expect(() => browserStrengthOptions({ simulations: 64, maxConsidered: value })).toThrow(/positive whole numbers/);
    }
  });
});
