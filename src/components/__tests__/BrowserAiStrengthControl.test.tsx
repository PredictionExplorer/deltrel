import { cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { AiCapability } from '@/lib/deltrel/ai/capabilities';
import { BrowserAiStrengthControl } from '../BrowserAiStrengthControl';

const capability: AiCapability = {
  status: 'available', label: 'Browser AI',
  search: { default: { simulations: 8, maxConsidered: 4 }, maximum: { simulations: 64, maxConsidered: 8 }, presets: {} },
};
afterEach(cleanup);

describe('BrowserAiStrengthControl', () => {
  it('exposes public, keyboard-operable presets and applies only an explicit changed selection', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const { container, rerender } = render(<BrowserAiStrengthControl capability={capability} budget={{ simulations: 8, maxConsidered: 4 }} onChange={onChange} />);
    const group = screen.getByRole('group', { name: 'Browser AI playing strength' });
    expect(within(group).getAllByRole('button')).toHaveLength(3);
    expect(screen.getByRole('button', { name: 'Quick browser AI strength' })).toHaveAttribute('aria-pressed', 'true');
    await user.click(screen.getByRole('button', { name: 'Quick browser AI strength' }));
    expect(onChange).not.toHaveBeenCalled();
    const balanced = screen.getByRole('button', { name: 'Balanced browser AI strength' });
    balanced.focus();
    await user.keyboard('{Enter}');
    expect(onChange).toHaveBeenCalledExactlyOnceWith({ simulations: 32, maxConsidered: 8 });
    rerender(<BrowserAiStrengthControl capability={capability} budget={{ simulations: 32, maxConsidered: 8 }} onChange={onChange} />);
    expect(balanced).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText('Your choice is saved for future games.')).toBeVisible();
    expect((await axe(container)).violations).toEqual([]);
  });

  it('shows a custom saved budget without silently changing it during capability updates', () => {
    const onChange = vi.fn();
    const budget = { simulations: 24, maxConsidered: 6 };
    const { rerender } = render(<BrowserAiStrengthControl capability={capability} budget={budget} onChange={onChange} inGame />);
    expect(screen.getByText('Custom')).toBeVisible();
    expect(screen.getByText(/24 simulations, up to 6 candidate moves/)).toBeVisible();
    expect(screen.getByText(/Applies to the next search/)).toBeVisible();
    expect(screen.getAllByRole('button').every((button) => button.getAttribute('aria-pressed') === 'false')).toBe(true);
    rerender(<BrowserAiStrengthControl capability={{ status: 'available', label: 'Browser AI', search: { default: { simulations: 8, maxConsidered: 4 }, maximum: { simulations: 16, maxConsidered: 4 }, presets: {} } }} budget={budget} onChange={onChange} inGame />);
    expect(screen.getByText(/limits each search to 16 simulations and 4 candidate moves/)).toBeVisible();
    expect(onChange).not.toHaveBeenCalled();
  });

  it('deduplicates capped levels and offers a deliberate repair for an unsupported saved budget', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<BrowserAiStrengthControl capability={{ status: 'available', label: 'Browser AI', search: { default: { simulations: 8, maxConsidered: 4 }, maximum: { simulations: 8, maxConsidered: 4 }, presets: {} } }} budget={{ simulations: 64, maxConsidered: 8 }} onChange={onChange} />);
    expect(screen.getAllByRole('button')).toHaveLength(1);
    expect(screen.getByText(/saved setting exceeds/)).toBeVisible();
    await user.click(screen.getByRole('button', { name: 'Quick browser AI strength' }));
    expect(onChange).toHaveBeenCalledExactlyOnceWith({ simulations: 8, maxConsidered: 4 });
  });

  it('does not invent model limits before capability metadata is available', () => {
    const { rerender } = render(<BrowserAiStrengthControl capability={{ status: 'checking', label: 'Browser AI' }} budget={{ simulations: 8, maxConsidered: 4 }} onChange={vi.fn()} />);
    expect(screen.queryByRole('group')).not.toBeInTheDocument();
    rerender(<BrowserAiStrengthControl capability={{ status: 'available', label: 'Browser AI' }} budget={{ simulations: 8, maxConsidered: 4 }} onChange={vi.fn()} />);
    expect(screen.queryByRole('group')).not.toBeInTheDocument();
  });
});
