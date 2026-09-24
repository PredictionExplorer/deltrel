import { cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { AiCapability } from '@/lib/deltrel/ai/capabilities';
import { MAX_BROWSER_AI_SIMULATIONS, MAX_BROWSER_AI_MAX_CONSIDERED } from '@/lib/deltrel/ai/manifest';
import { BrowserAiStrengthControl } from '../BrowserAiStrengthControl';

const capability: AiCapability = {
  status: 'available', label: 'AI',
  search: { default: { simulations: 8, maxConsidered: 4 }, maximum: { simulations: MAX_BROWSER_AI_SIMULATIONS, maxConsidered: MAX_BROWSER_AI_MAX_CONSIDERED }, presets: {} },
};
afterEach(cleanup);

describe('BrowserAiStrengthControl', () => {
  it('exposes public, keyboard-operable presets and applies only an explicit changed selection', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const { container, rerender } = render(<BrowserAiStrengthControl capability={capability} budget={{ simulations: 128, maxConsidered: 8 }} onChange={onChange} />);
    const group = screen.getByRole('group', { name: 'AI playing strength' });
    expect(within(group).getAllByRole('button')).toHaveLength(3);
    expect(screen.getByRole('button', { name: 'Quick AI strength' })).toHaveAttribute('aria-pressed', 'true');
    await user.click(screen.getByRole('button', { name: 'Quick AI strength' }));
    expect(onChange).not.toHaveBeenCalled();
    const balanced = screen.getByRole('button', { name: 'Standard AI strength' });
    balanced.focus();
    await user.keyboard('{Enter}');
    expect(onChange).toHaveBeenCalledExactlyOnceWith({ simulations: 512, maxConsidered: 16 });
    rerender(<BrowserAiStrengthControl capability={capability} budget={{ simulations: 512, maxConsidered: 16 }} onChange={onChange} />);
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
    expect(within(screen.getByRole('group', { name: 'AI playing strength' })).getAllByRole('button').every((button) => button.getAttribute('aria-pressed') === 'false')).toBe(true);
    rerender(<BrowserAiStrengthControl capability={{ status: 'available', label: 'AI', search: { default: { simulations: 8, maxConsidered: 4 }, maximum: { simulations: 16, maxConsidered: 4 }, presets: {} } }} budget={budget} onChange={onChange} inGame />);
    expect(screen.getByText(/limits each search to 16 simulations and 4 candidate moves/)).toBeVisible();
    expect(onChange).not.toHaveBeenCalled();
  });

  it('deduplicates capped levels and offers a deliberate repair for an unsupported saved budget', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<BrowserAiStrengthControl capability={{ status: 'available', label: 'AI', search: { default: { simulations: 8, maxConsidered: 4 }, maximum: { simulations: 8, maxConsidered: 4 }, presets: {} } }} budget={{ simulations: 64, maxConsidered: 8 }} onChange={onChange} />);
    expect(within(screen.getByRole('group', { name: 'AI playing strength' })).getAllByRole('button')).toHaveLength(1);
    expect(screen.getByText(/saved setting exceeds/)).toBeVisible();
    await user.click(screen.getByRole('button', { name: 'Quick AI strength' }));
    expect(onChange).toHaveBeenCalledExactlyOnceWith({ simulations: 8, maxConsidered: 4 });
  });

  it('does not invent model limits before capability metadata is available', () => {
    const { rerender } = render(<BrowserAiStrengthControl capability={{ status: 'checking', label: 'AI' }} budget={{ simulations: 8, maxConsidered: 4 }} onChange={vi.fn()} />);
    expect(screen.queryByRole('group')).not.toBeInTheDocument();
    rerender(<BrowserAiStrengthControl capability={{ status: 'available', label: 'AI' }} budget={{ simulations: 8, maxConsidered: 4 }} onChange={vi.fn()} />);
    expect(screen.queryByRole('group')).not.toBeInTheDocument();
  });

  it('applies exact custom values beyond Deep only when requested and follows external setting changes', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const { container, rerender } = render(<BrowserAiStrengthControl capability={capability} budget={{ simulations: 64, maxConsidered: 8 }} onChange={onChange} inGame />);
    await user.click(screen.getByText('Custom search budget'));
    const simulations = screen.getByRole('spinbutton', { name: 'Simulations' });
    const candidates = screen.getByRole('spinbutton', { name: 'Candidate moves' });
    await user.clear(simulations);
    await user.type(simulations, '4096');
    await user.clear(candidates);
    await user.type(candidates, '256');
    expect(onChange).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: 'Apply custom budget' }));
    expect(onChange).toHaveBeenCalledExactlyOnceWith({ simulations: 4096, maxConsidered: 256 });
    rerender(<BrowserAiStrengthControl capability={capability} budget={{ simulations: 4096, maxConsidered: 256 }} onChange={onChange} inGame />);
    expect(screen.getByText('Custom')).toBeVisible();
    expect(screen.getByText(/4,096 simulations, up to 256 candidate moves/)).toBeVisible();
    expect(screen.getByRole('button', { name: 'Apply custom budget' })).toBeDisabled();
    expect((await axe(container)).violations).toEqual([]);
    rerender(<BrowserAiStrengthControl capability={capability} budget={{ simulations: 8, maxConsidered: 4 }} onChange={onChange} inGame />);
    expect(simulations).toHaveValue(8);
    expect(candidates).toHaveValue(4);
  });

  it.each(['', '0', '-1', '1.5', '4294967296'])('keeps invalid custom input %j out of the active budget', async (value) => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<BrowserAiStrengthControl capability={capability} budget={{ simulations: 4096, maxConsidered: 64 }} onChange={onChange} />);
    await user.click(screen.getByText('Custom search budget'));
    for (const name of ['Simulations', 'Candidate moves']) {
      const input = screen.getByRole('spinbutton', { name });
      await user.clear(input);
      if (value) await user.type(input, value);
      expect(input).toHaveAttribute('aria-invalid', 'true');
      expect(screen.getByRole('button', { name: 'Apply custom budget' })).toBeDisabled();
      expect(onChange).not.toHaveBeenCalled();
      await user.click(screen.getByRole('button', { name: 'Deep AI strength' }));
      expect(input).toHaveAttribute('aria-invalid', 'false');
    }
  });
});
