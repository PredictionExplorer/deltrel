import { cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { AiCapability } from '@/lib/deltrel/ai/capabilities';
import { MAX_BROWSER_AI_SIMULATIONS, MAX_BROWSER_AI_MAX_CONSIDERED } from '@/lib/deltrel/ai/manifest';
import { BrowserAiStrengthControl } from '../BrowserAiStrengthControl';

const capability: AiCapability = {
  status: 'available', label: 'AI',
  search: { default: { simulations: 544, maxConsidered: 16 }, maximum: { simulations: MAX_BROWSER_AI_SIMULATIONS, maxConsidered: MAX_BROWSER_AI_MAX_CONSIDERED }, presets: {} },
};
afterEach(cleanup);

describe('BrowserAiStrengthControl', () => {
  it('exposes only keyboard-operable Standard and Deep presets', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const { container, rerender } = render(<BrowserAiStrengthControl capability={capability} budget={{ simulations: 544, maxConsidered: 16 }} onChange={onChange} />);
    const group = screen.getByRole('group', { name: 'AI playing strength' });
    expect(within(group).getAllByRole('button')).toHaveLength(2);
    expect(screen.queryByRole('button', { name: 'Quick AI strength' })).not.toBeInTheDocument();
    expect(screen.queryByText('Custom search budget')).not.toBeInTheDocument();
    expect(screen.queryByRole('spinbutton')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Standard AI strength' })).toHaveAttribute('aria-pressed', 'true');
    await user.click(screen.getByRole('button', { name: 'Standard AI strength' }));
    expect(onChange).not.toHaveBeenCalled();
    const deep = screen.getByRole('button', { name: 'Deep AI strength' });
    deep.focus();
    await user.keyboard('{Enter}');
    expect(onChange).toHaveBeenCalledExactlyOnceWith({ simulations: 4096, maxConsidered: 64 });
    rerender(<BrowserAiStrengthControl capability={capability} budget={{ simulations: 4096, maxConsidered: 64 }} onChange={onChange} />);
    expect(deep).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText('Your choice is saved for future games.')).toBeVisible();
    expect((await axe(container)).violations).toEqual([]);
    await user.click(screen.getByRole('button', { name: 'Standard AI strength' }));
    expect(onChange).toHaveBeenLastCalledWith({ simulations: 544, maxConsidered: 16 });
  });

  it('keeps cloud custom budgets explicit and enforces published limits before applying', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const cloud: AiCapability = { status: 'available', label: 'Cloud AI', search: {
      default: { simulations: 544, maxConsidered: 16 }, maximum: { simulations: 16384, maxConsidered: 128 }, presets: {},
    } };
    const { container } = render(<BrowserAiStrengthControl runtime="server" capability={cloud} budget={{ simulations: 777, maxConsidered: 21 }} onChange={onChange} />);
    expect(screen.getByLabelText('Selected strength: Custom')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Standard Cloud AI strength' })).toHaveAttribute('aria-pressed', 'false');
    await user.click(screen.getByText('Custom cloud search'));
    const simulations = screen.getByRole('spinbutton', { name: 'Simulations' });
    const candidates = screen.getByRole('spinbutton', { name: 'Candidate moves' });
    expect(simulations).toHaveValue(777);
    expect(candidates).toHaveValue(21);
    await user.clear(simulations);
    await user.type(simulations, '16385');
    expect(screen.getByRole('button', { name: 'Apply custom search' })).toBeDisabled();
    expect(onChange).not.toHaveBeenCalled();
    await user.clear(simulations);
    await user.type(simulations, '1024');
    await user.clear(candidates);
    await user.type(candidates, '32');
    await user.click(screen.getByRole('button', { name: 'Apply custom search' }));
    expect(onChange).toHaveBeenCalledExactlyOnceWith({ simulations: 1024, maxConsidered: 32 });
    expect((await axe(container)).violations).toEqual([]);
  });

  it('presents old saved settings as supported presets without side effects during render', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<BrowserAiStrengthControl capability={capability} budget={{ simulations: 128, maxConsidered: 8 }} onChange={onChange} inGame />);
    expect(screen.getByRole('button', { name: 'Standard AI strength' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.queryByText('Custom')).not.toBeInTheDocument();
    expect(screen.getByText(/Applies to the next search/)).toBeVisible();
    expect(onChange).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: 'Standard AI strength' }));
    expect(onChange).toHaveBeenCalledExactlyOnceWith({ simulations: 544, maxConsidered: 16 });
  });

  it('deduplicates capped levels and applies a supported budget', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<BrowserAiStrengthControl capability={{ status: 'available', label: 'AI', search: { default: { simulations: 8, maxConsidered: 4 }, maximum: { simulations: 8, maxConsidered: 4 }, presets: {} } }} budget={{ simulations: 64, maxConsidered: 8 }} onChange={onChange} />);
    expect(within(screen.getByRole('group', { name: 'AI playing strength' })).getAllByRole('button')).toHaveLength(1);
    await user.click(screen.getByRole('button', { name: 'Standard AI strength' }));
    expect(onChange).toHaveBeenCalledExactlyOnceWith({ simulations: 8, maxConsidered: 4 });
  });

  it('does not invent model limits before capability metadata is available', () => {
    const { rerender } = render(<BrowserAiStrengthControl capability={{ status: 'checking', label: 'AI' }} budget={{ simulations: 544, maxConsidered: 16 }} onChange={vi.fn()} />);
    expect(screen.queryByRole('group')).not.toBeInTheDocument();
    rerender(<BrowserAiStrengthControl capability={{ status: 'available', label: 'AI' }} budget={{ simulations: 544, maxConsidered: 16 }} onChange={vi.fn()} />);
    expect(screen.queryByRole('group')).not.toBeInTheDocument();
  });

  it('follows external preset changes during play', () => {
    const onChange = vi.fn();
    const { rerender } = render(<BrowserAiStrengthControl capability={capability} budget={{ simulations: 4096, maxConsidered: 64 }} onChange={onChange} inGame />);
    expect(screen.getByRole('button', { name: 'Deep AI strength' })).toHaveAttribute('aria-pressed', 'true');
    rerender(<BrowserAiStrengthControl capability={capability} budget={{ simulations: 544, maxConsidered: 16 }} onChange={onChange} inGame />);
    expect(screen.getByRole('button', { name: 'Standard AI strength' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'Deep AI strength' })).toHaveAttribute('aria-pressed', 'false');
    expect(onChange).not.toHaveBeenCalled();
  });
});
