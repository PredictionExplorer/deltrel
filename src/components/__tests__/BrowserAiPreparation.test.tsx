import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { AiCapability } from '@/lib/deltrel/ai/capabilities';
import { BrowserAiPreparation } from '../BrowserAiPreparation';

const capability: AiCapability = {
  status: 'available', label: 'Browser AI',
  browserModel: { modelVersion: 'champion-v3', bytes: 37_577_312, sha256: 'a'.repeat(64) },
};
const ready = { modelVersion: 'champion-v3', bytes: 37_577_312, backend: 'wasm' as const, cached: true };
afterEach(cleanup);

describe('BrowserAiPreparation', () => {
  it('reports failed availability even when an earlier model was prepared', async () => {
    const user = userEvent.setup();
    const check = vi.fn();
    render(<BrowserAiPreparation status={{ phase: 'ready', info: ready }} authorized capability={{ status: 'unavailable', label: 'Browser AI', code: 'offline', reason: 'The model is temporarily unavailable.', retryable: true }} onPrepare={vi.fn()} onCancel={vi.fn()} onCheck={check} />);
    expect(screen.getByRole('status')).toHaveTextContent('Not available');
    expect(screen.getByText('The model is temporarily unavailable.')).toBeVisible();
    expect(screen.queryByText('Ready on this device')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Check browser AI availability' }));
    expect(check).toHaveBeenCalledOnce();
  });

  it('explains the download size and only prepares after an explicit click', async () => {
    const user = userEvent.setup();
    const prepare = vi.fn();
    const { container } = render(<BrowserAiPreparation status={{ phase: 'idle' }} authorized={false} capability={capability} onPrepare={prepare} onCancel={vi.fn()} />);
    expect(prepare).not.toHaveBeenCalled();
    expect(screen.getByText(/37.6 MB for the model/)).toBeVisible();
    expect(screen.getByText('No installation')).toBeVisible();
    await user.click(screen.getByRole('button', { name: 'Download browser AI' }));
    expect(prepare).toHaveBeenCalledOnce();
    expect((await axe(container)).violations).toEqual([]);
  });

  it('shows actual bytes and percentage during download and offers cancellation', async () => {
    const user = userEvent.setup();
    const cancel = vi.fn();
    render(<BrowserAiPreparation status={{ phase: 'downloading', loadedBytes: 1_000_000, totalBytes: 4_000_000, modelVersion: 'champion-v3', cached: false }} authorized={false} capability={capability} inGame onPrepare={vi.fn()} onCancel={cancel} />);
    expect(screen.getByRole('progressbar', { name: 'Browser AI preparation' })).toHaveAttribute('value', '25');
    expect(screen.getByText('1.0 MB of 4.0 MB')).toBeVisible();
    expect(screen.getByText('25%')).toBeVisible();
    expect(screen.getByText('Canceling preparation pauses AI play.')).toBeVisible();
    await user.click(screen.getByRole('button', { name: 'Cancel browser AI preparation' }));
    expect(cancel).toHaveBeenCalledOnce();
  });

  it.each([
    ['checking', 'Checking the latest model…'],
    ['verifying', 'Checking the download…'],
    ['initializing', 'Preparing the engine…'],
  ] as const)('represents %s as an honest indeterminate stage', (phase, label) => {
    render(<BrowserAiPreparation status={{ phase, loadedBytes: 4_000_000, totalBytes: 4_000_000, modelVersion: 'champion-v3', cached: true }} authorized={false} capability={capability} onPrepare={vi.fn()} onCancel={vi.fn()} />);
    expect(screen.getByRole('status')).toHaveTextContent(label);
    expect(screen.getByRole('progressbar')).not.toHaveAttribute('value');
    expect(screen.queryByText('100%')).not.toBeInTheDocument();
  });

  it('does not invent a percentage when the transfer length is unknown', () => {
    render(<BrowserAiPreparation status={{ phase: 'downloading', loadedBytes: 1_024, totalBytes: null, modelVersion: null, cached: false }} authorized={false} onPrepare={vi.fn()} onCancel={vi.fn()} />);
    expect(screen.getByText('1 KB received')).toBeVisible();
    expect(screen.getByRole('progressbar')).not.toHaveAttribute('value');
    expect(screen.queryByText('0%')).not.toBeInTheDocument();
  });

  it('identifies cache reuse and lets a prepared model be selected without another download', async () => {
    const user = userEvent.setup();
    const select = vi.fn();
    const prepare = vi.fn();
    const { rerender } = render(<BrowserAiPreparation status={{ phase: 'ready', info: ready }} authorized capability={capability} onPrepare={prepare} onCancel={vi.fn()} onSelect={select} />);
    expect(screen.getByText('Loaded from this browser’s saved model.')).toBeVisible();
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Play against browser AI' }));
    expect(select).toHaveBeenCalledOnce();
    expect(prepare).not.toHaveBeenCalled();
    rerender(<BrowserAiPreparation status={{ phase: 'ready', info: ready }} authorized selected capability={capability} onPrepare={prepare} onCancel={vi.fn()} onSelect={select} />);
    expect(screen.getByRole('button', { name: 'Browser AI selected' })).toHaveAttribute('aria-pressed', 'true');
  });

  it('offers retry after a failed preparation and a lightweight availability recheck when unavailable', async () => {
    const user = userEvent.setup();
    const prepare = vi.fn();
    const check = vi.fn();
    const { rerender } = render(<BrowserAiPreparation status={{ phase: 'error', message: 'The download was interrupted.', retryable: true }} authorized={false} capability={capability} onPrepare={prepare} onCancel={vi.fn()} />);
    expect(screen.getByRole('alert')).toHaveTextContent('The download was interrupted.');
    await user.click(screen.getByRole('button', { name: 'Retry browser AI' }));
    expect(prepare).toHaveBeenCalledOnce();
    rerender(<BrowserAiPreparation status={{ phase: 'idle' }} authorized={false} capability={{ status: 'unavailable', label: 'Browser AI', reason: 'The model is temporarily unavailable.', code: 'offline', retryable: true }} onPrepare={prepare} onCancel={vi.fn()} onCheck={check} />);
    await user.click(screen.getByRole('button', { name: 'Check browser AI availability' }));
    expect(check).toHaveBeenCalledOnce();
    expect(prepare).toHaveBeenCalledOnce();
  });
});
