import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { GameRecord } from '@/lib/deltrel/game-record';
import { serializeGameRecord } from '@/lib/deltrel/game-record';
import { DEFAULT_AI_SEARCH_SETTINGS, useAppStore } from '@/lib/store';
import { GameLibraryButton, GameLibraryDialog, ShareGameRecord } from '../GameLibrary';

const library = vi.hoisted(() => ({ records: [] as GameRecord[], error: null as string | null, save: vi.fn(), remove: vi.fn() }));
vi.mock('@/lib/game-library', () => ({
  useGameLibrary: () => ({ records: library.records, error: library.error }),
  saveGameRecord: library.save,
  deleteGameRecord: library.remove,
  retryGameLibrarySaves: vi.fn(),
}));

const record: GameRecord = {
  id: 'test-game', createdAt: '2026-09-26T12:00:00.000Z', updatedAt: '2026-09-26T12:01:00.000Z',
  config: { rings: 4, mode: 'double', pieRule: true, handicap: 1, playerNames: ['Ada', 'Grace'] },
  controllers: ['human', 'server'], aiSearchSettings: DEFAULT_AI_SEARCH_SETTINGS,
  log: [{ type: 'place', node: 0 }, { type: 'swap' }, { type: 'place', node: 1 }, { type: 'place', node: 2 }], earlyOutcome: null,
};

beforeEach(() => {
  library.records = [record];
  library.error = null;
  library.save.mockReturnValue(true);
  library.remove.mockReturnValue(true);
  useAppStore.setState({ phase: 'setup', gameId: null, log: [], earlyOutcome: null });
});
afterEach(() => { cleanup(); vi.useRealTimers(); });

describe('Game library', () => {
  it('opens from its button, searches games, and closes with restored focus', async () => {
    const user = userEvent.setup();
    const onOpen = vi.fn();
    render(<GameLibraryButton onOpen={onOpen} />);
    await user.click(screen.getByRole('button', { name: 'Game library' }));
    expect(onOpen).toHaveBeenCalledOnce();
    expect(screen.getByRole('dialog', { name: 'Game library' })).toBeVisible();
    const search = screen.getByRole('searchbox', { name: 'Search saved games' });
    await user.type(search, 'nobody');
    expect(screen.getByText('No games match your search.')).toBeVisible();
    await user.clear(search);
    await user.type(search, 'cloud');
    expect(screen.getByRole('button', { name: 'Review Ada vs Grace' })).toBeVisible();
    await user.click(screen.getByRole('button', { name: 'Close game library' }));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Game library' })).toHaveFocus();
  });

  it('reviews moves without changing the live match, including slider and keyboard navigation', async () => {
    const user = userEvent.setup();
    const before = useAppStore.getState();
    render(<GameLibraryDialog onClose={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: 'Review Ada vs Grace' }));
    expect(screen.getByText('Saved position')).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Play from here' })).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Step one move back' }));
    expect(screen.getByText('Viewing move 3 of 4')).toBeVisible();
    await user.keyboard('{Home}');
    expect(screen.getByText('Viewing the start')).toBeVisible();
    await user.keyboard('{ArrowRight}');
    expect(screen.getByText('Viewing move 1 of 4')).toBeVisible();
    await user.keyboard('{End}');
    expect(screen.getByText('Saved position')).toBeVisible();
    fireEvent.change(screen.getByRole('slider', { name: 'Review move' }), { target: { value: '2' } });
    expect(screen.getByText('Viewing move 2 of 4')).toBeVisible();
    await user.click(screen.getByRole('checkbox', { name: 'Show influence' }));
    expect(useAppStore.getState().log).toEqual(before.log);
    expect(useAppStore.getState().config).toEqual(before.config);
    await user.click(screen.getByRole('button', { name: 'Share this game' }));
    expect(screen.getByRole('textbox', { name: 'Game notation' })).toHaveValue(serializeGameRecord(record));
    await user.click(screen.getByRole('button', { name: '← All games' }));
    expect(screen.getByRole('searchbox')).toBeVisible();
  });

  it('imports validated text and reports invalid notation without saving it', async () => {
    const user = userEvent.setup();
    render(<GameLibraryDialog initialTab="import" onClose={vi.fn()} />);
    const input = screen.getByRole('textbox', { name: 'Game notation' });
    expect(screen.getByRole('button', { name: 'Import and review' })).toBeDisabled();
    fireEvent.change(input, { target: { value: 'not a game' } });
    await user.click(screen.getByRole('button', { name: 'Import and review' }));
    expect(screen.getByRole('alert')).toBeVisible();
    expect(library.save).not.toHaveBeenCalled();
    fireEvent.change(input, { target: { value: serializeGameRecord(record) } });
    await user.click(screen.getByRole('button', { name: 'Import and review' }));
    expect(library.save).toHaveBeenCalledWith(expect.objectContaining({ config: record.config, log: record.log }));
    expect(screen.getByText('Saved position')).toBeVisible();
  });

  it('loads files with a size limit and read errors', async () => {
    const user = userEvent.setup();
    render(<GameLibraryDialog initialTab="import" onClose={vi.fn()} />);
    const fileInput = screen.getByLabelText('Open game file');
    fireEvent.change(fileInput, { target: { files: [{ size: 300000 }] } });
    expect(screen.getByRole('alert')).toHaveTextContent('256 KB');
    fireEvent.change(fileInput, { target: { files: [{ size: 10, text: () => Promise.reject(new Error('read')) }] } });
    expect(await screen.findByText('Could not read this file.')).toBeVisible();
    fireEvent.change(fileInput, { target: { files: [{ size: 1000, text: () => Promise.resolve(serializeGameRecord(record)) }] } });
    await user.click(screen.getByRole('button', { name: 'Import and review' }));
    expect(screen.getByText('Saved position')).toBeVisible();
  });

  it('requires confirmation before deleting and keeps storage errors visible', async () => {
    const user = userEvent.setup();
    library.error = 'Local game storage is unavailable.';
    render(<GameLibraryDialog onClose={vi.fn()} />);
    expect(screen.getByRole('alert')).toHaveTextContent('unavailable');
    await user.click(screen.getByRole('button', { name: 'Delete saved game' }));
    expect(library.remove).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: 'Keep game' }));
    expect(screen.queryByText('Remove this saved game?')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Delete saved game' }));
    await user.click(screen.getByRole('button', { name: 'Delete game' }));
    expect(library.remove).toHaveBeenCalledWith(record.id);
  });

  it('shares the current game and preserves its explicit resignation result', async () => {
    const user = userEvent.setup();
    const ended: GameRecord = { ...record, earlyOutcome: { reason: 'resignation', winner: 1, loser: 0 } };
    useAppStore.setState({ ...ended, phase: 'playing', gameId: record.id, gameCreatedAt: record.createdAt, gameUpdatedAt: record.updatedAt });
    render(<GameLibraryButton share />);
    await user.click(screen.getByRole('button', { name: 'Share game' }));
    expect(screen.getByRole('textbox', { name: 'Game notation' })).toHaveValue(serializeGameRecord(ended));
    await user.click(screen.getByRole('button', { name: /Saved games/ }));
    expect(screen.queryByRole('button', { name: 'Delete saved game' })).not.toBeInTheDocument();
  });

  it('copies, falls back to text selection, and downloads a portable file', async () => {
    const user = userEvent.setup();
    render(<ShareGameRecord record={record} />);
    await user.click(screen.getByRole('button', { name: 'Copy game' }));
    expect(await navigator.clipboard.readText()).toBe(serializeGameRecord(record));
    expect(screen.getByRole('status')).toHaveTextContent('Copied');
    vi.spyOn(navigator.clipboard, 'writeText').mockRejectedValue(new Error('denied'));
    await user.click(screen.getByRole('button', { name: 'Copy game' }));
    expect(screen.getByRole('status')).toHaveTextContent('Text selected');
    const area = screen.getByRole('textbox') as HTMLTextAreaElement;
    expect(area.selectionEnd - area.selectionStart).toBe(area.value.length);
    const create = vi.fn(() => 'blob:test');
    const revoke = vi.fn();
    vi.stubGlobal('URL', { createObjectURL: create, revokeObjectURL: revoke });
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    await user.click(screen.getByRole('button', { name: 'Download .dgn' }));
    expect(create).toHaveBeenCalledOnce();
    expect(click).toHaveBeenCalledOnce();
    expect(revoke).toHaveBeenCalledWith('blob:test');
  });

  it('shows clear empty states and passes library and review accessibility checks', async () => {
    const user = userEvent.setup();
    const { container } = render(<GameLibraryDialog onClose={vi.fn()} />);
    expect((await axe(container)).violations).toEqual([]);
    await user.click(screen.getByRole('button', { name: 'Review Ada vs Grace' }));
    expect((await axe(container)).violations).toEqual([]);
    expect(within(screen.getByRole('region', { name: 'Move history' })).getByText('Swap')).toBeVisible();
    cleanup();
    library.records = [];
    render(<GameLibraryDialog onClose={vi.fn()} />);
    expect(screen.getByText(/Your games will appear here/)).toBeVisible();
  });
});
