'use client';

import { Trophy } from 'lucide-react';
import type { AiCapability } from '@/lib/deltrel/ai/capabilities';
import { useAppStore } from '@/lib/store';

const EFFORTS = [
  ['quick', 'Quick', 'Faster replies'],
  ['strong', 'Strong', 'More search per move'],
  ['maximum', 'Maximum', 'Longest thinking time'],
] as const;

export function ChampionPanel({
  capability,
  selected,
  onChoose,
}: {
  capability: AiCapability;
  selected: boolean;
  onChoose: () => void;
}) {
  const budget = useAppStore((state) => state.aiSearchSettings.server);
  const setBudget = useAppStore((state) => state.setAiSearchBudget);
  const champion = capability.status === 'available' ? capability.champion : undefined;
  const search = capability.status === 'available' ? capability.search : undefined;

  return (
    <section aria-labelledby="champion-heading" className="rounded-2xl border border-sand/30 bg-sand-faint px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="champion-heading" className="flex items-center gap-2 text-sm font-medium text-ink">
          <Trophy className="h-4 w-4 text-sand" aria-hidden /> Current champion
        </h2>
        <span className="text-xs text-sand-strong" role="status">
          {capability.status === 'checking'
            ? 'Checking connection…'
            : champion
              ? 'Ready to play'
              : 'Not connected'}
        </span>
      </div>
      {champion ? (
        <>
          <p className="mt-2 text-xs leading-relaxed text-muted">
            Full trained model · step {champion.modelStep.toLocaleString('en-US')}.
            {' '}Choose a variant and opening below.
          </p>
          <button
            type="button"
            onClick={onChoose}
            aria-pressed={selected}
            className="mt-3 min-h-10 w-full rounded-xl border border-sand/50 bg-white/[0.03] px-3 py-2 text-sm font-medium text-sand-strong transition-colors hover:bg-sand/15"
          >
            Play against the champion
          </button>
          {selected && search && (
            <div className="mt-3">
              <div role="group" aria-label="Champion thinking time" className="grid grid-cols-3 gap-2">
                {EFFORTS.flatMap(([key, label, description]) => {
                  const preset = search.presets[key];
                  if (!preset) return [];
                  return [
                    <button
                      key={key}
                      type="button"
                      aria-label={`${label} champion search`}
                      aria-pressed={budget.simulations === preset.simulations && budget.maxConsidered === preset.maxConsidered}
                      onClick={() => setBudget('server', { ...preset })}
                      className={`min-h-14 rounded-xl border px-2 py-2 text-left transition-colors ${
                        budget.simulations === preset.simulations && budget.maxConsidered === preset.maxConsidered
                          ? 'border-sand/70 bg-sand-faint'
                          : 'border-white/10 bg-white/[0.03] hover:border-sand/35'
                      }`}
                    >
                      <span className="block text-xs font-medium text-ink">{label}</span>
                      <span className="mt-1 block text-[10px] leading-relaxed text-muted">{description}</span>
                    </button>,
                  ];
                })}
              </div>
              <p className="mt-2 text-[11px] leading-relaxed text-muted">
                Every setting uses the same online champion. More search takes longer.
              </p>
            </div>
          )}
          <details className="mt-3 text-xs text-muted">
            <summary className="min-h-8 cursor-pointer py-1 text-sand-strong">Champion details</summary>
            <dl className="space-y-2 pb-1 pt-2">
              <div>
                <dt>Model version</dt>
                <dd className="mt-0.5 break-all font-mono text-ink">{champion.modelVersion}</dd>
              </div>
              <div>
                <dt>Full model identity</dt>
                <dd className="mt-0.5 break-all font-mono text-ink">{champion.modelIdentity}</dd>
              </div>
            </dl>
          </details>
        </>
      ) : (
        <p className="mt-2 text-xs leading-relaxed text-muted">
          {capability.status === 'checking'
            ? 'Checking that the champion is loaded and ready.'
            : 'The champion connection is not ready. You can still set up a two-player game.'}
        </p>
      )}
    </section>
  );
}
