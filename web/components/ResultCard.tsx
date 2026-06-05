'use client';
import { useState } from 'react';
import type { VizSpec, HeroDraws } from '@/lib/types';
import type { ResultContext } from '@/lib/buildQuery';
import { ResultViz } from '@/components/viz/ResultViz';
import { CardRow } from '@/components/viz/CardRow';
import { ShareButton } from '@/components/ShareButton';
import { isConcreteHand, rangeLabel } from '@/lib/cards';

function modeBadge(viz: VizSpec) {
  return viz.mode === 'enumeration' ? `Exact · ${viz.trials} runouts` : `Monte Carlo · ${viz.trials} trials`;
}

// A one-line, share-friendly takeaway derived from the viz when easily available.
function shareSummary(viz: VizSpec): string | undefined {
  if (viz.kind === 'equity' && Array.isArray(viz.rows)) {
    const parts = viz.rows.map((r) => `${r.name} ${(r.equity * 100).toFixed(1)}%`);
    if (parts.length) return `Poker equity — ${parts.join(' vs ')}`;
  }
  if (viz.kind === 'frequency' && typeof viz.pct === 'number') {
    return `Poker math — ${viz.label}: ${(viz.pct * 100).toFixed(1)}%`;
  }
  return undefined;
}

function SetupHeader({ context }: { context: ResultContext }) {
  const hasBoard = !!context.board && context.board.length > 0;
  return (
    <div className="mb-2 flex flex-col gap-1 text-xs">
      {context.game && (
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="w-10 shrink-0 text-zinc-500">Game</span>
          <span className="font-medium text-zinc-300">{context.game}</span>
        </div>
      )}
      {hasBoard && (
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="w-10 shrink-0 text-zinc-500">Board</span>
          <CardRow cards={context.board as string} />
        </div>
      )}
      {context.players.map((pl, idx) => (
        <div key={pl.name} className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="w-10 shrink-0 text-zinc-500">P{idx + 1}</span>
          {isConcreteHand(pl.cards)
            ? <CardRow cards={pl.cards} />
            : <span className="rounded-full border border-zinc-700 bg-zinc-800 px-2 py-0.5 text-zinc-200">{rangeLabel(pl.cards)}</span>}
        </div>
      ))}
    </div>
  );
}

// Engine-verified draws line: lists ONLY the true draws + nonzero out counts.
function drawsSummary(d: HeroDraws): string | undefined {
  const parts: string[] = [];
  if (d.flushDraw) parts.push(d.flushOuts > 0 ? `flush draw (${d.flushOuts} outs)` : 'flush draw');
  if (d.oesd) parts.push('OESD');
  else if (d.straightDraw) parts.push('straight draw');
  if (d.gutshot) parts.push('gutshot');
  if (!parts.length) return undefined;
  return `Engine-checked draws: ${parts.join(', ')}`;
}

export function ResultCard({ resolvedQuery, viz, context, heroDraws }: { resolvedQuery: string; viz: VizSpec; context?: ResultContext; heroDraws?: HeroDraws }) {
  const [open, setOpen] = useState(false);
  const draws = heroDraws ? drawsSummary(heroDraws) : undefined;
  return (
    <div className="my-2 rounded-xl border border-zinc-700 bg-zinc-900 p-3">
      <div className="mb-2 flex items-center gap-2 text-xs text-zinc-400">
        <span>{modeBadge(viz)}</span>
      </div>
      {context && <SetupHeader context={context} />}
      <ResultViz spec={viz} />
      {draws && <div className="mt-2 text-xs text-zinc-500">{draws}</div>}
      <button onClick={() => setOpen((o) => !o)} className="mt-3 text-xs text-zinc-500 hover:text-zinc-300">
        {open ? 'Hide' : 'Show'} query
      </button>
      <ShareButton summary={shareSummary(viz)} />
      {open && <pre className="mt-2 overflow-auto rounded bg-black/40 p-2 text-xs text-zinc-300">{resolvedQuery}</pre>}
    </div>
  );
}
