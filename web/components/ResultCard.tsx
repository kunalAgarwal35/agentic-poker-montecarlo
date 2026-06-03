'use client';
import { useState } from 'react';
import type { VizSpec } from '@/lib/types';
import type { ResultContext } from '@/lib/buildQuery';
import { ResultViz } from '@/components/viz/ResultViz';
import { CardRow } from '@/components/viz/CardRow';
import { isConcreteHand, rangeLabel } from '@/lib/cards';

function modeBadge(viz: VizSpec) {
  return viz.mode === 'enumeration' ? `Exact · ${viz.trials} runouts` : `Monte Carlo · ${viz.trials} trials`;
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

export function ResultCard({ resolvedQuery, viz, context }: { resolvedQuery: string; viz: VizSpec; context?: ResultContext }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="my-2 rounded-xl border border-zinc-700 bg-zinc-900 p-3">
      <div className="mb-2 flex items-center gap-2 text-xs text-zinc-400">
        <span>{modeBadge(viz)}</span>
      </div>
      {context && <SetupHeader context={context} />}
      <ResultViz spec={viz} />
      <button onClick={() => setOpen((o) => !o)} className="mt-3 text-xs text-zinc-500 hover:text-zinc-300">
        {open ? 'Hide' : 'Show'} query
      </button>
      {open && <pre className="mt-2 overflow-auto rounded bg-black/40 p-2 text-xs text-zinc-300">{resolvedQuery}</pre>}
    </div>
  );
}
