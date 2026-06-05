'use client';
import { useState } from 'react';
import type { VizSpec, HeroDraws, EquityRow, WinTieLossRow } from '@/lib/types';
import type { ResultContext } from '@/lib/buildQuery';
import { ResultViz } from '@/components/viz/ResultViz';
import { CardRow } from '@/components/viz/CardRow';
import { ShareButton } from '@/components/ShareButton';
import { ACCENTS } from '@/components/viz/theme';
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

// The "headline" pct keyed by player name, used both for the verdict and to
// co-locate each player's number next to their cards in the matchup header.
// Only equity / heads-up win-tie-loss vizzes expose a single hero number.
function headlinePcts(viz: VizSpec): { byName: Record<string, number>; hero?: { name: string; pct: number } } | null {
  if (viz.kind === 'equity') {
    const rows = viz.rows as EquityRow[];
    if (!rows.length) return null;
    const byName: Record<string, number> = {};
    rows.forEach((r) => { byName[r.name] = r.equity; });
    const heroRow = rows.find((r) => r.isHero) ?? rows[0];
    return { byName, hero: { name: heroRow.name, pct: heroRow.equity } };
  }
  if (viz.kind === 'win-tie-loss') {
    const rows = viz.rows as WinTieLossRow[];
    if (!rows.length) return null;
    const byName: Record<string, number> = {};
    rows.forEach((r) => { byName[r.name] = r.win; });
    return { byName, hero: { name: rows[0].name, pct: rows[0].win } };
  }
  return null;
}

type Verdict = { label: 'Ahead' | 'Behind' | 'Coinflip'; color: string };
function verdictFor(pct: number): Verdict {
  if (pct >= 0.55) return { label: 'Ahead', color: ACCENTS.hero };
  if (pct <= 0.45) return { label: 'Behind', color: ACCENTS.behind };
  return { label: 'Coinflip', color: ACCENTS.coinflip };
}

// The headline verdict line: only for equity / win-tie-loss with exactly 2 rows
// (heads-up). Multiway and other viz kinds skip it. Returns null when not
// applicable so the header can lay out the right-side meta on its own.
function VerdictLine({ viz }: { viz: VizSpec }) {
  const rowCount =
    viz.kind === 'equity' ? viz.rows.length :
    viz.kind === 'win-tie-loss' ? viz.rows.length : 0;
  if ((viz.kind !== 'equity' && viz.kind !== 'win-tie-loss') || rowCount !== 2) return null;
  const head = headlinePcts(viz);
  if (!head?.hero) return null;
  const { name, pct } = head.hero;
  const v = verdictFor(pct);
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
      <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: v.color }} aria-hidden />
      <span className="font-semibold text-zinc-100">
        {name} {v.label.toLowerCase()}
      </span>
      <span className="text-zinc-500">—</span>
      <span className="text-xl font-bold tabular-nums text-zinc-50">{Math.round(pct * 100)}%</span>
      <span
        className="rounded-full px-2 py-0.5 text-xs font-semibold"
        style={{ color: v.color, background: `${v.color}1f`, border: `1px solid ${v.color}55` }}
      >
        {v.label}
      </span>
    </div>
  );
}

// Top header row: verdict on the LEFT, the mode badge stacked above the game
// chip on the RIGHT, with comfortable spacing via justify-between. This
// replaces the old absolutely-positioned mode badge + the game chip that used
// to live inside MatchupHeader, so the three pieces no longer overlap.
function CardHeader({ viz, game }: { viz: VizSpec; game?: string }) {
  return (
    <div className="mb-2 flex items-start justify-between gap-3">
      <div className="min-w-0">
        <VerdictLine viz={viz} />
      </div>
      <div className="flex shrink-0 flex-col items-end gap-0.5 text-right">
        <span className="text-[10px] text-zinc-500">{modeBadge(viz)}</span>
        {game && (
          <span className="rounded-full border border-zinc-700 bg-zinc-800 px-2 py-0.5 text-xs font-medium text-zinc-400">
            {game}
          </span>
        )}
      </div>
    </div>
  );
}

// Redesigned matchup header: game chip, board row, and a co-located row per
// player (cards / range pill + their inline % when the viz exposes one).
function MatchupHeader({ context, viz }: { context: ResultContext; viz: VizSpec }) {
  const hasBoard = !!context.board && context.board.length > 0;
  const head = headlinePcts(viz);
  const pctSuffix = viz.kind === 'win-tie-loss' ? ' win' : '';
  return (
    <div className="mb-2 text-xs">
      {hasBoard && (
        <div className="mb-1.5 flex flex-wrap items-center gap-x-2 gap-y-0.5">
          <span className="w-10 shrink-0 uppercase tracking-wide text-zinc-500">Board</span>
          <CardRow cards={context.board as string} />
        </div>
      )}
      <div className="flex flex-col gap-1">
        {context.players.map((pl, idx) => {
          const isHero = head?.hero?.name === pl.name;
          const pct = head?.byName[pl.name];
          return (
            <div
              key={pl.name}
              className="flex flex-wrap items-center gap-x-2 gap-y-0.5 rounded-lg border-l-2 py-0.5 pl-2"
              style={isHero
                ? { borderColor: ACCENTS.hero, background: `${ACCENTS.hero}14` }
                : { borderColor: 'transparent' }}
            >
              <span className="w-8 shrink-0 text-zinc-500">P{idx + 1}</span>
              {isConcreteHand(pl.cards)
                ? <CardRow cards={pl.cards} fan />
                : <span className="rounded-full border border-zinc-700 bg-zinc-800 px-2 py-0.5 text-zinc-200">{rangeLabel(pl.cards)}</span>}
              {typeof pct === 'number' && (
                <span className={`ml-auto tabular-nums ${isHero ? 'font-semibold text-zinc-100' : 'text-zinc-400'}`}>
                  {(pct * 100).toFixed(1)}%{pctSuffix}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Engine-verified draws as pills: ONE pill per true draw with nonzero outs.
function drawsPills(d: HeroDraws): { label: string }[] {
  const pills: { label: string }[] = [];
  if (d.flushDraw) {
    const fd = d.nutFlushDraw ? 'Nut flush draw' : 'Flush draw';
    pills.push({ label: d.flushOuts > 0 ? `${fd} · ${d.flushOuts} outs` : fd });
  }
  if (d.oesd) pills.push({ label: d.straightOuts > 0 ? `Open-ended straight draw · ${d.straightOuts} outs` : 'Open-ended straight draw' });
  else if (d.straightDraw) pills.push({ label: d.straightOuts > 0 ? `Straight draw · ${d.straightOuts} outs` : 'Straight draw' });
  if (d.gutshot) pills.push({ label: 'Gutshot' });
  return pills;
}

function DrawsPills({ heroDraws }: { heroDraws: HeroDraws }) {
  const pills = drawsPills(heroDraws);
  if (!pills.length) return null;
  return (
    <div className="mt-2">
      <div className="mb-1 text-[10px] uppercase tracking-wide text-zinc-600">Engine-checked draws</div>
      <div className="flex flex-wrap gap-1.5">
        {pills.map((p) => (
          <span
            key={p.label}
            className="rounded-full border border-sky-500/30 bg-sky-500/10 px-2 py-0.5 text-xs font-medium text-sky-300"
          >
            {p.label}
          </span>
        ))}
      </div>
    </div>
  );
}

export function ResultCard({ resolvedQuery, viz, context, heroDraws }: { resolvedQuery: string; viz: VizSpec; context?: ResultContext; heroDraws?: HeroDraws }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="my-1.5 rounded-xl border border-zinc-700 bg-zinc-900 p-3">
      <CardHeader viz={viz} game={context?.game} />
      {context && <MatchupHeader context={context} viz={viz} />}
      <ResultViz spec={viz} />
      {heroDraws && <DrawsPills heroDraws={heroDraws} />}
      <div className="mt-2 flex items-center gap-3 border-t border-zinc-800 pt-2 text-xs text-zinc-500">
        <button onClick={() => setOpen((o) => !o)} className="hover:text-zinc-300">
          {open ? 'Hide' : 'Show'} query
        </button>
        <ShareButton summary={shareSummary(viz)} className="text-xs text-zinc-500 hover:text-zinc-300" />
      </div>
      {open && <pre className="mt-2 overflow-auto rounded bg-black/40 p-2 text-xs text-zinc-300">{resolvedQuery}</pre>}
    </div>
  );
}
