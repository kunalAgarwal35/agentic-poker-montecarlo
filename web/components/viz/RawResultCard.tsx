'use client';
import { useState } from 'react';
import type { RawResult } from '@/lib/types';
import { CardRow } from '@/components/viz/CardRow';
import { isConcreteHand } from '@/lib/cards';
import { ShareButton } from '@/components/ShareButton';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell } from 'recharts';
import { PLAYER_COLORS, ACCENTS } from '@/components/viz/theme';

function parseFrom(query: string) {
  const game = /game\s*=\s*'([^']*)'/i.exec(query)?.[1] ?? '';
  const board = /board\s*=\s*'([^']*)'/i.exec(query)?.[1] ?? '';
  const players: { name: string; cards: string }[] = [];
  const re = /(PLAYER_\d+)\s*=\s*'([^']*)'/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(query))) players.push({ name: m[1], cards: m[2] });
  return { game, board, players };
}

function fmtScalar(v: number | null | undefined): string {
  if (v === null || v === undefined) return 'n/a (no matching trials)';
  if (v >= 0 && v <= 1) return `${(v * 100).toFixed(1)}%`;
  return Number.isInteger(v) ? String(v) : v.toFixed(3);
}

export function RawResultCard({ data }: { data: RawResult }) {
  const { query, result } = data;
  const { game, board, players } = parseFrom(query);
  const usedWhere = /\bwhere\b/i.test(query);
  const scalarCols = result.columns.filter((c) => c in result.values);
  const histCols = result.columns.filter((c) => result.histograms != null && c in result.histograms);
  const [open, setOpen] = useState(false);
  const shareSummary = scalarCols.length
    ? `Poker math — ${scalarCols.map((c) => `${c} ${fmtScalar(result.values[c])}`).join(', ')}`
    : undefined;

  return (
    <div className="relative my-1.5 rounded-xl border border-zinc-700 bg-zinc-900/60 p-3 text-sm">
      <span className="absolute right-3 top-3 text-[10px] text-zinc-500">
        {result.trials.toLocaleString()} {usedWhere ? 'matching trials' : 'trials'} · {result.mode === 'enumeration' ? 'exact' : 'Monte Carlo'}
      </span>

      <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-0.5 pr-24 text-xs text-zinc-400">
        {game && (
          <span className="rounded-full border border-zinc-700 bg-zinc-800 px-2 py-0.5 font-medium text-zinc-400">{game}</span>
        )}
        {board && (<span className="flex items-center gap-1">board <CardRow cards={board} /></span>)}
        {players.map((p) => (
          <span key={p.name} className="flex items-center gap-1">{p.name} <CardRow cards={p.cards} fan={isConcreteHand(p.cards)} /></span>
        ))}
      </div>

      {scalarCols.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-2">
          {scalarCols.map((c) => (
            <div key={c} className="rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2">
              <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">{c}</div>
              <div className="text-2xl font-bold tabular-nums text-zinc-50">{fmtScalar(result.values[c])}</div>
            </div>
          ))}
        </div>
      )}

      {histCols.map((c) => {
        const hist = result.histograms![c];
        const rows = hist.pairs.map((p) => ({
          label: hist.labels ? hist.labels[String(p.value)] ?? String(p.value) : String(p.value),
          pct: result.trials ? (p.count / result.trials) * 100 : 0,
        }));
        return (
          <div key={c} className="mb-2">
            <div className="mb-1 font-mono text-xs text-zinc-400">{c}</div>
            <ResponsiveContainer width="100%" height={Math.max(120, rows.length * 26)}>
              <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 24 }}>
                <XAxis type="number" hide domain={[0, 100]} />
                <YAxis type="category" dataKey="label" width={90} tick={{ fill: ACCENTS.text, fontSize: 11 }} />
                <Bar dataKey="pct" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                  {rows.map((_, i) => <Cell key={i} fill={PLAYER_COLORS[i % PLAYER_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <ul className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-zinc-400">
              {rows.map((r) => (<li key={r.label}><span>{r.label}</span>: {r.pct.toFixed(1)}%</li>))}
            </ul>
          </div>
        );
      })}

      <div className="mt-2 flex items-center gap-3 border-t border-zinc-800 pt-2 text-xs text-zinc-500">
        <button type="button" onClick={() => setOpen((o) => !o)} className="hover:text-zinc-300">
          {open ? 'Hide' : 'Show'} query
        </button>
        <ShareButton summary={shareSummary} className="text-xs text-zinc-500 hover:text-zinc-300" />
      </div>
      {open && <code className="mt-2 block whitespace-pre-wrap break-all rounded bg-black/40 p-2 text-xs text-zinc-300">{query}</code>}
    </div>
  );
}
