'use client';
import type { RawResult } from '@/lib/types';
import { CardRow } from '@/components/viz/CardRow';
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

  return (
    <div className="rounded-xl border border-zinc-700 bg-zinc-900/60 p-3 text-sm">
      <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-zinc-400">
        {game && <span className="font-medium text-zinc-300">{game}</span>}
        {board && (<span className="flex items-center gap-1">board <CardRow cards={board} /></span>)}
        {players.map((p) => (
          <span key={p.name} className="flex items-center gap-1">{p.name} <CardRow cards={p.cards} /></span>
        ))}
      </div>

      {scalarCols.length > 0 && (
        <table className="mb-2 w-full text-left">
          <tbody>
            {scalarCols.map((c) => (
              <tr key={c} className="border-t border-zinc-800">
                <td className="py-1 pr-3 font-mono text-zinc-400">{c}</td>
                <td className="py-1 font-medium text-zinc-100">{fmtScalar(result.values[c])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {histCols.map((c) => {
        const hist = result.histograms![c];
        const rows = hist.pairs.map((p) => ({
          label: hist.labels ? hist.labels[String(p.value)] ?? String(p.value) : String(p.value),
          pct: result.trials ? (p.count / result.trials) * 100 : 0,
        }));
        return (
          <div key={c} className="mb-1">
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

      <div className="mt-1 text-xs text-zinc-500">
        {result.trials.toLocaleString()} {usedWhere ? 'matching trials' : 'trials'} · {result.mode === 'enumeration' ? 'exact' : 'Monte Carlo'}
      </div>
      <details className="mt-1 text-xs text-zinc-500">
        <summary className="cursor-pointer">query</summary>
        <code className="block whitespace-pre-wrap break-all pt-1 text-zinc-400">{query}</code>
      </details>
    </div>
  );
}
