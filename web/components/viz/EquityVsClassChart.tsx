'use client';
import { ACCENTS } from '@/components/viz/theme';

export function EquityVsClassChart(
  { player, rows }:
  { player: string; rows: { category: string; label: string; equity: number; freq: number }[] },
) {
  return (
    <div>
      <div className="mb-1 text-xs text-zinc-400">{player} — equity by what the opponent makes</div>
      <div className="flex flex-col gap-1">
        {rows.map((r) => (
          <div key={r.category} className="flex items-center gap-2">
            <div className="w-20 shrink-0 truncate text-xs text-zinc-300">{r.label}</div>
            <div className="h-3 flex-1 rounded bg-zinc-800">
              <div className="h-3 rounded" style={{ width: `${(r.equity * 100).toFixed(1)}%`, background: ACCENTS.hero }} />
            </div>
            <div className="w-10 shrink-0 text-right text-xs tabular-nums text-zinc-100">{Math.round(r.equity * 100)}%</div>
            <div className="w-14 shrink-0 text-right text-[10px] tabular-nums text-zinc-500">{(r.freq * 100).toFixed(0)}% freq</div>
          </div>
        ))}
      </div>
    </div>
  );
}
