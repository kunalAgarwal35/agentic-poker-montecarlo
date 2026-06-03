'use client';
import { ACCENTS } from '@/components/viz/theme';

export function DrawsChart({ player, bars }: { player: string; bars: { label: string; pct: number }[] }) {
  return (
    <div>
      <div className="mb-1 text-xs text-zinc-400">{player} — draw frequency</div>
      <div className="flex flex-col gap-1">
        {bars.map((b) => (
          <div key={b.label} className="flex items-center gap-2">
            <div className="w-20 shrink-0 truncate text-xs text-zinc-300">{b.label}</div>
            <div className="h-3 flex-1 rounded bg-zinc-800">
              <div className="h-3 rounded" style={{ width: `${(b.pct * 100).toFixed(1)}%`, background: ACCENTS.bar }} />
            </div>
            <div className="w-12 shrink-0 text-right text-xs tabular-nums text-zinc-100">{(b.pct * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>
    </div>
  );
}
