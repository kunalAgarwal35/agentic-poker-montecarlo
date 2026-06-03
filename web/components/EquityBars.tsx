import type { EquityRow } from '@/lib/types';

export function EquityBars({ rows }: { rows: EquityRow[] }) {
  return (
    <div className="flex flex-col gap-2">
      {rows.map((r) => (
        <div key={r.name} className="flex items-center gap-2 text-sm">
          <span className={`w-24 truncate ${r.isHero ? 'font-semibold' : ''}`}>{r.name}</span>
          <div className="h-4 flex-1 rounded bg-zinc-800">
            <div
              className={`h-4 rounded ${r.isHero ? 'bg-green-500' : 'bg-zinc-500'}`}
              style={{ width: `${Math.round(r.equity * 100)}%` }}
            />
          </div>
          <span className="w-14 text-right tabular-nums">{(r.equity * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}
