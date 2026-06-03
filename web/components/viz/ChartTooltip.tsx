'use client';
import type { TooltipContentProps } from 'recharts';

export function ChartTooltip({ active, payload, label }: Partial<TooltipContentProps<number, string>>) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="rounded border border-zinc-700 bg-zinc-900/95 px-2 py-1 text-xs text-zinc-100 shadow">
      {label !== undefined && <div className="mb-0.5 text-zinc-400">{label}</div>}
      {payload.map((p, i) => (
        <div key={i} className="tabular-nums">
          <span className="text-zinc-400">{p.name}: </span>{(p.value as number).toFixed(1)}%
        </div>
      ))}
    </div>
  );
}
