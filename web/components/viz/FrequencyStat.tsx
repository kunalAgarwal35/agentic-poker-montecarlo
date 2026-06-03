'use client';
import { ACCENTS } from '@/components/viz/theme';

export function FrequencyStat({ label, pct }: { label: string; pct: number }) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-sm">
        <span className="text-zinc-200">{label}</span>
        <span className="tabular-nums font-semibold text-zinc-100">{(pct * 100).toFixed(1)}%</span>
      </div>
      <div className="h-3 w-full rounded bg-zinc-800">
        <div className="h-3 rounded" style={{ width: `${Math.round(pct * 100)}%`, background: ACCENTS.hero }} />
      </div>
    </div>
  );
}
