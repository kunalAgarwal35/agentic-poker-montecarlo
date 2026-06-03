'use client';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { ACCENTS } from '@/components/viz/theme';
import { ChartTooltip } from '@/components/viz/ChartTooltip';

export function EquityDistributionChart(
  { player, mean, combos, sampledCombos, buckets }:
  { player: string; mean: number; combos: number; sampledCombos: number | null;
    buckets: { lo: number; hi: number; pct: number }[] },
) {
  const data = buckets.map((b) => ({ label: `${b.lo}`, pct: +(b.pct * 100).toFixed(1) }));
  return (
    <div>
      <div className="mb-1 text-xs text-zinc-400">
        {player} — equity distribution vs the range
        {sampledCombos ? ` (sampled ${sampledCombos} of ${combos}+ combos)` : ` (${combos} combos)`}
      </div>
      <div style={{ width: '100%', height: 150 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ left: 0, right: 8, top: 4 }}>
            <XAxis dataKey="label" tick={{ fill: ACCENTS.text, fontSize: 10 }} interval={1} />
            <YAxis width={32} tick={{ fill: ACCENTS.text, fontSize: 10 }} tickFormatter={(v) => `${v}%`} />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
            <Bar dataKey="pct" fill={ACCENTS.bar} radius={[3, 3, 0, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="text-xs text-zinc-400">
        Mean equity: <span className="tabular-nums text-zinc-100">{(mean * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}
