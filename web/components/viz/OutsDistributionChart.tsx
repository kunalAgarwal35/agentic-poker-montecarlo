'use client';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { ACCENTS } from '@/components/viz/theme';
import { ChartTooltip } from '@/components/viz/ChartTooltip';

export function OutsDistributionChart(
  { player, handtype, street, avg, bars }:
  { player: string; handtype: string; street: string; avg: number; bars: { outs: number; prob: number }[] },
) {
  const lastOuts = bars.length ? bars[bars.length - 1].outs : 0;
  const data = bars.map((b) => ({
    label: b.outs === lastOuts ? `${b.outs}+` : `${b.outs}`,
    pct: +(b.prob * 100).toFixed(1),
  }));
  return (
    <div>
      <div className="mb-1 text-xs text-zinc-400">
        {player} — outs to a {handtype} on the {street}
      </div>
      <div style={{ width: '100%', height: 150 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ left: 0, right: 8, top: 4 }}>
            <XAxis dataKey="label" tick={{ fill: ACCENTS.text, fontSize: 10 }} interval={0} />
            <YAxis width={32} tick={{ fill: ACCENTS.text, fontSize: 10 }} tickFormatter={(v) => `${v}%`} />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
            <Bar dataKey="pct" fill={ACCENTS.bar} radius={[3, 3, 0, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="text-xs text-zinc-400">
        Average: <span className="tabular-nums text-zinc-100">{avg.toFixed(1)}</span> outs
      </div>
    </div>
  );
}
