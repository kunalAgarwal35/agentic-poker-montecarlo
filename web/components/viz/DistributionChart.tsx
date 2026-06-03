'use client';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import type { DistributionBar } from '@/lib/types';
import { ACCENTS } from '@/components/viz/theme';
import { ChartTooltip } from '@/components/viz/ChartTooltip';

export function DistributionChart({ player, bars }: { player: string; bars: DistributionBar[] }) {
  const data = bars.map((b) => ({ label: b.label, pct: +(b.freq * 100).toFixed(1) }));
  const top = [...bars].sort((a, b) => b.freq - a.freq)[0];
  return (
    <div>
      <div className="mb-1 text-xs text-zinc-400">{player} — made-hand distribution by the river</div>
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
        Most likely: <span className="text-zinc-100">{top.label}</span>{' '}
        <span className="tabular-nums">{(top.freq * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}
