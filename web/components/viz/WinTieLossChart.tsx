'use client';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import type { WinTieLossRow } from '@/lib/types';
import { ACCENTS } from '@/components/viz/theme';
import { ChartTooltip } from '@/components/viz/ChartTooltip';

export function WinTieLossChart({ rows }: { rows: WinTieLossRow[] }) {
  const data = rows.map((r) => ({ name: r.name, Win: +(r.win * 100).toFixed(1), Tie: +(r.tie * 100).toFixed(1), Loss: +(r.loss * 100).toFixed(1) }));
  return (
    <div>
      <div style={{ width: '100%', height: 40 + rows.length * 34 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
            <XAxis type="number" domain={[0, 100]} tick={{ fill: ACCENTS.text, fontSize: 10 }} tickFormatter={(v) => `${v}%`} height={18} />
            <YAxis type="category" dataKey="name" width={28} tick={{ fill: ACCENTS.text, fontSize: 12 }} />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
            <Bar dataKey="Win" stackId="a" fill={ACCENTS.hero} isAnimationActive={false} />
            <Bar dataKey="Tie" stackId="a" fill={ACCENTS.fourth} isAnimationActive={false} />
            <Bar dataKey="Loss" stackId="a" fill="#484f58" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <ul className="mt-1 flex flex-col gap-1 text-xs text-zinc-400">
        {rows.map((r) => (
          <li key={r.name}><span className="text-zinc-100">{r.name}</span>{' — '}
            Win {(r.win * 100).toFixed(1)}% · Tie {(r.tie * 100).toFixed(1)}% · Lose {(r.loss * 100).toFixed(1)}%</li>
        ))}
      </ul>
    </div>
  );
}
