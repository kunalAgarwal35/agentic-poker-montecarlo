'use client';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { PLAYER_COLORS, ACCENTS } from '@/components/viz/theme';
import { ChartTooltip } from '@/components/viz/ChartTooltip';

type Series = { name: string; points: { street: string; equity: number }[] };

export function EquityByStreetChart({ series }: { series: Series[] }) {
  const streets = series[0]?.points.map((p) => p.street) ?? [];
  const data = streets.map((street, i) => {
    const row: Record<string, number | string> = { street };
    series.forEach((s) => { row[s.name] = +(s.points[i].equity * 100).toFixed(1); });
    return row;
  });
  return (
    <div>
      <div style={{ width: '100%', height: 150 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ left: 0, right: 12, top: 4 }}>
            <XAxis dataKey="street" tick={{ fill: ACCENTS.text, fontSize: 10 }} />
            <YAxis domain={[0, 100]} width={32} tick={{ fill: ACCENTS.text, fontSize: 10 }} tickFormatter={(v) => `${v}%`} />
            <Tooltip content={<ChartTooltip />} />
            {series.map((s, i) => (
              <Line key={s.name} type="monotone" dataKey={s.name} stroke={PLAYER_COLORS[i % PLAYER_COLORS.length]} strokeWidth={2} dot isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <ul className="flex gap-3 text-xs text-zinc-400">
        {series.map((s, i) => <li key={s.name}><span style={{ color: PLAYER_COLORS[i % PLAYER_COLORS.length] }}>■</span> {s.name}</li>)}
      </ul>
    </div>
  );
}
