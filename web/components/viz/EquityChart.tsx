'use client';
import { BarChart, Bar, XAxis, YAxis, Cell, Tooltip, ResponsiveContainer } from 'recharts';
import type { EquityRow } from '@/lib/types';
import { PLAYER_COLORS, ACCENTS } from '@/components/viz/theme';
import { ChartTooltip } from '@/components/viz/ChartTooltip';

// Shorten a technical name like "PLAYER_1" to a compact axis label ("P1").
function shortName(name: string, idx: number): string {
  const m = /^PLAYER[_\s-]?(\d+)$/i.exec(name);
  if (m) return `P${m[1]}`;
  return name.length <= 4 ? name : `P${idx + 1}`;
}

export function EquityChart({ rows }: { rows: EquityRow[] }) {
  const data = rows.map((r, i) => ({
    name: r.name,
    short: shortName(r.name, i),
    equity: +(r.equity * 100).toFixed(1),
    fill: r.isHero ? ACCENTS.hero : PLAYER_COLORS[i % PLAYER_COLORS.length],
  }));
  return (
    <div>
      <div style={{ width: '100%', height: 40 + rows.length * 34 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
            <XAxis type="number" domain={[0, 100]} tick={{ fill: ACCENTS.text, fontSize: 10 }} tickFormatter={(v) => `${v}%`} height={18} />
            <YAxis type="category" dataKey="short" width={28} tick={{ fill: ACCENTS.text, fontSize: 12 }} />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
            <Bar dataKey="equity" radius={[0, 4, 4, 0]} isAnimationActive={false}>
              {data.map((d, i) => <Cell key={i} fill={d.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <ul className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-zinc-400">
        {data.map((d, i) => (
          <li key={d.name} className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 shrink-0 rounded-full" style={{ background: d.fill }} aria-hidden />
            <span className={rows[i].isHero ? 'font-semibold text-zinc-100' : ''}>{d.name}</span>{' '}
            <span className="tabular-nums">{(rows[i].equity * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
