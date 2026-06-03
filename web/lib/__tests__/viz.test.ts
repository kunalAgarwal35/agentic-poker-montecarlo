import { describe, it, expect } from 'vitest';
import { pickViz } from '@/lib/viz';
import { QueryIntent } from '@/lib/intent';
import type { PQLResult } from '@/lib/types';

const players = [
  { name: 'Hero', cards: 'AsKs' },
  { name: 'Villain', cards: 'QdQh' },
];

describe('pickViz', () => {
  it('equity comparison with hero flagged', () => {
    const intent = QueryIntent.parse({ game: 'holdem', players, metrics: ['equity'] });
    const result: PQLResult = { values: { p1: 0.46, p2: 0.54 }, columns: ['p1', 'p2'], trials: 5000, mode: 'monte_carlo', seed: 1 };
    const viz = pickViz(intent, result);
    expect(viz.kind).toBe('equity');
    if (viz.kind !== 'equity') throw new Error('x');
    expect(viz.rows[0]).toEqual({ name: 'P1', equity: 0.46, isHero: true });
  });

  it('labels equity rows by compact P-tags (the header shows the hands)', () => {
    const intent = QueryIntent.parse({ game: 'holdem', players, metrics: ['equity'] });
    const result: PQLResult = { values: { p1: 0.46, p2: 0.54 }, columns: ['p1', 'p2'], trials: 5000, mode: 'monte_carlo', seed: 1 };
    const viz = pickViz(intent, result);
    if (viz.kind !== 'equity') throw new Error('x');
    expect(viz.rows.map((r) => r.name)).toEqual(['P1', 'P2']);
  });

  it('win-tie-loss from wins+ties counts', () => {
    const intent = QueryIntent.parse({ game: 'holdem', players, metrics: ['winsHi', 'tiesHi'] });
    const result: PQLResult = { values: { p1_wins: 80, p1_ties: 4, p2_wins: 16, p2_ties: 4 }, columns: [], trials: 100, mode: 'monte_carlo', seed: 1 };
    const viz = pickViz(intent, result);
    expect(viz.kind).toBe('win-tie-loss');
    if (viz.kind !== 'win-tie-loss') throw new Error('x');
    expect(viz.rows[0]).toEqual({ name: 'P1', win: 0.8, tie: 0.04, loss: 0.16 });
  });

  it('distribution from per-category counts', () => {
    const intent = QueryIntent.parse({ game: 'holdem', players, metrics: ['distribution'] });
    const values: Record<string, number> = {};
    for (const t of ['highcard','pair','twopair','trips','straight','flush','fullhouse','quads','straightflush']) values[`p1_${t}`] = 0;
    values['p1_pair'] = 50; values['p1_twopair'] = 50;
    const result: PQLResult = { values, columns: [], trials: 100, mode: 'monte_carlo', seed: 1 };
    const viz = pickViz(intent, result);
    expect(viz.kind).toBe('distribution');
    if (viz.kind !== 'distribution') throw new Error('x');
    expect(viz.bars.find((b) => b.token === 'pair')!.freq).toBe(0.5);
    expect(viz.bars).toHaveLength(9);
  });

  it('maps nutHi to a frequency viz', () => {
    const intent = QueryIntent.parse({ game: 'holdem', players, metrics: ['nutHi'] });
    const result: PQLResult = { values: { p1_nuts: 25, p2_nuts: 5 }, columns: [], trials: 100, mode: 'monte_carlo', seed: 1 };
    const viz = pickViz(intent, result);
    expect(viz.kind).toBe('frequency');
    if (viz.kind !== 'frequency') throw new Error('x');
    expect(viz.pct).toBe(0.25);
    expect(viz.label).toMatch(/nut/i);
  });

  it('maps winningDistribution to a distribution viz', () => {
    const intent = QueryIntent.parse({ game: 'holdem', players, metrics: ['winningDistribution'] });
    const values: Record<string, number> = {};
    for (const t of ['highcard','pair','twopair','trips','straight','flush','fullhouse','quads','straightflush']) values[`win_${t}`] = 0;
    values['win_pair'] = 60; values['win_twopair'] = 40;
    const result: PQLResult = { values, columns: [], trials: 100, mode: 'monte_carlo', seed: 1 };
    const viz = pickViz(intent, result);
    expect(viz.kind).toBe('distribution');
  });
});

describe('pickViz draws & outs', () => {
  it('builds a draws viz from draw counts', () => {
    const intent = QueryIntent.parse({
      game: 'holdem',
      players: [{ name: 'PLAYER_1', cards: 'KhQh' }, { name: 'PLAYER_2', cards: 'AsAd' }],
      board: 'Ah7c2h',
      drawQuery: { player: 'PLAYER_1', kinds: ['flushDraw', 'oesd'] },
    });
    const result = { values: { p1_flushDraw: 100, p1_oesd: 0 }, columns: [], trials: 100, mode: 'monte_carlo' as const, seed: 1 };
    const spec = pickViz(intent, result);
    expect(spec.kind).toBe('draws');
    if (spec.kind === 'draws') {
      expect(spec.bars.find((b) => b.label.toLowerCase().includes('flush'))?.pct).toBeCloseTo(1);
    }
  });

  it('builds an outs-distribution viz from cumulative buckets', () => {
    const intent = QueryIntent.parse({
      game: 'holdem',
      players: [{ name: 'PLAYER_1', cards: 'KhQh' }, { name: 'PLAYER_2', cards: 'AsAd' }],
      board: 'Ah7c2h',
      outsQuery: { player: 'PLAYER_1', handtype: 'flush', street: 'turn', buckets: 2 },
    });
    // ge1=100, ge2=100 -> exactly-1 = 0, exactly-2(+) tail = 100
    const result = { values: { p1_avgouts: 2, p1_ge1: 100, p1_ge2: 100 }, columns: [], trials: 100, mode: 'monte_carlo' as const, seed: 1 };
    const spec = pickViz(intent, result);
    expect(spec.kind).toBe('outs-distribution');
    if (spec.kind === 'outs-distribution') {
      expect(spec.avg).toBe(2);
      expect(spec.bars.length).toBe(3); // outs = 0, 1, 2+
    }
  });
});
