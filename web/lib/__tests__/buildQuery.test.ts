import { describe, it, expect, vi } from 'vitest';
import { executeBuildQuery } from '@/lib/buildQuery';
import type { PQLResult } from '@/lib/types';

describe('executeBuildQuery', () => {
  it('compiles, runs the engine, and returns resolvedQuery + result + viz', async () => {
    const fakeResult: PQLResult = { values: { p1: 0.84, p2: 0.16 }, columns: ['p1', 'p2'], trials: 741, mode: 'enumeration', seed: null };
    const runPql = vi.fn().mockResolvedValue(fakeResult);

    const out = await executeBuildQuery(
      {
        game: 'omahahi5',
        players: [
          { name: 'Hero', cards: 'AsAhKsKhQs' },
          { name: 'Villain', cards: 'JdTd9d8d7d' },
        ],
        board: '2c3c4c',
        metrics: ['equity'],
      },
      { runPql },
    );

    expect(runPql).toHaveBeenCalledOnce();
    expect(out.resolvedQuery).toContain("board='2c3c4c'");
    expect(out.result.mode).toBe('enumeration');
    expect(out.viz.kind).toBe('equity');
  });

  it('surfaces engine errors', async () => {
    const runPql = vi.fn().mockRejectedValue(new Error('Engine error: Duplicate card(s) ... As'));
    await expect(
      executeBuildQuery(
        { game: 'omahahi5', players: [{ name: 'A', cards: 'AsAhKsKhQs' }, { name: 'B', cards: 'AsTd9d8d7d' }] },
        { runPql },
      ),
    ).rejects.toThrow(/Duplicate card/);
  });

  it('routes a graphQuery to runGraph and returns a graph viz', async () => {
    const runPql = vi.fn();
    const runGraph = vi.fn().mockResolvedValue({
      kind: 'vsclass',
      rows: [{ category: 'pair', label: 'One Pair', equity: 0.7, freq: 0.6 },
             { category: 'flush', label: 'Flush', equity: 0.1, freq: 0.4 }],
      trials: 20000, mode: 'monte_carlo', seed: null,
    });
    const out = await executeBuildQuery(
      {
        game: 'holdem',
        players: [{ name: 'PLAYER_1', cards: 'AsKs' }, { name: 'PLAYER_2', cards: 'QQ+' }],
        board: 'Ah7c2d',
        graphQuery: { kind: 'vsclass', hero: 'PLAYER_1' },
      },
      { runPql, runGraph },
    );
    expect(runPql).not.toHaveBeenCalled();
    expect(runGraph).toHaveBeenCalledOnce();
    expect(out.viz.kind).toBe('equity-vs-class');
    if (out.viz.kind === 'equity-vs-class') expect(out.viz.rows).toHaveLength(2);
  });

  it('returns a context carrying game, board, and players', async () => {
    const runPql = vi.fn().mockResolvedValue({ values: { p1: 0.6, p2: 0.4 }, columns: ['p1', 'p2'], trials: 100, mode: 'monte_carlo', seed: 1 } as PQLResult);
    const out = await executeBuildQuery(
      {
        game: 'omahahi5',
        players: [{ name: 'PLAYER_1', cards: 'AsAhKsKhQs' }, { name: 'PLAYER_2', cards: '25%' }],
        board: 'Kh7d2c',
      },
      { runPql },
    );
    expect(out.context.game).toBe('omahahi5');
    expect(out.context.board).toBe('Kh7d2c');
    expect(out.context.players).toEqual([
      { name: 'PLAYER_1', cards: 'AsAhKsKhQs' }, { name: 'PLAYER_2', cards: '25%' },
    ]);
  });
});
