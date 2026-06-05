import { describe, it, expect, vi } from 'vitest';
import { executeBuildQuery } from '@/lib/buildQuery';
import type { PQLResult } from '@/lib/types';

// The draws probe is a single extra runPql whose query contains 'flushDraw'.
// Helper to build a runPql mock that answers the draws probe and the main query
// separately, so tests can assert on each independently.
function runPqlWith(opts: { main: PQLResult; draws?: PQLResult }) {
  const mainResult = opts.main;
  const drawsResult: PQLResult = opts.draws ?? {
    values: { fd: 0, od: 0, gs: 0, sd: 0, fo: 0, so: 0 },
    columns: ['fd', 'od', 'gs', 'sd', 'fo', 'so'],
    trials: 400, mode: 'enumeration', seed: null,
  };
  return vi.fn(async (q: string) => (q.includes('flushDraw') ? drawsResult : mainResult));
}

describe('executeBuildQuery', () => {
  it('compiles, runs the engine, and returns resolvedQuery + result + viz', async () => {
    const fakeResult: PQLResult = { values: { p1: 0.84, p2: 0.16 }, columns: ['p1', 'p2'], trials: 741, mode: 'enumeration', seed: null };
    const runPql = runPqlWith({ main: fakeResult });

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

    // One call for the draws probe (flop + exact hero) and one for the main query.
    expect(runPql).toHaveBeenCalledTimes(2);
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
    const runPql = vi.fn(async () => ({
      values: { fd: 0, od: 0, gs: 0, sd: 0, fo: 0, so: 0 },
      columns: [], trials: 400, mode: 'enumeration', seed: null,
    } as PQLResult));
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
    expect(runGraph).toHaveBeenCalledOnce();
    expect(out.viz.kind).toBe('equity-vs-class');
    if (out.viz.kind === 'equity-vs-class') expect(out.viz.rows).toHaveLength(2);
  });

  it('returns a context carrying game, board, and players', async () => {
    const runPql = runPqlWith({ main: { values: { p1: 0.6, p2: 0.4 }, columns: ['p1', 'p2'], trials: 100, mode: 'monte_carlo', seed: 1 } });
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

  describe('heroDraws (engine-verified)', () => {
    it('attaches heroDraws on a flop board with an exact holdem hero', async () => {
      const main: PQLResult = { values: { p1: 0.7, p2: 0.3 }, columns: ['p1', 'p2'], trials: 100, mode: 'monte_carlo', seed: null };
      const draws: PQLResult = {
        values: { fd: 400, od: 0, gs: 400, sd: 0, fo: 9, so: 0 },
        columns: ['fd', 'od', 'gs', 'sd', 'fo', 'so'], trials: 400, mode: 'enumeration', seed: null,
      };
      const runPql = runPqlWith({ main, draws });
      const out = await executeBuildQuery(
        {
          game: 'holdem',
          players: [{ name: 'PLAYER_1', cards: 'KhQh' }, { name: 'PLAYER_2', cards: 'AsAd' }],
          board: 'Ah7h2c',
        },
        { runPql },
      );
      expect(out.heroDraws).toEqual({
        player: 'PLAYER_1',
        flushDraw: true,
        // KhQh on Ah7h2c: board heart is A, highest LIVE heart is K, hero holds Kh → nut.
        nutFlushDraw: true,
        straightDraw: false,
        oesd: false,
        gutshot: true,
        flushOuts: 9,
        straightOuts: 0,
      });
    });

    // Mock the engine probe to always report a flush draw, so these cases isolate
    // the pure nut-flush-draw card logic computed in code.
    function flushDrawProbe(): PQLResult {
      return {
        values: { fd: 400, od: 0, gs: 0, sd: 0, fo: 9, so: 0 },
        columns: ['fd', 'od', 'gs', 'sd', 'fo', 'so'], trials: 400, mode: 'enumeration', seed: null,
      };
    }
    const main: PQLResult = { values: { p1: 0.7, p2: 0.3 }, columns: ['p1', 'p2'], trials: 100, mode: 'monte_carlo', seed: null };

    it('marks the A-high flush draw as the nut flush draw', async () => {
      // AsKsQhJh, board 2s7sTd: spade draw, no spade on board, hero holds As → nut.
      const runPql = runPqlWith({ main, draws: flushDrawProbe() });
      const out = await executeBuildQuery(
        {
          game: 'omahahi',
          players: [{ name: 'PLAYER_1', cards: 'AsKsQhJh' }, { name: 'PLAYER_2', cards: 'AsAd' }],
          board: '2s7sTd',
        },
        { runPql },
      );
      expect(out.heroDraws?.flushDraw).toBe(true);
      expect(out.heroDraws?.nutFlushDraw).toBe(true);
    });

    it('treats the K-high draw as nut when the ace of that suit is on the board', async () => {
      // KhQh, board Ah7h2h: Ah is on the board (plays for all), highest live heart is K,
      // hero holds Kh → nut flush draw.
      const runPql = runPqlWith({ main, draws: flushDrawProbe() });
      const out = await executeBuildQuery(
        {
          game: 'holdem',
          players: [{ name: 'PLAYER_1', cards: 'KhQh' }, { name: 'PLAYER_2', cards: 'AsAd' }],
          board: 'Ah7h2h',
        },
        { runPql },
      );
      expect(out.heroDraws?.nutFlushDraw).toBe(true);
    });

    it('does NOT mark a non-nut flush draw (hero lacks the highest live card)', async () => {
      // JhTh, board Ah7h2h: Ah on board, highest live heart is K which the hero does NOT hold → not nut.
      const runPql = runPqlWith({ main, draws: flushDrawProbe() });
      const out = await executeBuildQuery(
        {
          game: 'holdem',
          players: [{ name: 'PLAYER_1', cards: 'JhTh' }, { name: 'PLAYER_2', cards: 'AsAd' }],
          board: 'Ah7h2h',
        },
        { runPql },
      );
      expect(out.heroDraws?.flushDraw).toBe(true);
      expect(out.heroDraws?.nutFlushDraw).toBe(false);
    });

    it('nutFlushDraw is false when the engine reports no flush draw', async () => {
      // Even though AsKs would be the nut spade draw on a spade board, the probe says
      // flushDraw=false (e.g. fewer than 4 spades) so nutFlushDraw must be false too.
      const noFlush: PQLResult = {
        values: { fd: 0, od: 0, gs: 0, sd: 0, fo: 0, so: 0 },
        columns: ['fd', 'od', 'gs', 'sd', 'fo', 'so'], trials: 400, mode: 'enumeration', seed: null,
      };
      const runPql = runPqlWith({ main, draws: noFlush });
      const out = await executeBuildQuery(
        {
          game: 'holdem',
          players: [{ name: 'PLAYER_1', cards: 'AsKs' }, { name: 'PLAYER_2', cards: 'QdQc' }],
          board: '7s2sTd',
        },
        { runPql },
      );
      expect(out.heroDraws?.flushDraw).toBe(false);
      expect(out.heroDraws?.nutFlushDraw).toBe(false);
    });

    it('picks the hero player by name for heroDraws', async () => {
      const main: PQLResult = { values: { p1: 0.7, p2: 0.3 }, columns: ['p1', 'p2'], trials: 100, mode: 'monte_carlo', seed: null };
      const runPql = runPqlWith({ main });
      const out = await executeBuildQuery(
        {
          game: 'holdem',
          players: [{ name: 'Hero', cards: 'KhQh' }, { name: 'Villain', cards: 'AsAd' }],
          board: 'Ah7h2c',
        },
        { runPql },
      );
      expect(out.heroDraws?.player).toBe('Hero');
    });

    it('does not attach heroDraws preflop (no board)', async () => {
      const runPql = runPqlWith({ main: { values: { p1: 0.5, p2: 0.5 }, columns: ['p1', 'p2'], trials: 100, mode: 'monte_carlo', seed: null } });
      const out = await executeBuildQuery(
        { game: 'holdem', players: [{ name: 'PLAYER_1', cards: 'KhQh' }, { name: 'PLAYER_2', cards: 'AsAd' }] },
        { runPql },
      );
      expect(out.heroDraws).toBeUndefined();
      // No draws probe was issued (no query containing flushDraw).
      expect(runPql.mock.calls.some((c) => String(c[0]).includes('flushDraw'))).toBe(false);
    });

    it('does not attach heroDraws when the hero is a range', async () => {
      const runPql = runPqlWith({ main: { values: { p1: 0.5, p2: 0.5 }, columns: ['p1', 'p2'], trials: 100, mode: 'monte_carlo', seed: null } });
      const out = await executeBuildQuery(
        { game: 'holdem', players: [{ name: 'PLAYER_1', cards: 'QQ+' }, { name: 'PLAYER_2', cards: '25%' }], board: 'Ah7h2c' },
        { runPql },
      );
      expect(out.heroDraws).toBeUndefined();
    });

    it('best-effort: a failing draws probe still returns the main result', async () => {
      const main: PQLResult = { values: { p1: 0.7, p2: 0.3 }, columns: ['p1', 'p2'], trials: 100, mode: 'monte_carlo', seed: null };
      const runPql = vi.fn(async (q: string) => {
        if (q.includes('flushDraw')) throw new Error('Engine error: boom');
        return main;
      });
      const out = await executeBuildQuery(
        {
          game: 'holdem',
          players: [{ name: 'PLAYER_1', cards: 'KhQh' }, { name: 'PLAYER_2', cards: 'AsAd' }],
          board: 'Ah7h2c',
        },
        { runPql },
      );
      expect(out.heroDraws).toBeUndefined();
      expect(out.result).toBe(main);
      expect(out.viz.kind).toBe('equity');
    });
  });
});
