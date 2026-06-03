import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('@/lib/engine', () => ({
  runPql: vi.fn(async (query: string) => ({
    values: { e: 0.55 },
    columns: ['e', 'h'],
    trials: 1000,
    mode: 'monte_carlo',
    seed: null,
    histograms: { h: { pairs: [{ value: 1, count: 600 }, { value: 2, count: 400 }], labels: { '1': 'Pair', '2': 'Two pair' } } },
  })),
}));

import { runPqlTool } from '@/lib/runPqlTool';

describe('run_pql tool', () => {
  beforeEach(() => vi.clearAllMocks());
  it('returns the query and the engine result incl. histograms', async () => {
    const out = await runPqlTool({ query: "select avg(riverEquity(PLAYER_1)) as e from game='holdem'" });
    expect(out.query).toContain('riverEquity');
    expect(out.result.values.e).toBe(0.55);
    expect(out.result.histograms?.h.pairs).toHaveLength(2);
  });
});
