import { describe, it, expect, vi, afterEach } from 'vitest';
import { runPql } from '@/lib/engine';

afterEach(() => vi.restoreAllMocks());

describe('runPql', () => {
  it('POSTs the query to ENGINE_URL/pql and returns the parsed result', async () => {
    const body = { values: { p1: 0.6, p2: 0.4 }, columns: ['p1', 'p2'], trials: 5000, mode: 'monte_carlo', seed: 1 };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => body });
    vi.stubGlobal('fetch', fetchMock);

    const res = await runPql("select avg(riverEquity(PLAYER_1)) as p1 from game='omahahi5', PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'", { trials: 5000, seed: 1 });

    expect(res.mode).toBe('monte_carlo');
    expect(res.values.p1).toBe(0.6);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/pql$/);
    expect(JSON.parse(init.body)).toMatchObject({ trials: 5000, seed: 1 });
  });

  it('throws a useful error when the engine returns a non-ok status', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 400, json: async () => ({ error: 'Bad query', details: 'ranges not supported' }) });
    vi.stubGlobal('fetch', fetchMock);
    await expect(runPql('bad')).rejects.toThrow(/ranges not supported/);
  });
});
