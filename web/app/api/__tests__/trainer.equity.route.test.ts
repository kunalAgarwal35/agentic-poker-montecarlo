// @vitest-environment node
// (happy-dom strips the forbidden `Origin` request header, which CORS needs.)
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { buildEquityQuery, parseEquityRequest } from '@/lib/trainerEquity';

const runPql = vi.fn();
vi.mock('@/lib/engine', () => ({ runPql: (...a: unknown[]) => runPql(...a) }));

async function loadRoute() {
  return import('@/app/api/trainer/equity/route');
}

function post(body: unknown, origin = 'https://train.magicflop.com') {
  return new Request('http://test/api/trainer/equity', {
    method: 'POST',
    headers: { 'content-type': 'application/json', origin },
    body: JSON.stringify(body),
  });
}

describe('trainer equity helpers', () => {
  it('infers the game from hole-card count and builds a multi-player query', () => {
    const p = parseEquityRequest({ hands: [['Ah', 'Kh', 'Qh', 'Jh', 'Td'], 'AsKsQsJs9c'], board: ['2h', '3d', '9h'] });
    expect(p.ok).toBe(true);
    if (!p.ok) return;
    expect(p.game).toBe('omahahi5');
    expect(buildEquityQuery(p.game, p.hands, p.board, p.dead)).toBe(
      "select avg(riverEquity(PLAYER_1)) as p1, avg(riverEquity(PLAYER_2)) as p2 from game='omahahi5', " +
        "board='2h3d9h', PLAYER_1='AhKhQhJhTd', PLAYER_2='AsKsQsJs9c'",
    );
  });

  it.each([
    [{ hands: [['Ah', 'Kh', 'Qh', 'Jh']] }, 'between 2'],
    [{ hands: [['Ah', 'Kh', 'Qh', 'Jh'], ['As', 'Ks', 'Qs']] }, 'same number'],
    [{ hands: [['Ah', 'Kh', 'Qh'], ['As', 'Ks', 'Qs']] }, 'unsupported hand size'],
    [{ hands: [['Ah', 'Kh', 'Qh', 'Jh'], ['Ah', 'Ks', 'Qs', 'Js']] }, 'duplicate'],
    [{ hands: [['Ah', 'Kh', 'Qh', 'Xx'], ['As', 'Ks', 'Qs', 'Js']] }, 'invalid hand'],
    [{ hands: [['Ah', 'Kh', 'Qh', 'Jh'], ['As', 'Ks', 'Qs', 'Js']], board: ['2c'] }, 'board must'],
    [{ hands: [["Ah'", 'Kh', 'Qh', 'Jh'], ['As', 'Ks', 'Qs', 'Js']] }, 'invalid hand'],
  ])('rejects %j', (body, msg) => {
    const p = parseEquityRequest(body);
    expect(p.ok).toBe(false);
    if (!p.ok) expect(p.error).toContain(msg);
  });
});

describe('/api/trainer/equity route', () => {
  beforeEach(() => {
    vi.resetModules();
    runPql.mockReset();
  });
  afterEach(() => vi.unstubAllEnvs());

  it('returns percent equities and CORS for the trainer origin', async () => {
    runPql.mockResolvedValue({ values: { p1: 0.61234, p2: 0.38766 }, columns: ['p1', 'p2'], trials: 30000, mode: 'monte_carlo', seed: null });
    const { POST } = await loadRoute();
    const res = await POST(post({ hands: [['Ah', 'Kh', 'Qh', 'Jh'], ['As', 'Ks', '2d', '3c']], board: [] }));
    expect(res.status).toBe(200);
    expect(res.headers.get('access-control-allow-origin')).toBe('https://train.magicflop.com');
    const body = await res.json();
    expect(body.hero).toBe(61.23);
    expect(body.equities).toEqual([61.23, 38.77]);
    expect(body.is_fallback).toBe(false);
    expect(runPql.mock.calls[0][0]).toContain("game='omahahi'");
  });

  it('400s on bad input without calling the engine', async () => {
    const { POST } = await loadRoute();
    const res = await POST(post({ hands: 'nope' }));
    expect(res.status).toBe(400);
    expect(runPql).not.toHaveBeenCalled();
  });

  it('503s when the engine is unreachable', async () => {
    runPql.mockRejectedValue(new Error('Engine unreachable'));
    const { POST } = await loadRoute();
    const res = await POST(post({ hands: [['Ah', 'Kh', 'Qh', 'Jh'], ['As', 'Ks', 'Qs', 'Js']] }));
    expect(res.status).toBe(503);
  });

  it('omits CORS allow-origin for unknown origins', async () => {
    const { OPTIONS } = await loadRoute();
    const req = new Request('http://test/api/trainer/equity', { method: 'OPTIONS', headers: { origin: 'https://evil.example' } });
    const res = await OPTIONS(req);
    expect(res.headers.get('access-control-allow-origin')).toBeNull();
  });
});
