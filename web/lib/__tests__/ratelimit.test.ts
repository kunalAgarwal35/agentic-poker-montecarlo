import { describe, it, expect, vi, beforeEach } from 'vitest';

// We mock the whole db layer so checkRateLimit's behaviour is driven entirely by
// what dbConfigured() reports and what the sql tagged-template returns.
const sqlMock = vi.fn();
const dbConfiguredMock = vi.fn();
const ensureSchemaMock = vi.fn(async () => {});

vi.mock('@/lib/db', () => ({
  dbConfigured: () => dbConfiguredMock(),
  ensureSchema: () => ensureSchemaMock(),
  getSql: () => sqlMock,
}));

import { checkRateLimit, clientIp } from '@/lib/ratelimit';

// The first sql`...` call in checkRateLimit is the GLOBAL count, the second is
// the per-IP count, the third is the INSERT. Queue results in that order.
function queueSqlResults(results: any[]) {
  let i = 0;
  sqlMock.mockImplementation(() => {
    const r = results[i] ?? [];
    i++;
    return Promise.resolve(r);
  });
}

describe('checkRateLimit', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    dbConfiguredMock.mockReturnValue(true);
    // Avoid the random cleanup branch firing nondeterministically.
    vi.spyOn(Math, 'random').mockReturnValue(0.99);
  });

  it('returns {ok:true} and inserts when under both limits', async () => {
    queueSqlResults([
      [{ n: 5 }], // global day count
      [{ n: 1 }], // per-IP hour count
      [], // INSERT
    ]);
    const res = await checkRateLimit('1.2.3.4', 'chat');
    expect(res.ok).toBe(true);
    // 3 calls: global count, ip count, insert.
    expect(sqlMock).toHaveBeenCalledTimes(3);
    // Verify the third call (INSERT) carries the ip + route as parameters.
    const insertCall = sqlMock.mock.calls[2];
    expect(insertCall[1]).toBe('1.2.3.4'); // first interpolated value = ip
    expect(insertCall[2]).toBe('chat'); // second = route
  });

  it('returns 503 (cost kill-switch) when the global day-count is at the cap', async () => {
    queueSqlResults([
      [{ n: 300 }], // global day count >= default 300
    ]);
    const res = await checkRateLimit('1.2.3.4', 'chat');
    expect(res.ok).toBe(false);
    expect(res.status).toBe(503);
    expect(res.message).toMatch(/usage cap/i);
    // Must short-circuit before the per-IP query or INSERT.
    expect(sqlMock).toHaveBeenCalledTimes(1);
  });

  it('returns 429 when the per-IP hour-count is at the cap', async () => {
    queueSqlResults([
      [{ n: 10 }], // global under cap
      [{ n: 20 }], // per-IP hour count >= default 20
    ]);
    const res = await checkRateLimit('1.2.3.4', 'chat');
    expect(res.ok).toBe(false);
    expect(res.status).toBe(429);
    expect(res.message).toMatch(/too quickly/i);
    // global + ip queries, but no INSERT.
    expect(sqlMock).toHaveBeenCalledTimes(2);
  });

  it('returns {ok:true} without touching the DB when not configured', async () => {
    dbConfiguredMock.mockReturnValue(false);
    const res = await checkRateLimit('1.2.3.4', 'chat');
    expect(res.ok).toBe(true);
    expect(sqlMock).not.toHaveBeenCalled();
    expect(ensureSchemaMock).not.toHaveBeenCalled();
  });

  it('FAILS OPEN ({ok:true}) when a DB call throws', async () => {
    sqlMock.mockImplementation(() => {
      throw new Error('connection reset');
    });
    const res = await checkRateLimit('1.2.3.4', 'chat');
    expect(res.ok).toBe(true);
  });
});

describe('clientIp', () => {
  it('reads the first IP from x-forwarded-for', () => {
    const req = new Request('http://t', {
      headers: { 'x-forwarded-for': '9.9.9.9, 10.0.0.1' },
    });
    expect(clientIp(req)).toBe('9.9.9.9');
  });

  it('falls back to x-real-ip', () => {
    const req = new Request('http://t', { headers: { 'x-real-ip': '8.8.8.8' } });
    expect(clientIp(req)).toBe('8.8.8.8');
  });

  it("returns 'unknown' when no IP headers are present", () => {
    const req = new Request('http://t');
    expect(clientIp(req)).toBe('unknown');
  });
});
