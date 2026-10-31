// @vitest-environment node
// (happy-dom strips the forbidden `Origin` request header, which CORS needs.)
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { parseVote } from '@/lib/trainerFeedback';

// Every SQL call is recorded; SELECTs return `nextRows`.
const calls: string[] = [];
let nextRows: unknown[] = [];
vi.mock('@neondatabase/serverless', () => ({
  neon: () => (strings: TemplateStringsArray) => {
    calls.push(strings.join('?'));
    return Promise.resolve(strings.join('').includes('SELECT') ? nextRows : []);
  },
}));

const good = { questionId: 'plo4-rfi-BU-Js8h3h-flush_draw', template: 'flush_draw', clientId: 'abcd1234-ef', vote: 'up' };

function req(method: string, body?: unknown) {
  return new Request('http://test/api/trainer/feedback', {
    method,
    headers: { 'content-type': 'application/json', origin: 'https://train.magicflop.com' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

describe('parseVote', () => {
  it('accepts a well-formed vote', () => {
    expect(parseVote(good)).toEqual(good);
  });
  it.each([
    [{ ...good, vote: 'meh' }, 'vote'],
    [{ ...good, questionId: "x'; drop table" }, 'questionId'],
    [{ ...good, template: 'Flush Draw' }, 'template'],
    [{ ...good, clientId: 'short' }, 'clientId'],
    [null, 'questionId'],
  ])('rejects %j', (body, field) => {
    const r = parseVote(body);
    expect('error' in r && r.error).toContain(field);
  });
});

describe('/api/trainer/feedback', () => {
  beforeEach(() => {
    vi.resetModules();
    calls.length = 0;
    nextRows = [];
  });
  afterEach(() => vi.unstubAllEnvs());

  it('GET without a DB returns empty stats', async () => {
    vi.stubEnv('DATABASE_URL', '');
    vi.stubEnv('POSTGRES_URL', '');
    const { GET } = await import('@/app/api/trainer/feedback/route');
    const res = await GET(req('GET'));
    expect(await res.json()).toEqual({ questions: {}, templates: {} });
    expect(res.headers.get('access-control-allow-origin')).toBe('https://train.magicflop.com');
  });

  it('GET aggregates per question and per template', async () => {
    vi.stubEnv('DATABASE_URL', 'postgres://test');
    nextRows = [
      { question_id: 'q1', template: 'set', up: 3, down: 1 },
      { question_id: 'q2', template: 'set', up: 1, down: 2 },
      { question_id: 'q3', template: 'nuts', up: 0, down: 4 },
    ];
    const { GET } = await import('@/app/api/trainer/feedback/route');
    const body = await (await GET(req('GET'))).json();
    expect(body.questions.q1).toEqual({ up: 3, down: 1 });
    expect(body.templates).toEqual({ set: { up: 4, down: 3 }, nuts: { up: 0, down: 4 } });
  });

  it('POST upserts one vote per question+client', async () => {
    vi.stubEnv('DATABASE_URL', 'postgres://test');
    const { POST } = await import('@/app/api/trainer/feedback/route');
    const res = await POST(req('POST', good));
    expect(await res.json()).toEqual({ ok: true });
    expect(calls.some((c) => c.includes('ON CONFLICT (question_id, client_id)'))).toBe(true);
  });

  it('POST rejects bad input with 400 and no SQL', async () => {
    vi.stubEnv('DATABASE_URL', 'postgres://test');
    const { POST } = await import('@/app/api/trainer/feedback/route');
    const res = await POST(req('POST', { ...good, vote: 'sideways' }));
    expect(res.status).toBe(400);
    expect(calls).toHaveLength(0);
  });
});
