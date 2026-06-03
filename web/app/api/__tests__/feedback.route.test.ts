import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

vi.mock('@neondatabase/serverless', () => ({
  neon: () => (_strings: TemplateStringsArray, ..._values: any[]) =>
    Promise.resolve([]),
}));

async function loadRoute() {
  return import('@/app/api/feedback/route');
}

function post(body: any) {
  return new Request('http://test/api/feedback', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
}

describe('/api/feedback route', () => {
  beforeEach(() => vi.resetModules());
  afterEach(() => vi.unstubAllEnvs());

  it('POST with an invalid rating returns 400', async () => {
    vi.stubEnv('DATABASE_URL', 'postgres://test');
    const { POST } = await loadRoute();
    const res = await POST(post({ rating: 'meh' }));
    expect(res.status).toBe(400);
  });

  it('POST with no DB configured is a no-op', async () => {
    vi.stubEnv('DATABASE_URL', '');
    vi.stubEnv('POSTGRES_URL', '');
    const { POST } = await loadRoute();
    const res = await POST(post({ rating: 'up' }));
    const body = await res.json();
    expect(body).toEqual({ ok: false, skipped: true });
  });

  it('POST with a valid rating and DB mocked returns {ok:true, id}', async () => {
    vi.stubEnv('DATABASE_URL', 'postgres://test');
    const { POST } = await loadRoute();
    const res = await POST(
      post({ rating: 'down', reason: 'wrong_numbers', comment: 'off' }),
    );
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.id).toBeTruthy();
  });
});
