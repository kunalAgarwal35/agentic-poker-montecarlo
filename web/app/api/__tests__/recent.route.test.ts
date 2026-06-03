import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock the neon client so DB-configured paths don't hit a real database.
let nextRows: any[] = [];
vi.mock('@neondatabase/serverless', () => ({
  neon: () => (_strings: TemplateStringsArray, ..._values: any[]) =>
    Promise.resolve(nextRows),
}));

// db.ts reads the connection string at module load, so reset + re-import per
// test after toggling the env.
async function loadRoute() {
  return import('@/app/api/recent/route');
}

describe('/api/recent route', () => {
  beforeEach(() => {
    vi.resetModules();
    nextRows = [];
  });
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('GET with no DB env returns the seed examples', async () => {
    vi.stubEnv('DATABASE_URL', '');
    vi.stubEnv('POSTGRES_URL', '');
    const { GET } = await loadRoute();
    const res = await GET();
    const body = await res.json();
    const { SEED_EXAMPLES } = await import('@/lib/seedExamples');
    expect(body.items).toHaveLength(SEED_EXAMPLES.length);
    expect(body.items[0].id).toBe(SEED_EXAMPLES[0].id);
  });

  it('GET with DB but empty table falls back to seed', async () => {
    vi.stubEnv('DATABASE_URL', 'postgres://test');
    nextRows = [];
    const { GET } = await loadRoute();
    const res = await GET();
    const body = await res.json();
    const { SEED_EXAMPLES } = await import('@/lib/seedExamples');
    expect(body.items).toHaveLength(SEED_EXAMPLES.length);
  });

  it('GET with DB rows returns those rows', async () => {
    vi.stubEnv('DATABASE_URL', 'postgres://test');
    nextRows = [
      {
        id: 'row1',
        ts: '2026-06-02T00:00:00.000Z',
        question: 'q',
        kind: 'raw',
        summary: null,
        payload: {},
      },
    ];
    const { GET } = await loadRoute();
    const res = await GET();
    const body = await res.json();
    expect(body.items).toHaveLength(1);
    expect(body.items[0].id).toBe('row1');
  });

  it('POST without question returns 400', async () => {
    vi.stubEnv('DATABASE_URL', 'postgres://test');
    const { POST } = await loadRoute();
    const res = await POST(
      new Request('http://test/api/recent', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ kind: 'raw' }),
      }),
    );
    expect(res.status).toBe(400);
  });

  it('POST with no DB configured is a no-op', async () => {
    vi.stubEnv('DATABASE_URL', '');
    vi.stubEnv('POSTGRES_URL', '');
    const { POST } = await loadRoute();
    const res = await POST(
      new Request('http://test/api/recent', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ question: 'q', kind: 'raw' }),
      }),
    );
    const body = await res.json();
    expect(body).toEqual({ ok: false, skipped: true });
  });

  it('POST with a question and DB mocked returns the stored item', async () => {
    vi.stubEnv('DATABASE_URL', 'postgres://test');
    const { POST } = await loadRoute();
    const res = await POST(
      new Request('http://test/api/recent', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          question: 'AsKs vs QQ',
          kind: 'equity',
          vizSpecs: [{ kind: 'equity', rows: [] }],
        }),
      }),
    );
    const body = await res.json();
    expect(body.question).toBe('AsKs vs QQ');
    expect(body.kind).toBe('equity');
    expect(body.payload).toEqual({ vizSpecs: [{ kind: 'equity', rows: [] }] });
    expect(body.id).toBeTruthy();
  });
});
