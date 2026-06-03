import { describe, it, expect, vi, beforeEach, beforeAll } from 'vitest';
import type {
  dbConfigured as DbConfigured,
  listRecent as ListRecent,
  addRecent as AddRecent,
  addFeedback as AddFeedback,
  RecentItem,
} from '@/lib/db';

// Shared mock state for the neon tagged-template client.
const calls: { strings: string[]; values: any[] }[] = [];
let nextRows: any[] = [];

// The neon client is itself a tagged-template function:
//   sql`SELECT ...${v}...`  ->  fn(stringsArray, ...values)
vi.mock('@neondatabase/serverless', () => ({
  neon: () => {
    const fn = (strings: TemplateStringsArray, ...values: any[]) => {
      calls.push({ strings: Array.from(strings), values });
      return Promise.resolve(nextRows);
    };
    return fn;
  },
}));

// db.ts reads the connection string at module-load time. Static ESM imports are
// hoisted above `vi.stubEnv`, so import the module dynamically AFTER the env is
// set to guarantee its `sql` client is created.
let dbConfigured: typeof DbConfigured;
let listRecent: typeof ListRecent;
let addRecent: typeof AddRecent;
let addFeedback: typeof AddFeedback;

beforeAll(async () => {
  vi.stubEnv('DATABASE_URL', 'postgres://test');
  vi.resetModules();
  const mod = await import('@/lib/db');
  dbConfigured = mod.dbConfigured;
  listRecent = mod.listRecent;
  addRecent = mod.addRecent;
  addFeedback = mod.addFeedback;
});

const lastCall = () => calls[calls.length - 1];

describe('db data layer', () => {
  beforeEach(() => {
    calls.length = 0;
    nextRows = [];
  });

  it('dbConfigured reflects the env connection string', () => {
    expect(dbConfigured()).toBe(true);
  });

  it('listRecent maps rows and converts ts to ISO', async () => {
    nextRows = [
      {
        id: 'a',
        ts: '2026-06-01T10:00:00.000Z',
        question: 'q1',
        kind: 'equity',
        summary: 's1',
        payload: { vizSpecs: [{ kind: 'equity' }] },
      },
    ];
    const items: RecentItem[] = await listRecent();
    expect(items).toHaveLength(1);
    expect(items[0].id).toBe('a');
    expect(items[0].ts).toBe('2026-06-01T10:00:00.000Z');
    expect(items[0].payload.vizSpecs[0].kind).toBe('equity');
    // limit is parameterized, not interpolated into the SQL text.
    const sqlText = lastCall().strings.join('?');
    expect(sqlText).toContain('FROM recent_queries');
    expect(lastCall().values).toContain(30);
  });

  it('addRecent generates an id and builds payload from vizSpecs', async () => {
    const item = await addRecent({
      question: 'AsKs vs QQ',
      kind: 'equity',
      vizSpecs: [{ kind: 'equity', rows: [] }],
    });
    expect(item.id).toBeTruthy();
    expect(item.payload).toEqual({ vizSpecs: [{ kind: 'equity', rows: [] }] });
    // First call is the INSERT — id + payload are passed as parameters.
    const insert = calls[0];
    expect(insert.values).toContain(item.id);
    expect(insert.values).toContain('AsKs vs QQ');
    expect(insert.values).toContain(
      JSON.stringify({ vizSpecs: [{ kind: 'equity', rows: [] }] }),
    );
    // Second call caps the table.
    expect(calls[1].strings.join(' ')).toContain('DELETE FROM recent_queries');
  });

  it('addRecent builds payload from raw when no vizSpecs', async () => {
    const raw = { query: 'q', result: { values: { eq: 0.99 } } };
    const item = await addRecent({ question: 'flush', kind: 'raw', raw });
    expect(item.payload).toEqual({ raw });
  });

  it('addRecent honors a provided id', async () => {
    const item = await addRecent({ id: 'fixed', question: 'q', kind: 'raw' });
    expect(item.id).toBe('fixed');
    expect(item.payload).toEqual({});
  });

  it('addFeedback inserts and returns an id', async () => {
    const { id } = await addFeedback({ rating: 'up', comment: 'nice' });
    expect(id).toBeTruthy();
    expect(calls[0].strings.join(' ')).toContain('INSERT INTO feedback');
    expect(calls[0].values).toContain('up');
    expect(calls[0].values).toContain('nice');
  });

  it('addFeedback rejects an invalid rating without touching the DB', async () => {
    await expect(
      addFeedback({ rating: 'sideways' as any }),
    ).rejects.toThrow(/invalid rating/i);
    expect(calls).toHaveLength(0);
  });
});
