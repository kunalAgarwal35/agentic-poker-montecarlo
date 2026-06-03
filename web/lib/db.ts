import { neon } from '@neondatabase/serverless';

// Support either env name: Vercel's native Postgres injects POSTGRES_URL,
// the Neon marketplace integration injects DATABASE_URL.
const url = process.env.DATABASE_URL ?? process.env.POSTGRES_URL;

export function dbConfigured(): boolean {
  return !!url;
}

// Lazily-created neon HTTP client (a tagged-template function). null when no
// connection string is configured so routes can degrade gracefully.
const sql = url ? neon(url) : null;

type SqlClient = NonNullable<typeof sql>;

// Internal accessor that throws a clear error if used without a configured DB.
// Exposed (non-public-API) so the data layer is unit-testable: tests can
// vi.spyOn this module's getSql.
export function getSql(): SqlClient {
  if (!sql) {
    throw new Error(
      'DATABASE_URL/POSTGRES_URL is not set — the database is not configured.',
    );
  }
  return sql;
}

let schemaReady = false;

export async function ensureSchema(): Promise<void> {
  if (schemaReady) return;
  const db = getSql();
  // The neon HTTP driver runs ONE statement per call — issue them sequentially.
  await db`
    CREATE TABLE IF NOT EXISTS recent_queries (
      id TEXT PRIMARY KEY,
      ts TIMESTAMPTZ NOT NULL DEFAULT now(),
      question TEXT NOT NULL,
      kind TEXT NOT NULL,
      summary TEXT,
      payload JSONB NOT NULL
    )
  `;
  await db`CREATE INDEX IF NOT EXISTS recent_queries_ts_idx ON recent_queries (ts DESC)`;
  await db`
    CREATE TABLE IF NOT EXISTS feedback (
      id TEXT PRIMARY KEY,
      ts TIMESTAMPTZ NOT NULL DEFAULT now(),
      session_id TEXT,
      question TEXT,
      answer_summary TEXT,
      rating TEXT NOT NULL,
      reason TEXT,
      comment TEXT,
      meta JSONB
    )
  `;
  await db`CREATE INDEX IF NOT EXISTS feedback_ts_idx ON feedback (ts DESC)`;
  // Rate-limit / cost-abuse ledger: one row per allowed request. Counts in a
  // time window drive checkRateLimit() in lib/ratelimit.ts.
  await db`
    CREATE TABLE IF NOT EXISTS rate_events (
      id BIGSERIAL PRIMARY KEY,
      ip TEXT,
      route TEXT,
      ts TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `;
  await db`CREATE INDEX IF NOT EXISTS rate_events_ts_idx ON rate_events (ts)`;
  schemaReady = true;
}

export type RecentItem = {
  id: string;
  ts?: string;
  question: string;
  kind: string;
  summary?: string | null;
  payload: any;
};

function toIso(ts: any): string | undefined {
  if (ts == null) return undefined;
  if (ts instanceof Date) return ts.toISOString();
  return new Date(ts).toISOString();
}

export async function listRecent(limit = 30): Promise<RecentItem[]> {
  const db = getSql();
  const rows = (await db`
    SELECT id, ts, question, kind, summary, payload
    FROM recent_queries
    ORDER BY ts DESC
    LIMIT ${limit}
  `) as any[];
  return rows.map((r) => ({
    id: r.id,
    ts: toIso(r.ts),
    question: r.question,
    kind: r.kind,
    summary: r.summary ?? null,
    payload: r.payload ?? {},
  }));
}

export async function addRecent(item: {
  id?: string;
  question: string;
  kind: string;
  summary?: string;
  vizSpecs?: any;
  raw?: any;
}): Promise<RecentItem> {
  const db = getSql();
  const id = item.id ?? crypto.randomUUID();
  const payload = item.vizSpecs
    ? { vizSpecs: item.vizSpecs }
    : item.raw
      ? { raw: item.raw }
      : {};
  const summary = item.summary ?? null;

  await db`
    INSERT INTO recent_queries (id, question, kind, summary, payload)
    VALUES (${id}, ${item.question}, ${item.kind}, ${summary}, ${JSON.stringify(payload)})
    ON CONFLICT (id) DO NOTHING
  `;
  // Cap the table: keep only the 60 newest rows.
  await db`
    DELETE FROM recent_queries
    WHERE id NOT IN (
      SELECT id FROM recent_queries ORDER BY ts DESC LIMIT 60
    )
  `;

  return { id, question: item.question, kind: item.kind, summary, payload };
}

export type FeedbackEntry = {
  id?: string;
  sessionId?: string;
  question?: string;
  answerSummary?: string;
  rating: 'up' | 'down';
  reason?: string;
  comment?: string;
  meta?: any;
};

export async function addFeedback(e: FeedbackEntry): Promise<{ id: string }> {
  if (e.rating !== 'up' && e.rating !== 'down') {
    throw new Error(`Invalid rating: ${e.rating} (expected 'up' or 'down')`);
  }
  const db = getSql();
  const id = e.id ?? crypto.randomUUID();
  await db`
    INSERT INTO feedback (id, session_id, question, answer_summary, rating, reason, comment, meta)
    VALUES (
      ${id},
      ${e.sessionId ?? null},
      ${e.question ?? null},
      ${e.answerSummary ?? null},
      ${e.rating},
      ${e.reason ?? null},
      ${e.comment ?? null},
      ${e.meta != null ? JSON.stringify(e.meta) : null}
    )
  `;
  return { id };
}
