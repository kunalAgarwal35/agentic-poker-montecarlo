-- Neon Postgres schema for the Next app (recent gallery + user feedback).
-- Idempotent; matches ensureSchema() in web/lib/db.ts. Safe to run manually
-- against DATABASE_URL / POSTGRES_URL for a one-time migration.

CREATE TABLE IF NOT EXISTS recent_queries (
  id TEXT PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  question TEXT NOT NULL,
  kind TEXT NOT NULL,
  summary TEXT,
  payload JSONB NOT NULL          -- { vizSpecs?: VizSpec[] } | { raw?: RawResult }
);
CREATE INDEX IF NOT EXISTS recent_queries_ts_idx ON recent_queries (ts DESC);

CREATE TABLE IF NOT EXISTS feedback (
  id TEXT PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  session_id TEXT,
  question TEXT,
  answer_summary TEXT,
  rating TEXT NOT NULL,           -- 'up' | 'down'
  reason TEXT,                    -- reason code, when rating='down'
  comment TEXT,                   -- free-text ('verbal') feedback
  meta JSONB                      -- { tool?, vizKind?, url? }
);
CREATE INDEX IF NOT EXISTS feedback_ts_idx ON feedback (ts DESC);
