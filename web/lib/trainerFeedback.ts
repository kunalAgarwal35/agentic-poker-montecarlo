import { getSql } from '@/lib/db';

// Thumbs up/down on PLO Trainer questions. One vote per (question, client):
// a re-vote replaces the earlier one, so a single browser can't stuff a
// question's score. The trainer reads the aggregates to show well-liked
// questions (and question types) more often.

export type Vote = 'up' | 'down';
export type Tally = { up: number; down: number };
export type FeedbackStats = {
  questions: Record<string, Tally>;
  templates: Record<string, Tally>;
};

const ID_RE = /^[A-Za-z0-9_-]{1,120}$/;
const TEMPLATE_RE = /^[a-z0-9_]{1,40}$/;
const CLIENT_RE = /^[A-Za-z0-9-]{8,64}$/;

export type VoteInput = { questionId: string; template: string; clientId: string; vote: Vote };

export function parseVote(raw: unknown): VoteInput | { error: string } {
  const b = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>;
  const { questionId, template, clientId, vote } = b;
  if (typeof questionId !== 'string' || !ID_RE.test(questionId)) return { error: 'invalid questionId' };
  if (typeof template !== 'string' || !TEMPLATE_RE.test(template)) return { error: 'invalid template' };
  if (typeof clientId !== 'string' || !CLIENT_RE.test(clientId)) return { error: 'invalid clientId' };
  if (vote !== 'up' && vote !== 'down') return { error: "vote must be 'up' or 'down'" };
  return { questionId, template, clientId, vote };
}

let ready = false;

export async function ensureTrainerSchema(): Promise<void> {
  if (ready) return;
  const db = getSql();
  await db`
    CREATE TABLE IF NOT EXISTS trainer_feedback (
      question_id TEXT NOT NULL,
      template TEXT NOT NULL,
      client_id TEXT NOT NULL,
      vote TEXT NOT NULL,
      ts TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (question_id, client_id)
    )
  `;
  await db`CREATE INDEX IF NOT EXISTS trainer_feedback_template_idx ON trainer_feedback (template)`;
  ready = true;
}

export async function recordVote(v: VoteInput): Promise<void> {
  const db = getSql();
  await db`
    INSERT INTO trainer_feedback (question_id, template, client_id, vote)
    VALUES (${v.questionId}, ${v.template}, ${v.clientId}, ${v.vote})
    ON CONFLICT (question_id, client_id)
    DO UPDATE SET vote = EXCLUDED.vote, template = EXCLUDED.template, ts = now()
  `;
}

type Row = { question_id: string; template: string; up: number; down: number };

export async function getStats(): Promise<FeedbackStats> {
  const db = getSql();
  const rows = (await db`
    SELECT question_id, template,
           count(*) FILTER (WHERE vote = 'up')::int AS up,
           count(*) FILTER (WHERE vote = 'down')::int AS down
    FROM trainer_feedback
    GROUP BY question_id, template
  `) as Row[];
  const stats: FeedbackStats = { questions: {}, templates: {} };
  for (const r of rows) {
    stats.questions[r.question_id] = { up: r.up, down: r.down };
    const t = (stats.templates[r.template] ??= { up: 0, down: 0 });
    t.up += r.up;
    t.down += r.down;
  }
  return stats;
}
