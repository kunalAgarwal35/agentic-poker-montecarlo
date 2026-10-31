import { getSql } from '@/lib/db';
import { bank, GAMES, MODES, score, type Game, type Mode } from '@/lib/trainerBank';

// PLO Trainer challenges: one shared set of questions behind a short code,
// played by anyone with the link, ranked on a small friends' leaderboard.

export const CODE_ALPHABET = '23456789ABCDEFGHJKMNPQRSTUVWXYZ'; // no 0/O, 1/I/L
export const CODE_RE = /^[23456789ABCDEFGHJKMNPQRSTUVWXYZ]{6}$/;
export const LENGTHS = [5, 10] as const;
export const CHALLENGE_DAYS = 7;
const PLAYER_RE = /^[A-Za-z0-9-]{8,64}$/;

export function newCode(rand: () => number = Math.random): string {
  let s = '';
  for (let i = 0; i < 6; i++) s += CODE_ALPHABET[Math.floor(rand() * CODE_ALPHABET.length)];
  return s;
}

/** Display names: trimmed, collapsed whitespace, 1..20 printable chars. */
export function cleanName(raw: unknown): string | null {
  if (typeof raw !== 'string') return null;
  const s = raw.replace(/[\u0000-\u001f\u007f<>]/g, '').replace(/\s+/g, ' ').trim().slice(0, 20);
  return s.length ? s : null;
}

export type CreateInput = { mode: Mode | 'mixed'; game: Game | 'mixed'; length: number; name: string; playerId: string };

export function parseCreate(raw: unknown): CreateInput | { error: string } {
  const b = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>;
  const mode = b.mode as Mode | 'mixed';
  const game = b.game as Game | 'mixed';
  if (mode !== 'mixed' && !MODES.includes(mode)) return { error: 'invalid mode' };
  if (game !== 'mixed' && !GAMES.includes(game)) return { error: 'invalid game' };
  if (!LENGTHS.includes(b.length as 5 | 10)) return { error: 'length must be 5 or 10' };
  const name = cleanName(b.name);
  if (!name) return { error: 'name is required' };
  if (typeof b.playerId !== 'string' || !PLAYER_RE.test(b.playerId)) return { error: 'invalid playerId' };
  return { mode, game, length: b.length as number, name, playerId: b.playerId };
}

export type AnswerInput = { playerId: string; name: string; index: number; guess: number; ms: number };

export function parseAnswer(raw: unknown): AnswerInput | { error: string } {
  const b = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>;
  if (typeof b.playerId !== 'string' || !PLAYER_RE.test(b.playerId)) return { error: 'invalid playerId' };
  const name = cleanName(b.name);
  if (!name) return { error: 'name is required' };
  if (!Number.isInteger(b.index) || (b.index as number) < 0 || (b.index as number) > 50) return { error: 'invalid index' };
  if (typeof b.guess !== 'number' || !Number.isFinite(b.guess) || b.guess < 0 || b.guess > 100) {
    return { error: 'guess must be a number from 0 to 100' };
  }
  const ms = typeof b.ms === 'number' && Number.isFinite(b.ms) ? Math.max(0, Math.min(Math.round(b.ms), 3_600_000)) : 0;
  return { playerId: b.playerId, name, index: b.index as number, guess: b.guess, ms };
}

export type Challenge = {
  code: string;
  mode: Mode | 'mixed';
  game: Game | 'mixed';
  question_ids: string[];
  creator_name: string;
  creator_id: string;
  created_at: string;
  expires_at: string;
};

export type AnswerRecord = { i: number; guess: number; points: number; ms: number };

export type Entry = {
  player_id: string;
  name: string;
  answers: AnswerRecord[];
  total: number;
  ms: number;
  finished: boolean;
  updated_at: string;
};

let ready = false;
export async function ensureChallengeSchema() {
  if (ready) return;
  const db = getSql();
  await db`
    CREATE TABLE IF NOT EXISTS trainer_challenges (
      code TEXT PRIMARY KEY,
      mode TEXT NOT NULL,
      game TEXT NOT NULL,
      question_ids JSONB NOT NULL,
      creator_name TEXT NOT NULL,
      creator_id TEXT NOT NULL,
      creator_ip TEXT,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      expires_at TIMESTAMPTZ NOT NULL
    )
  `;
  await db`CREATE INDEX IF NOT EXISTS trainer_challenges_ip_idx ON trainer_challenges (creator_ip, created_at)`;
  await db`
    CREATE TABLE IF NOT EXISTS trainer_challenge_entries (
      code TEXT NOT NULL REFERENCES trainer_challenges(code) ON DELETE CASCADE,
      player_id TEXT NOT NULL,
      name TEXT NOT NULL,
      answers JSONB NOT NULL DEFAULT '[]',
      total INT NOT NULL DEFAULT 0,
      ms BIGINT NOT NULL DEFAULT 0,
      finished BOOLEAN NOT NULL DEFAULT false,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (code, player_id)
    )
  `;
  ready = true;
}

const MAX_CREATES_PER_IP_HOUR = 30;

export async function createChallenge(input: CreateInput, ids: string[], ip: string): Promise<string | { error: string; status: number }> {
  const db = getSql();
  const recent = (await db`
    SELECT count(*)::int AS n FROM trainer_challenges
    WHERE creator_ip = ${ip} AND created_at > now() - interval '1 hour'
  `) as { n: number }[];
  if ((recent[0]?.n ?? 0) >= MAX_CREATES_PER_IP_HOUR) {
    return { error: 'Too many challenges created recently. Try again in a bit.', status: 429 };
  }
  for (let attempt = 0; attempt < 5; attempt++) {
    const code = newCode();
    const rows = (await db`
      INSERT INTO trainer_challenges (code, mode, game, question_ids, creator_name, creator_id, creator_ip, expires_at)
      VALUES (${code}, ${input.mode}, ${input.game}, ${JSON.stringify(ids)}, ${input.name}, ${input.playerId}, ${ip},
              now() + make_interval(days => ${CHALLENGE_DAYS}))
      ON CONFLICT (code) DO NOTHING
      RETURNING code
    `) as { code: string }[];
    if (rows.length) return rows[0].code;
  }
  return { error: 'Could not allocate a challenge code', status: 500 };
}

export async function getChallenge(code: string): Promise<Challenge | null> {
  const db = getSql();
  const rows = (await db`
    SELECT code, mode, game, question_ids, creator_name, creator_id, created_at, expires_at
    FROM trainer_challenges WHERE code = ${code}
  `) as Challenge[];
  return rows[0] ?? null;
}

export async function getEntries(code: string): Promise<Entry[]> {
  const db = getSql();
  return (await db`
    SELECT player_id, name, answers, total, ms, finished, updated_at
    FROM trainer_challenge_entries WHERE code = ${code}
  `) as Entry[];
}

/** Leaderboard order: finished first, then points desc, then time asc. */
export function rankEntries(entries: Entry[]): Entry[] {
  return [...entries].sort(
    (a, b) => Number(b.finished) - Number(a.finished) || b.total - a.total || a.ms - b.ms,
  );
}

/**
 * Record one answer. Answers must come in order (index == answers so far), so
 * a player can't skip, retry, or answer after seeing a later question.
 * Returns the updated entry, or an error.
 */
export async function recordAnswer(
  c: Challenge,
  input: AnswerInput,
): Promise<{ entry: Entry; points: number } | { error: string; status: number }> {
  const db = getSql();
  const qid = c.question_ids[input.index];
  const q = qid ? bank().byId.get(qid) : undefined;
  if (!q) return { error: 'no such question', status: 400 };
  const points = score(q, q.kind === 'choice' ? Math.round(input.guess) : input.guess);
  const rec: AnswerRecord = { i: input.index, guess: input.guess, points, ms: input.ms };
  const finished = input.index === c.question_ids.length - 1;

  if (input.index === 0) {
    const rows = (await db`
      INSERT INTO trainer_challenge_entries (code, player_id, name, answers, total, ms, finished)
      VALUES (${c.code}, ${input.playerId}, ${input.name}, ${JSON.stringify([rec])}, ${points}, ${input.ms}, ${finished})
      ON CONFLICT (code, player_id) DO NOTHING
      RETURNING player_id, name, answers, total, ms, finished, updated_at
    `) as Entry[];
    if (!rows.length) return { error: 'already answered', status: 409 };
    return { entry: rows[0], points };
  }
  // Conditional append: only if exactly `index` answers exist (atomic guard).
  const rows = (await db`
    UPDATE trainer_challenge_entries
    SET answers = answers || ${JSON.stringify([rec])}::jsonb,
        total = total + ${points},
        ms = ms + ${input.ms},
        finished = ${finished},
        name = ${input.name},
        updated_at = now()
    WHERE code = ${c.code} AND player_id = ${input.playerId}
      AND jsonb_array_length(answers) = ${input.index}
    RETURNING player_id, name, answers, total, ms, finished, updated_at
  `) as Entry[];
  if (!rows.length) return { error: 'answers must be submitted in order', status: 409 };
  return { entry: rows[0], points };
}

/** Field stats for one question, from everyone else who answered it. */
export function fieldStats(entries: Entry[], index: number, exceptPlayer?: string) {
  const guesses = entries
    .filter((e) => e.player_id !== exceptPlayer)
    .map((e) => e.answers.find((a) => a.i === index))
    .filter((a): a is AnswerRecord => !!a);
  if (!guesses.length) return { count: 0 };
  const avg = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / xs.length;
  return {
    count: guesses.length,
    avgGuess: Math.round(avg(guesses.map((g) => g.guess)) * 10) / 10,
    avgPoints: Math.round(avg(guesses.map((g) => g.points))),
    // choice questions: how many picked each option
    picks: [0, 1].map((k) => guesses.filter((g) => Math.round(g.guess) === k).length),
  };
}

export function publicEntry(e: Entry, rank: number | null, you: boolean) {
  return {
    rank,
    you,
    name: e.name,
    total: e.total,
    ms: Number(e.ms),
    answered: e.answers.length,
    finished: e.finished,
    grid: e.answers.map((a) => a.points),
  };
}
