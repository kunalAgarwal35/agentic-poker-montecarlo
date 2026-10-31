// @vitest-environment node
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ---------------------------------------------------------------------------
// In-memory stand-in for the two challenge tables, driven by the SQL text the
// data layer sends. Enough to exercise the real route + lib code end to end.
// ---------------------------------------------------------------------------
type Row = Record<string, unknown>;
const db = { challenges: [] as Row[], entries: [] as Row[] };

function fake(strings: TemplateStringsArray, ...v: unknown[]) {
  const sql = strings.join('?').replace(/\s+/g, ' ');
  if (sql.startsWith(' CREATE') || sql.startsWith('CREATE')) return Promise.resolve([]);
  if (sql.includes('count(*)::int AS n FROM trainer_challenges')) return Promise.resolve([{ n: 0 }]);
  if (sql.includes('INSERT INTO trainer_challenges')) {
    const [code, mode, game, ids, name, id, ip, days] = v as [string, string, string, string, string, string, string, number];
    if (db.challenges.some((c) => c.code === code)) return Promise.resolve([]);
    db.challenges.push({
      code, mode, game, question_ids: JSON.parse(ids), creator_name: name, creator_id: id, creator_ip: ip,
      created_at: new Date().toISOString(), expires_at: new Date(Date.now() + days * 864e5).toISOString(),
    });
    return Promise.resolve([{ code }]);
  }
  if (sql.includes('FROM trainer_challenges WHERE code')) return Promise.resolve(db.challenges.filter((c) => c.code === v[0]));
  if (sql.includes('FROM trainer_challenge_entries WHERE code')) return Promise.resolve(db.entries.filter((e) => e.code === v[0]));
  if (sql.includes('INSERT INTO trainer_challenge_entries')) {
    const [code, player_id, name, answers, total, ms, finished] = v as [string, string, string, string, number, number, boolean];
    if (db.entries.some((e) => e.code === code && e.player_id === player_id)) return Promise.resolve([]);
    const row = { code, player_id, name, answers: JSON.parse(answers), total, ms, finished, updated_at: '' };
    db.entries.push(row);
    return Promise.resolve([row]);
  }
  if (sql.includes('UPDATE trainer_challenge_entries')) {
    const [rec, points, ms, finished, name, code, player, index] = v as [string, number, number, boolean, string, string, string, number];
    const e = db.entries.find((x) => x.code === code && x.player_id === player);
    if (!e || (e.answers as unknown[]).length !== index) return Promise.resolve([]);
    e.answers = [...(e.answers as unknown[]), ...JSON.parse(rec)];
    e.total = (e.total as number) + points;
    e.ms = (e.ms as number) + ms;
    e.finished = finished;
    e.name = name;
    return Promise.resolve([e]);
  }
  if (sql.includes('FROM trainer_feedback')) return Promise.resolve([]);
  throw new Error('unexpected SQL: ' + sql);
}
vi.mock('@neondatabase/serverless', () => ({ neon: () => fake }));

const ORIGIN = 'https://train.magicflop.com';
const json = (method: string, url: string, body?: unknown) =>
  new Request(url, { method, headers: { 'content-type': 'application/json', origin: ORIGIN }, body: body ? JSON.stringify(body) : undefined });
const ctx = (code: string) => ({ params: Promise.resolve({ code }) });

describe('challenge input parsing', () => {
  it('parseCreate validates mode/game/length/name/playerId', async () => {
    const { parseCreate } = await import('@/lib/trainerChallenges');
    const ok = { mode: 'range_reads', game: 'PLO4', length: 10, name: '  Kunal  ', playerId: 'abcd1234-ef' };
    expect(parseCreate(ok)).toMatchObject({ name: 'Kunal' });
    expect(parseCreate({ ...ok, mode: 'mixed', game: 'mixed', length: 5 })).not.toHaveProperty('error');
    for (const bad of [{ mode: 'x' }, { game: 'PLO6' }, { length: 7 }, { name: '   ' }, { playerId: 'x' }]) {
      expect(parseCreate({ ...ok, ...bad })).toHaveProperty('error');
    }
  });
  it('cleanName strips markup/control chars and caps length', async () => {
    const { cleanName } = await import('@/lib/trainerChallenges');
    expect(cleanName('<b>Kunal</b>\n')).toBe('bKunal/b');
    expect(cleanName('x'.repeat(40))).toHaveLength(20);
    expect(cleanName(42)).toBeNull();
  });
  it('codes avoid ambiguous characters', async () => {
    const { newCode, CODE_RE } = await import('@/lib/trainerChallenges');
    for (let i = 0; i < 200; i++) {
      const c = newCode();
      expect(c).toMatch(CODE_RE);
      expect(c).not.toMatch(/[01OIL]/);
    }
  });
});

describe('question bank + selection', () => {
  it('server scoring matches the trainer formula', async () => {
    const { score } = await import('@/lib/trainerBank');
    const est = { kind: 'estimate', answer: 20 } as never;
    expect(score(est, 20)).toBe(100);
    expect(score(est, 26)).toBe(Math.round(100 * Math.exp(-1)));
    expect(score({ kind: 'choice', correct: 1 } as never, 1)).toBe(100);
    expect(score({ kind: 'choice', correct: 1 } as never, 0)).toBe(0);
  });
  it('picks n questions of the mode/game with distinct boards; mixed covers every mode', async () => {
    const { pickQuestions, MODES } = await import('@/lib/trainerBank');
    const empty = { questions: {}, templates: {} };
    const qs = pickQuestions('card_removal', 'PLO5', 10, empty);
    expect(qs).toHaveLength(10);
    expect(qs.every((q) => q.mode === 'card_removal' && q.game === 'PLO5')).toBe(true);
    expect(new Set(qs.map((q) => q.board.join(''))).size).toBe(10);
    const mixed = pickQuestions('mixed', 'mixed', 10, empty);
    expect(new Set(mixed.map((q) => q.mode))).toEqual(new Set(MODES));
    expect(new Set(mixed.map((q) => q.game))).toEqual(new Set(['PLO4', 'PLO5']));
  });
  it('public questions hide answers', async () => {
    const { bank, toPublic } = await import('@/lib/trainerBank');
    for (const q of bank().all.slice(0, 50)) {
      const p = toPublic(q) as Record<string, unknown>;
      for (const k of ['answer', 'baseline', 'correct', 'values']) expect(p).not.toHaveProperty(k);
    }
  });
});

describe('challenge flow', () => {
  beforeEach(() => {
    vi.resetModules();
    db.challenges = [];
    db.entries = [];
    vi.stubEnv('DATABASE_URL', 'postgres://test');
  });
  afterEach(() => vi.unstubAllEnvs());

  it('create -> play in order -> leaderboard, with server-side scoring', async () => {
    const create = await import('@/app/api/trainer/challenges/route');
    const one = await import('@/app/api/trainer/challenges/[code]/route');
    const answer = await import('@/app/api/trainer/challenges/[code]/answer/route');
    const { bank } = await import('@/lib/trainerBank');

    const res = await create.POST(json('POST', 'http://t/api/trainer/challenges', {
      mode: 'range_reads', game: 'PLO4', length: 5, name: 'Kunal', playerId: 'player-aaaa',
    }));
    expect(res.status).toBe(201);
    const { code } = await res.json();

    // Before playing: questions without answers, empty board, no "me".
    let d = await (await one.GET(json('GET', `http://t/x?playerId=player-bbbb`), ctx(code))).json();
    expect(d.creatorName).toBe('Kunal');
    expect(d.questions).toHaveLength(5);
    expect(d.questions[0]).not.toHaveProperty('answer');
    expect(d.me).toBeNull();

    // Skipping ahead is refused.
    const skip = await answer.POST(json('POST', 'http://t', { playerId: 'player-bbbb', name: 'Alex', index: 2, guess: 10, ms: 1000 }), ctx(code));
    expect(skip.status).toBe(409);

    // Alex answers every question exactly right -> 500.
    for (let i = 0; i < 5; i++) {
      const truth = bank().byId.get((db.challenges[0].question_ids as string[])[i])!;
      const r = await answer.POST(json('POST', 'http://t', { playerId: 'player-bbbb', name: 'Alex', index: i, guess: truth.answer, ms: 2000 }), ctx(code));
      const b = await r.json();
      expect(b.points).toBe(100);
      expect(b.reveal.answer).toBe(truth.answer);
    }
    // Re-answering is refused.
    const again = await answer.POST(json('POST', 'http://t', { playerId: 'player-bbbb', name: 'Alex', index: 4, guess: 1, ms: 1 }), ctx(code));
    expect(again.status).toBe(409);

    // Kunal guesses 50% every time.
    let last;
    for (let i = 0; i < 5; i++) {
      last = await (await answer.POST(json('POST', 'http://t', { playerId: 'player-aaaa', name: 'Kunal', index: i, guess: 50, ms: 1000 }), ctx(code))).json();
    }
    expect(last.finished).toBe(true);
    expect(last.rank).toBe(2);
    expect(last.of).toBe(2);
    expect(last.field.count).toBe(1); // Alex's answer to the same question

    d = await (await one.GET(json('GET', `http://t/x?playerId=player-aaaa`), ctx(code))).json();
    expect(d.leaderboard.map((r: { name: string; rank: number; you: boolean }) => [r.name, r.rank, r.you])).toEqual([
      ['Alex', 1, false],
      ['Kunal', 2, true],
    ]);
    expect(d.isCreator).toBe(true);
    expect(d.me.answers).toHaveLength(5);
    expect(d.me.answers[0].reveal).toHaveProperty('answer');
    expect(d.leaderboard[0]).not.toHaveProperty('player_id');
  });

  it('unknown and malformed codes 404', async () => {
    const one = await import('@/app/api/trainer/challenges/[code]/route');
    expect((await one.GET(json('GET', 'http://t/x'), ctx('ZZZZZZ'))).status).toBe(404);
    expect((await one.GET(json('GET', 'http://t/x'), ctx('bad!'))).status).toBe(404);
  });

  it('expired challenges refuse answers', async () => {
    const create = await import('@/app/api/trainer/challenges/route');
    const answer = await import('@/app/api/trainer/challenges/[code]/answer/route');
    const { code } = await (await create.POST(json('POST', 'http://t', {
      mode: 'hero_equity', game: 'PLO5', length: 5, name: 'K', playerId: 'player-aaaa',
    }))).json();
    db.challenges[0].expires_at = new Date(Date.now() - 1000).toISOString();
    const r = await answer.POST(json('POST', 'http://t', { playerId: 'player-aaaa', name: 'K', index: 0, guess: 40, ms: 1 }), ctx(code));
    expect(r.status).toBe(410);
  });
});
