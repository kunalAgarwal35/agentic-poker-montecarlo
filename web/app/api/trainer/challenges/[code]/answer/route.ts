import { dbConfigured } from '@/lib/db';
import { bank, reveal } from '@/lib/trainerBank';
import {
  CODE_RE, ensureChallengeSchema, fieldStats, getChallenge, getEntries, parseAnswer, rankEntries, recordAnswer,
} from '@/lib/trainerChallenges';
import { corsHeaders } from '@/lib/trainerCors';

// POST /api/trainer/challenges/:code/answer
//   { playerId, name, index, guess, ms } -> { points, total, finished, reveal, field, rank?, of? }
// Scored here against the server's copy of the answer.
export const runtime = 'nodejs';

export async function OPTIONS(req: Request) {
  return new Response(null, { status: 204, headers: corsHeaders(req) });
}

export async function POST(req: Request, { params }: { params: Promise<{ code: string }> }) {
  const headers = corsHeaders(req);
  const code = (await params).code.toUpperCase();
  if (!CODE_RE.test(code)) return Response.json({ error: 'not found' }, { status: 404, headers });
  let body: unknown = null;
  try {
    body = await req.json();
  } catch {
    /* validation below */
  }
  const input = parseAnswer(body);
  if ('error' in input) return Response.json({ error: input.error }, { status: 400, headers });
  if (!dbConfigured()) return Response.json({ error: 'Challenges are unavailable right now.' }, { status: 503, headers });
  try {
    await ensureChallengeSchema();
    const c = await getChallenge(code);
    if (!c) return Response.json({ error: 'not found' }, { status: 404, headers });
    if (new Date(c.expires_at).getTime() < Date.now()) {
      return Response.json({ error: 'This challenge has ended.' }, { status: 410, headers });
    }
    const r = await recordAnswer(c, input);
    if ('error' in r) return Response.json({ error: r.error }, { status: r.status, headers });
    const q = bank().byId.get(c.question_ids[input.index])!;
    const entries = rankEntries(await getEntries(code));
    let rank: number | null = null;
    if (r.entry.finished) {
      const finished = entries.filter((e) => e.finished);
      rank = finished.findIndex((e) => e.player_id === input.playerId) + 1;
    }
    return Response.json(
      {
        points: r.points,
        total: r.entry.total,
        answered: r.entry.answers.length,
        finished: r.entry.finished,
        reveal: reveal(q),
        field: fieldStats(entries, input.index, input.playerId),
        rank,
        of: entries.filter((e) => e.finished).length,
      },
      { headers },
    );
  } catch (e) {
    console.error('answer failed', e);
    return Response.json({ error: 'Could not save your answer. Try again.' }, { status: 500, headers });
  }
}
