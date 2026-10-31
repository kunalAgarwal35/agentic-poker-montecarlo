import { dbConfigured } from '@/lib/db';
import { bank, reveal, toPublic } from '@/lib/trainerBank';
import {
  CODE_RE, ensureChallengeSchema, fieldStats, getChallenge, getEntries, publicEntry, rankEntries,
} from '@/lib/trainerChallenges';
import { corsHeaders } from '@/lib/trainerCors';

// GET /api/trainer/challenges/:code?playerId=...
// Questions (no answers), leaderboard, and -- for the requesting player --
// their answers with reveals and how everyone else answered those questions.
export const runtime = 'nodejs';

export async function OPTIONS(req: Request) {
  return new Response(null, { status: 204, headers: corsHeaders(req) });
}

export async function GET(req: Request, { params }: { params: Promise<{ code: string }> }) {
  const headers = { ...corsHeaders(req), 'Cache-Control': 'no-store' };
  const code = (await params).code.toUpperCase();
  if (!CODE_RE.test(code)) return Response.json({ error: 'not found' }, { status: 404, headers });
  if (!dbConfigured()) return Response.json({ error: 'Challenges are unavailable right now.' }, { status: 503, headers });
  const playerId = new URL(req.url).searchParams.get('playerId') ?? '';
  try {
    await ensureChallengeSchema();
    const c = await getChallenge(code);
    if (!c) return Response.json({ error: 'not found' }, { status: 404, headers });
    const found = c.question_ids.map((id) => bank().byId.get(id));
    if (found.some((q) => !q)) {
      // The bank was regenerated with different ids; never shift indexes.
      return Response.json({ error: 'This challenge uses questions that are no longer available.' }, { status: 410, headers });
    }
    const questions = found as NonNullable<(typeof found)[number]>[];
    const entries = rankEntries(await getEntries(code));
    let rank = 0;
    const leaderboard = entries.map((e) => publicEntry(e, e.finished ? ++rank : null, e.player_id === playerId));
    const mine = entries.find((e) => e.player_id === playerId);
    return Response.json(
      {
        code: c.code,
        mode: c.mode,
        game: c.game,
        length: questions.length,
        creatorName: c.creator_name,
        isCreator: !!playerId && c.creator_id === playerId,
        createdAt: c.created_at,
        expiresAt: c.expires_at,
        expired: new Date(c.expires_at).getTime() < Date.now(),
        questions: questions.map(toPublic),
        leaderboard,
        finishedCount: rank,
        me: mine
          ? {
              name: mine.name,
              total: mine.total,
              finished: mine.finished,
              answers: mine.answers.map((a) => ({
                ...a,
                reveal: reveal(questions[a.i]),
                field: fieldStats(entries, a.i, playerId),
              })),
            }
          : null,
      },
      { headers },
    );
  } catch (e) {
    console.error('get challenge failed', e);
    return Response.json({ error: 'Could not load the challenge.' }, { status: 500, headers });
  }
}
