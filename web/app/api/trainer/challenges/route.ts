import { dbConfigured } from '@/lib/db';
import { clientIp } from '@/lib/ratelimit';
import { pickQuestions } from '@/lib/trainerBank';
import { createChallenge, ensureChallengeSchema, parseCreate } from '@/lib/trainerChallenges';
import { corsHeaders } from '@/lib/trainerCors';
import { ensureTrainerSchema, getStats } from '@/lib/trainerFeedback';

// POST { mode, game, length, name, playerId } -> { code }
export const runtime = 'nodejs';

export async function OPTIONS(req: Request) {
  return new Response(null, { status: 204, headers: corsHeaders(req) });
}

export async function POST(req: Request) {
  const headers = corsHeaders(req);
  let body: unknown = null;
  try {
    body = await req.json();
  } catch {
    /* validation below */
  }
  const input = parseCreate(body);
  if ('error' in input) return Response.json({ error: input.error }, { status: 400, headers });
  if (!dbConfigured()) return Response.json({ error: 'Challenges are unavailable right now.' }, { status: 503, headers });
  try {
    await ensureTrainerSchema();
    await ensureChallengeSchema();
    const stats = await getStats().catch(() => ({ questions: {}, templates: {} }));
    const qs = pickQuestions(input.mode, input.game, input.length, stats);
    if (qs.length < input.length) return Response.json({ error: 'Not enough questions for that mode' }, { status: 400, headers });
    const r = await createChallenge(input, qs.map((q) => q.id), clientIp(req));
    if (typeof r !== 'string') return Response.json({ error: r.error }, { status: r.status, headers });
    return Response.json({ code: r }, { status: 201, headers });
  } catch (e) {
    console.error('create challenge failed', e);
    return Response.json({ error: 'Could not create the challenge. Try again.' }, { status: 500, headers });
  }
}
