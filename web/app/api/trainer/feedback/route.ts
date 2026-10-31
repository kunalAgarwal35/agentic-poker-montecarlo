import { dbConfigured } from '@/lib/db';
import { corsHeaders } from '@/lib/trainerCors';
import { ensureTrainerSchema, getStats, parseVote, recordVote } from '@/lib/trainerFeedback';

// PLO Trainer question feedback.
//   GET  -> { questions: {id: {up,down}}, templates: {id: {up,down}} }
//   POST { questionId, template, clientId, vote: 'up'|'down' } -> { ok }
// Degrades to empty stats / skipped writes when the DB isn't configured, so
// the trainer never breaks because of feedback.

export const runtime = 'nodejs';

export async function OPTIONS(req: Request) {
  return new Response(null, { status: 204, headers: corsHeaders(req) });
}

export async function GET(req: Request) {
  const headers = { ...corsHeaders(req), 'Cache-Control': 'public, s-maxage=60, stale-while-revalidate=300' };
  if (!dbConfigured()) return Response.json({ questions: {}, templates: {} }, { headers });
  try {
    await ensureTrainerSchema();
    return Response.json(await getStats(), { headers });
  } catch {
    return Response.json({ questions: {}, templates: {} }, { headers });
  }
}

export async function POST(req: Request) {
  const headers = corsHeaders(req);
  let body: unknown = null;
  try {
    body = await req.json();
  } catch {
    /* handled by validation */
  }
  const v = parseVote(body);
  if ('error' in v) return Response.json({ error: v.error }, { status: 400, headers });
  if (!dbConfigured()) return Response.json({ ok: false, skipped: true }, { headers });
  try {
    await ensureTrainerSchema();
    await recordVote(v);
    return Response.json({ ok: true }, { headers });
  } catch {
    return Response.json({ ok: false }, { headers });
  }
}
