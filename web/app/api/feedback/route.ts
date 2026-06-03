import { dbConfigured, ensureSchema, addFeedback } from '@/lib/db';
import { checkRateLimit, clientIp } from '@/lib/ratelimit';

export const runtime = 'nodejs';

const MAX_COMMENT = 2000;
const MAX_QUESTION = 500;

export async function POST(req: Request) {
  let body: any;
  try {
    body = await req.json();
  } catch {
    body = null;
  }
  const rating = body?.rating;
  if (rating !== 'up' && rating !== 'down') {
    return Response.json(
      { error: "rating must be 'up' or 'down'" },
      { status: 400 },
    );
  }
  // Size caps prevent stored-data bloat/abuse (truncate rather than reject so
  // best-effort feedback capture still records something).
  if (typeof body.comment === 'string' && body.comment.length > MAX_COMMENT) {
    body.comment = body.comment.slice(0, MAX_COMMENT);
  }
  if (typeof body.question === 'string' && body.question.length > MAX_QUESTION) {
    body.question = body.question.slice(0, MAX_QUESTION);
  }
  if (!dbConfigured()) {
    return Response.json({ ok: false, skipped: true });
  }
  const rl = await checkRateLimit(clientIp(req), 'feedback');
  if (!rl.ok) {
    return Response.json({ error: rl.message }, { status: rl.status });
  }
  try {
    await ensureSchema();
    const { id } = await addFeedback(body);
    return Response.json({ ok: true, id });
  } catch {
    // Best-effort — never block the UI.
    return Response.json({ ok: false });
  }
}
