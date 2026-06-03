import { dbConfigured, ensureSchema, listRecent, addRecent } from '@/lib/db';
import { SEED_EXAMPLES } from '@/lib/seedExamples';
import { checkRateLimit, clientIp } from '@/lib/ratelimit';

export const runtime = 'nodejs';

const MAX_QUESTION = 500;
const MAX_SUMMARY = 1000;

export async function GET() {
  if (!dbConfigured()) {
    return Response.json({ items: SEED_EXAMPLES });
  }
  try {
    await ensureSchema();
    const items = await listRecent();
    return Response.json({ items: items.length ? items : SEED_EXAMPLES });
  } catch {
    // Never 500 the homepage — fall back to seed examples.
    return Response.json({ items: SEED_EXAMPLES });
  }
}

export async function POST(req: Request) {
  let body: any;
  try {
    body = await req.json();
  } catch {
    body = null;
  }
  if (!body?.question || typeof body.question !== 'string') {
    return Response.json({ error: 'question is required' }, { status: 400 });
  }
  // Size caps prevent stored-data bloat/abuse.
  if (body.question.length > MAX_QUESTION) {
    return Response.json({ error: 'question is too long' }, { status: 400 });
  }
  if (typeof body.summary === 'string' && body.summary.length > MAX_SUMMARY) {
    body.summary = body.summary.slice(0, MAX_SUMMARY);
  }
  if (!dbConfigured()) {
    // No-op so capture never errors when the DB is unset.
    return Response.json({ ok: false, skipped: true });
  }
  const rl = await checkRateLimit(clientIp(req), 'recent');
  if (!rl.ok) {
    return Response.json({ error: rl.message }, { status: rl.status });
  }
  try {
    await ensureSchema();
    const item = await addRecent(body);
    return Response.json(item);
  } catch {
    // Best-effort capture — never block the UI.
    return Response.json({ ok: false });
  }
}
