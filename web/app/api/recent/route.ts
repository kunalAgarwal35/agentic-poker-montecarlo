import { dbConfigured, ensureSchema, listRecent, addRecent } from '@/lib/db';
import { SEED_EXAMPLES } from '@/lib/seedExamples';

export const runtime = 'nodejs';

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
  if (!body?.question) {
    return Response.json({ error: 'question is required' }, { status: 400 });
  }
  if (!dbConfigured()) {
    // No-op so capture never errors when the DB is unset.
    return Response.json({ ok: false, skipped: true });
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
