import { dbConfigured, ensureSchema, addFeedback } from '@/lib/db';

export const runtime = 'nodejs';

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
  if (!dbConfigured()) {
    return Response.json({ ok: false, skipped: true });
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
