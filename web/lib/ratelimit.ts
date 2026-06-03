import { dbConfigured, ensureSchema, getSql } from '@/lib/db';

// Defaults are intentionally conservative — they protect the owner's paid Claude
// key from runaway cost. Override via env on the deployment.
const IP_PER_HOUR = Number(process.env.RATE_LIMIT_IP_PER_HOUR ?? 20);
const GLOBAL_PER_DAY = Number(process.env.RATE_LIMIT_GLOBAL_PER_DAY ?? 300);

export type RateLimitResult = {
  ok: boolean;
  status?: number;
  message?: string;
};

/**
 * Best-effort, Neon-backed rate limiter + daily budget kill-switch.
 *
 * Order of checks (cost-first):
 *   1. GLOBAL day budget (fail-CLOSED, 503) — the cost ceiling for the whole app.
 *   2. Per-IP hourly cap (429) — stops a single abuser.
 *   3. Otherwise record the event and allow.
 *
 * Degrades OPEN when the DB is unconfigured (local dev) or on any DB error, so a
 * database hiccup never takes chat down. The input-size cap in the route is the
 * second line of defence and is enforced independently of this function.
 */
export async function checkRateLimit(
  ip: string,
  route: string,
): Promise<RateLimitResult> {
  // No DB configured (local dev): degrade open, never break dev.
  if (!dbConfigured()) {
    return { ok: true };
  }

  try {
    await ensureSchema();
    const sql = getSql();

    // GLOBAL budget over the last 24h — the cost kill-switch. Checked first so a
    // hit returns 503 regardless of which IP is asking.
    const globalRows = (await sql`
      SELECT count(*)::int AS n
      FROM rate_events
      WHERE ts > now() - interval '24 hours'
    `) as { n: number }[];
    const globalCount = globalRows[0]?.n ?? 0;
    if (globalCount >= GLOBAL_PER_DAY) {
      return {
        ok: false,
        status: 503,
        message:
          "The app has hit today's free usage cap — please try again tomorrow.",
      };
    }

    // Per-IP requests in the last hour.
    const ipRows = (await sql`
      SELECT count(*)::int AS n
      FROM rate_events
      WHERE ip = ${ip}
        AND ts > now() - interval '1 hour'
    `) as { n: number }[];
    const ipCount = ipRows[0]?.n ?? 0;
    if (ipCount >= IP_PER_HOUR) {
      return {
        ok: false,
        status: 429,
        message:
          "You're sending requests too quickly — please wait a minute and try again.",
      };
    }

    // Allowed: record the event.
    await sql`INSERT INTO rate_events (ip, route) VALUES (${ip}, ${route})`;

    // Opportunistic cleanup (~1 in 50 requests) so the ledger never grows
    // unbounded. The 24h/1h window filters already ignore stale rows, so this is
    // purely housekeeping — keep it cheap and best-effort.
    if (Math.random() < 0.02) {
      await sql`DELETE FROM rate_events WHERE ts < now() - interval '25 hours'`;
    }

    return { ok: true };
  } catch (err) {
    // FAIL OPEN: a DB error must not take down chat. The route's size cap still
    // applies.
    console.error('[ratelimit] DB error, failing open:', err);
    return { ok: true };
  }
}

/**
 * Extract the client IP from the request. Behind Vercel/Railway proxies the real
 * IP is in x-forwarded-for (first hop) or x-real-ip. Falls back to 'unknown'.
 */
export function clientIp(req: Request): string {
  const xff = req.headers.get('x-forwarded-for');
  if (xff) {
    const first = xff.split(',')[0]?.trim();
    if (first) return first;
  }
  const real = req.headers.get('x-real-ip');
  if (real) return real.trim();
  return 'unknown';
}
