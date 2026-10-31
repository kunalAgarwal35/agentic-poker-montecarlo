import { runPql } from '@/lib/engine';
import { buildEquityQuery, parseEquityRequest } from '@/lib/trainerEquity';
import { corsHeaders } from '@/lib/trainerCors';

// Public equity endpoint for the PLO Trainer (train.magicflop.com).
//
// The trainer is a static Vite SPA, so it can't hold ENGINE_KEY or call the
// Railway engine directly (no CORS there, and the key must stay server-side).
// This route is the thin, validated gateway: it turns "these exact hands on this
// board" into one PQL query and returns every player's equity in percent.
//
// Deliberately NOT behind checkRateLimit: that limiter's global day budget is
// the kill-switch for paid Claude calls, and trainer traffic (cheap, no LLM)
// would burn through it. The engine already clamps trials, and the input here
// is bounded to <= 6 exact hands, so a single request is cheap by construction.

export const runtime = 'nodejs';
export const maxDuration = 60;

const TRIALS = 30000; // engine ceiling; enumeration is used automatically when cheaper

export async function OPTIONS(req: Request) {
  return new Response(null, { status: 204, headers: corsHeaders(req) });
}

/** Health/warm-up: pings the engine's /health, which starts its numba warmup. */
export async function GET(req: Request) {
  const engineUrl = process.env.ENGINE_URL ?? 'http://localhost:5050';
  let engine = 'unreachable';
  try {
    const r = await fetch(`${engineUrl}/health`, { signal: AbortSignal.timeout(8000) });
    engine = r.ok ? 'ok' : `http ${r.status}`;
  } catch {
    /* engine asleep or down — reported below */
  }
  return Response.json({ ok: true, engine }, { headers: corsHeaders(req) });
}

export async function POST(req: Request) {
  const headers = corsHeaders(req);
  let body: unknown = null;
  try {
    body = await req.json();
  } catch {
    /* handled by validation */
  }
  const parsed = parseEquityRequest(body);
  if (!parsed.ok) return Response.json({ error: parsed.error }, { status: 400, headers });

  const query = buildEquityQuery(parsed.game, parsed.hands, parsed.board, parsed.dead);
  try {
    const result = await runPql(query, { trials: TRIALS });
    const equities = parsed.hands.map((_, i) => {
      const v = result.values[`p${i + 1}`];
      return typeof v === 'number' ? Math.round(v * 100 * 100) / 100 : null;
    });
    if (equities.some((e) => e === null)) {
      return Response.json({ error: 'engine returned no equity' }, { status: 502, headers });
    }
    return Response.json(
      {
        hero: equities[0],
        hero_equity: equities[0],
        equities,
        game: parsed.game,
        trials: result.trials,
        mode: result.mode,
        is_fallback: false,
      },
      { headers },
    );
  } catch (e) {
    return Response.json(
      { error: 'Equity calculation failed', details: (e as Error).message },
      { status: 503, headers },
    );
  }
}
