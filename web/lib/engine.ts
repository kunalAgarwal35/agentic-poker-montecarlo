import type { PQLResult } from '@/lib/types';

const ENGINE_URL = process.env.ENGINE_URL ?? 'http://localhost:5050';
// Bound engine calls so a slow/cold Railway instance can't hang the Vercel function.
const ENGINE_TIMEOUT_MS = 30000;

export type RunPqlOptions = { trials?: number; seed?: number | null };

export async function runPql(query: string, opts: RunPqlOptions = {}): Promise<PQLResult> {
  let resp: Response;
  try {
    resp = await fetch(`${ENGINE_URL}/pql`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-engine-key': process.env.ENGINE_KEY ?? '' },
      body: JSON.stringify({ query, trials: opts.trials ?? 20000, seed: opts.seed ?? null }),
      signal: AbortSignal.timeout(ENGINE_TIMEOUT_MS),
    });
  } catch (e) {
    throw new Error(`Engine unreachable or timed out at ${ENGINE_URL}. Is the engine running? (${(e as Error).message})`);
  }
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const detail = (data as any).details ?? (data as any).error ?? `HTTP ${resp.status}`;
    throw new Error(`Engine error: ${detail}`);
  }
  return data as PQLResult;
}

export type GraphPayload = {
  game: string;
  board: string;
  dead: string;
  hero: string;
  players: Record<string, string>;
  kind: 'street' | 'distribution' | 'vsclass';
  trials?: number;
  seed?: number | null;
};

export type GraphResult = {
  kind: 'street' | 'distribution' | 'vsclass';
  trials: number;
  mode: 'monte_carlo' | 'enumeration';
  seed: number | null;
  series?: { name: string; points: { street: string; equity: number }[] }[];
  buckets?: { lo: number; hi: number; pct: number }[];
  mean?: number;
  combos?: number;
  sampled_combos?: number | null;
  rows?: { category: string; label: string; equity: number; freq: number }[];
};

export async function runGraph(payload: GraphPayload): Promise<GraphResult> {
  let resp: Response;
  try {
    resp = await fetch(`${ENGINE_URL}/pql-graph`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-engine-key': process.env.ENGINE_KEY ?? '' },
      body: JSON.stringify({ trials: 20000, seed: null, ...payload }),
      signal: AbortSignal.timeout(ENGINE_TIMEOUT_MS),
    });
  } catch (e) {
    throw new Error(`Engine unreachable or timed out at ${ENGINE_URL}. Is the engine running? (${(e as Error).message})`);
  }
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const detail = (data as any).details ?? (data as any).error ?? `HTTP ${resp.status}`;
    throw new Error(`Engine error: ${detail}`);
  }
  return data as GraphResult;
}
