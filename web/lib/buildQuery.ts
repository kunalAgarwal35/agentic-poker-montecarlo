import { QueryIntent, compileIntentToPQL } from '@/lib/intent';
import { pickViz } from '@/lib/viz';
import { runPql as defaultRunPql, runGraph as defaultRunGraph, type GraphResult, type GraphPayload, type RunPqlOptions } from '@/lib/engine';
import type { PQLResult, VizSpec, HeroDraws } from '@/lib/types';

export type ResultContext = { game: string; board?: string; players: { name: string; cards: string }[] };
export type BuildQueryResult = { resolvedQuery: string; result: PQLResult; viz: VizSpec; context: ResultContext; heroDraws?: HeroDraws };
type Deps = {
  runPql?: (q: string, opts?: RunPqlOptions) => Promise<PQLResult>;
  runGraph?: (p: GraphPayload) => Promise<GraphResult>;
};

// Hole-card count per game (for distinguishing an EXACT hand from a range).
const HOLE_CARDS: Record<string, number> = { holdem: 2, omahahi: 4, omahahi5: 5, omahahi6: 6 };

// True only for an EXACT hand (rank+suit) of the right length for the game — NOT a range.
function isExactHand(cards: string, game: string): boolean {
  const n = HOLE_CARDS[game];
  if (!n) return false;
  return new RegExp(`^([2-9TJQKA][shdc]){${n}}$`, 'i').test(cards);
}

// Best-effort: compute the hero's REAL draws + out counts from the engine on a
// flop/turn board with an EXACT hero hand. Returns null when there's nothing to
// verify (preflop/river/missing board, or no exact hero). Throws on engine
// failure — the caller swallows it so the main answer is unaffected.
async function computeHeroDraws(
  intent: QueryIntent,
  runPql: (q: string, opts?: RunPqlOptions) => Promise<PQLResult>,
): Promise<HeroDraws | null> {
  const board = intent.board ?? '';
  let street: 'flop' | 'turn';
  if (board.length === 6) street = 'flop';
  else if (board.length === 8) street = 'turn';
  else return null; // preflop / river / missing — no draws to verify

  // Hero = first player with an EXACT hand for the game (ranges don't have draws).
  const heroIdx = intent.players.findIndex((p) => isExactHand(p.cards, intent.game));
  if (heroIdx < 0) return null;
  if (intent.players.length < 2) return null; // need an opponent for a valid scenario

  const heroRef = `PLAYER_${heroIdx + 1}`;
  const select =
    `count(flushDraw(${heroRef}, ${street})) as fd, ` +
    `count(oesd(${heroRef}, ${street})) as od, ` +
    `count(gutshot(${heroRef}, ${street})) as gs, ` +
    `count(straightDraw(${heroRef}, ${street})) as sd, ` +
    `avg(outsToHandType(${heroRef}, ${street}, flush)) as fo, ` +
    `avg(outsToHandType(${heroRef}, ${street}, straight)) as so`;

  const fromParts: string[] = [`game='${intent.game}'`, `board='${intent.board}'`];
  if (intent.dead) fromParts.push(`dead='${intent.dead}'`);
  intent.players.forEach((p, idx) => { fromParts.push(`PLAYER_${idx + 1}='${p.cards}'`); });

  const query = `select ${select} from ${fromParts.join(', ')}`;
  const { values } = await runPql(query, { trials: 400 });

  return {
    player: intent.players[heroIdx].name,
    flushDraw: (values.fd ?? 0) > 0,
    straightDraw: (values.sd ?? 0) > 0,
    oesd: (values.od ?? 0) > 0,
    gutshot: (values.gs ?? 0) > 0,
    flushOuts: Math.round(values.fo ?? 0),
    straightOuts: Math.round(values.so ?? 0),
  };
}

function graphViz(kind: 'street' | 'distribution' | 'vsclass', intent: QueryIntent, gr: GraphResult): VizSpec {
  const base = { mode: gr.mode, trials: gr.trials };
  if (kind === 'street') {
    const series = (gr.series ?? []).map((s, idx) => ({ name: `P${idx + 1}`, points: s.points }));
    return { kind: 'equity-by-street', ...base, series };
  }
  const heroIdx = intent.players.findIndex((p) => p.name === intent.graphQuery!.hero);
  const heroLabel = `P${(heroIdx < 0 ? 0 : heroIdx) + 1}`;
  if (kind === 'distribution') {
    return { kind: 'equity-distribution', ...base, player: heroLabel, mean: gr.mean ?? 0,
      combos: gr.combos ?? 0, sampledCombos: gr.sampled_combos ?? null, buckets: gr.buckets ?? [] };
  }
  return { kind: 'equity-vs-class', ...base, player: heroLabel, rows: gr.rows ?? [] };
}

export async function executeBuildQuery(rawIntent: unknown, deps: Deps = {}): Promise<BuildQueryResult> {
  const intent = QueryIntent.parse(rawIntent); // defensive validation

  const context: ResultContext = {
    game: intent.game,
    board: intent.board,
    players: intent.players.map((p) => ({ name: p.name, cards: p.cards })),
  };

  const runPql = deps.runPql ?? defaultRunPql;

  // Best-effort engine-verified hero draws/outs (flop/turn + exact hero only).
  // Any failure leaves heroDraws undefined; the main answer is unaffected.
  let heroDraws: HeroDraws | undefined;
  try {
    heroDraws = (await computeHeroDraws(intent, runPql)) ?? undefined;
  } catch {
    heroDraws = undefined;
  }

  if (intent.graphQuery) {
    const runGraph = deps.runGraph ?? defaultRunGraph;
    const { kind, hero } = intent.graphQuery;
    const players: Record<string, string> = {};
    intent.players.forEach((p, idx) => { players[`PLAYER_${idx + 1}`] = p.cards; });
    const gr = await runGraph({
      game: intent.game, board: intent.board ?? '', dead: intent.dead ?? '',
      hero, players, kind,
    });
    const viz = graphViz(kind, intent, gr);
    const result: PQLResult = { values: {}, columns: [], trials: gr.trials, mode: gr.mode, seed: gr.seed };
    return { resolvedQuery: `graph:${kind} (hero=${hero})`, result, viz, context, heroDraws };
  }

  const resolvedQuery = compileIntentToPQL(intent);
  const result = await runPql(resolvedQuery);
  const viz = pickViz(intent, result);
  return { resolvedQuery, result, viz, context, heroDraws };
}
