import { QueryIntent, compileIntentToPQL } from '@/lib/intent';
import { pickViz } from '@/lib/viz';
import { runPql as defaultRunPql, runGraph as defaultRunGraph, type GraphResult, type GraphPayload } from '@/lib/engine';
import type { PQLResult, VizSpec } from '@/lib/types';

export type ResultContext = { game: string; board?: string; players: { name: string; cards: string }[] };
export type BuildQueryResult = { resolvedQuery: string; result: PQLResult; viz: VizSpec; context: ResultContext };
type Deps = {
  runPql?: (q: string) => Promise<PQLResult>;
  runGraph?: (p: GraphPayload) => Promise<GraphResult>;
};

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
    return { resolvedQuery: `graph:${kind} (hero=${hero})`, result, viz, context };
  }

  const runPql = deps.runPql ?? defaultRunPql;
  const resolvedQuery = compileIntentToPQL(intent);
  const result = await runPql(resolvedQuery);
  const viz = pickViz(intent, result);
  return { resolvedQuery, result, viz, context };
}
