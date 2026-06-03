import { z } from 'zod';

// Phase-1 engine supports omaha-hi only, fixed hole cards, and the
// value functions riverEquity + winsHi (tiesHi/ranges come later).
export const Game = z.enum(['holdem', 'omahahi', 'omahahi5', 'omahahi6']);
export const Metric = z.enum(['equity', 'winsHi', 'tiesHi', 'distribution', 'nutHi', 'winningDistribution', 'draws', 'outs']);

export const Category = z.enum(['highcard', 'pair', 'twopair', 'trips', 'straight', 'flush', 'fullhouse', 'quads', 'straightflush']);
export const Street = z.enum(['flop', 'turn']);

export const DrawQuery = z.object({
  player: z.string(),
  kinds: z.array(z.enum(['flushDraw', 'straightDraw', 'oesd', 'gutshot'])).min(1),
  street: Street.optional(),
});

export const OutsQuery = z.object({
  player: z.string(),
  handtype: Category,
  street: Street,
  buckets: z.number().int().min(1).max(20).default(12),
});

export const GraphKind = z.enum(['street', 'distribution', 'vsclass']);
export const GraphQuery = z.object({
  kind: GraphKind,
  hero: z.string(),
});

export const Player = z.object({
  name: z.string(),
  cards: z.string(),
});

export const HandTypeQuery = z.object({
  player: z.string(),
  mode: z.enum(['min', 'max', 'exact']),
  category: z.enum(['highcard', 'pair', 'twopair', 'trips', 'straight', 'flush', 'fullhouse', 'quads', 'straightflush']),
  street: z.enum(['flop', 'turn', 'river']).default('river'),
});

export const QueryIntent = z.object({
  game: Game,
  players: z.array(Player).min(1),
  board: z.string().optional(),
  dead: z.string().optional(),
  metrics: z.array(Metric).default(['equity']),
  handTypeQuery: HandTypeQuery.optional(),
  drawQuery: DrawQuery.optional(),
  outsQuery: OutsQuery.optional(),
  graphQuery: GraphQuery.optional(),
});

export type QueryIntent = z.infer<typeof QueryIntent>;

const WIN_TOKENS = ['highcard', 'pair', 'twopair', 'trips', 'straight', 'flush', 'fullhouse', 'quads', 'straightflush'];

export function compileIntentToPQL(intent: QueryIntent): string {
  const metrics = intent.metrics.length ? intent.metrics : ['equity'];
  const selectCols: string[] = [];

  if (intent.drawQuery) {
    const { player, kinds, street } = intent.drawQuery;
    const i = (intent.players.findIndex((p) => p.name === player) + 1) || 1;
    const ref = `PLAYER_${i}`;   // players are named PLAYER_n in the FROM clause
    const arg = street ? `${ref}, ${street}` : ref;
    for (const k of kinds) selectCols.push(`count(${k}(${arg})) as p${i}_${k}`);
  } else if (intent.outsQuery) {
    const { player, handtype, street, buckets } = intent.outsQuery;
    const i = (intent.players.findIndex((p) => p.name === player) + 1) || 1;
    const ref = `PLAYER_${i}`;
    selectCols.push(`avg(outsToHandType(${ref}, ${street}, ${handtype})) as p${i}_avgouts`);
    for (let k = 1; k <= buckets; k++) {
      selectCols.push(`count(minOutsToHandType(${ref}, ${street}, ${handtype}, ${k})) as p${i}_ge${k}`);
    }
  } else if (intent.handTypeQuery) {
    const { player, mode, category, street } = intent.handTypeQuery;
    const i = (intent.players.findIndex((p) => p.name === player) + 1) || 1;
    selectCols.push(`count(${mode}HandType(${player}, ${street}, ${category})) as p${i}_ht`);
  } else if (metrics.includes('winningDistribution')) {
    for (const tok of WIN_TOKENS) {
      selectCols.push(`count(winningHandType(PLAYER_1, river, ${tok})) as win_${tok}`);
    }
  } else if (metrics.includes('distribution')) {
    for (const tok of WIN_TOKENS) {
      selectCols.push(`count(exactHandType(PLAYER_1, river, ${tok})) as p1_${tok}`);
    }
  } else {
    intent.players.forEach((_, idx) => {
      const i = idx + 1;
      for (const m of metrics) {
        if (m === 'equity') selectCols.push(`avg(riverEquity(PLAYER_${i})) as p${i}`);
        else if (m === 'winsHi') selectCols.push(`count(winsHi(PLAYER_${i})) as p${i}_wins`);
        else if (m === 'tiesHi') selectCols.push(`count(tiesHi(PLAYER_${i})) as p${i}_ties`);
        else if (m === 'nutHi') selectCols.push(`count(nutHi(PLAYER_${i})) as p${i}_nuts`);
      }
    });
  }

  const fromParts: string[] = [`game='${intent.game}'`];
  if (intent.board) fromParts.push(`board='${intent.board}'`);
  if (intent.dead) fromParts.push(`dead='${intent.dead}'`);
  intent.players.forEach((p, idx) => { fromParts.push(`PLAYER_${idx + 1}='${p.cards}'`); });

  return `select ${selectCols.join(', ')} from ${fromParts.join(', ')}`;
}
