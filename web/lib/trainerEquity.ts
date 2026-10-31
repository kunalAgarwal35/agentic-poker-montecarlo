// Validation + PQL query building for the PLO Trainer equity endpoint
// (app/api/trainer/equity/route.ts). Kept out of route.ts because Next.js only
// allows route handlers and segment config to be exported from a route file.

const CARD_RE = /^[2-9TJQKA][shdc]$/;
export const GAME_BY_HOLE: Record<number, string> = {
  2: 'holdem',
  4: 'omahahi',
  5: 'omahahi5',
  6: 'omahahi6',
};
export const MAX_PLAYERS = 6;

/** Accepts ['Ah','Kh',...] or 'AhKh...' and returns normalized 2-char cards. */
export function parseCards(raw: unknown): string[] | null {
  let cards: string[];
  if (Array.isArray(raw)) {
    if (!raw.every((c) => typeof c === 'string')) return null;
    cards = raw as string[];
  } else if (typeof raw === 'string') {
    const s = raw.replace(/[\s,]/g, '');
    if (s.length % 2 !== 0) return null;
    cards = s.match(/.{2}/g) ?? [];
  } else if (raw == null) {
    cards = [];
  } else {
    return null;
  }
  const trimmed = cards.map((c) => c.trim());
  // Exactly 2 chars each: never truncate, since these are interpolated into PQL.
  if (!trimmed.every((c) => c.length === 2)) return null;
  const norm = trimmed.map((c) => c.charAt(0).toUpperCase() + c.charAt(1).toLowerCase());
  return norm.every((c) => CARD_RE.test(c)) ? norm : null;
}

type Parsed =
  | { ok: true; game: string; hands: string[][]; board: string[]; dead: string[] }
  | { ok: false; error: string };

type EquityBody = { hands?: unknown; board?: unknown; dead_cards?: unknown; dead?: unknown };

export function parseEquityRequest(raw: unknown): Parsed {
  const body = (raw && typeof raw === 'object' ? raw : {}) as EquityBody;
  if (!Array.isArray(body.hands)) return { ok: false, error: '`hands` must be an array of hands' };
  if (body.hands.length < 2 || body.hands.length > MAX_PLAYERS) {
    return { ok: false, error: `between 2 and ${MAX_PLAYERS} hands are required` };
  }
  const hands: string[][] = [];
  for (const h of body.hands) {
    const cards = parseCards(h);
    if (!cards) return { ok: false, error: `invalid hand: ${JSON.stringify(h)}` };
    hands.push(cards);
  }
  const holeCount = hands[0].length;
  if (!hands.every((h) => h.length === holeCount)) {
    return { ok: false, error: 'all hands must have the same number of cards' };
  }
  const game = GAME_BY_HOLE[holeCount];
  if (!game) return { ok: false, error: `unsupported hand size ${holeCount} (use 2, 4, 5 or 6 cards)` };

  const board = parseCards(body.board);
  if (!board) return { ok: false, error: 'invalid board' };
  if (board.length > 5 || board.length === 1 || board.length === 2) {
    return { ok: false, error: 'board must have 0, 3, 4 or 5 cards' };
  }
  const dead = parseCards(body.dead_cards ?? body.dead);
  if (!dead) return { ok: false, error: 'invalid dead cards' };

  const all = [...hands.flat(), ...board, ...dead];
  if (new Set(all).size !== all.length) return { ok: false, error: 'duplicate cards' };
  return { ok: true, game, hands, board, dead };
}

export function buildEquityQuery(game: string, hands: string[][], board: string[], dead: string[] = []): string {
  const select = hands.map((_, i) => `avg(riverEquity(PLAYER_${i + 1})) as p${i + 1}`).join(', ');
  const from = [`game='${game}'`];
  if (board.length) from.push(`board='${board.join('')}'`);
  if (dead.length) from.push(`dead='${dead.join('')}'`);
  hands.forEach((h, i) => from.push(`PLAYER_${i + 1}='${h.join('')}'`));
  return `select ${select} from ${from.join(', ')}`;
}
