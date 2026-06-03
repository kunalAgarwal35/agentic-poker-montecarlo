import { describe, it, expect } from 'vitest';
import { compileIntentToPQL, QueryIntent } from '@/lib/intent';

function intent(over: Partial<any> = {}) {
  return QueryIntent.parse({
    game: 'omahahi5',
    players: [
      { name: 'PLAYER_1', cards: 'AsAhKsKhQs' },
      { name: 'PLAYER_2', cards: 'JdTd9d8d7d' },
    ],
    ...over,
  });
}

describe('compileIntentToPQL', () => {
  it('compiles a two-player equity query with a board', () => {
    const pql = compileIntentToPQL(intent({ board: '2c3c4c' }));
    expect(pql).toBe(
      "select avg(riverEquity(PLAYER_1)) as p1, avg(riverEquity(PLAYER_2)) as p2 " +
      "from game='omahahi5', board='2c3c4c', PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'",
    );
  });

  it('omits board/dead when absent', () => {
    const pql = compileIntentToPQL(intent());
    expect(pql).toBe(
      "select avg(riverEquity(PLAYER_1)) as p1, avg(riverEquity(PLAYER_2)) as p2 " +
      "from game='omahahi5', PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'",
    );
  });

  it('includes winsHi count columns when requested', () => {
    const pql = compileIntentToPQL(intent({ board: '2c3c4c5c', metrics: ['equity', 'winsHi'] }));
    expect(pql).toBe(
      "select avg(riverEquity(PLAYER_1)) as p1, count(winsHi(PLAYER_1)) as p1_wins, " +
      "avg(riverEquity(PLAYER_2)) as p2, count(winsHi(PLAYER_2)) as p2_wins " +
      "from game='omahahi5', board='2c3c4c5c', PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'",
    );
  });

  it('includes dead cards when present', () => {
    const pql = compileIntentToPQL(intent({ board: '2c3c4c', dead: 'JsJh' }));
    expect(pql).toContain("board='2c3c4c', dead='JsJh', PLAYER_1=");
  });

  it('compiles a holdem equity query', () => {
    const q = QueryIntent.parse({
      game: 'holdem',
      players: [
        { name: 'PLAYER_1', cards: 'AsKs' },
        { name: 'PLAYER_2', cards: 'QdQh' },
      ],
    });
    expect(compileIntentToPQL(q)).toBe(
      "select avg(riverEquity(PLAYER_1)) as p1, avg(riverEquity(PLAYER_2)) as p2 " +
      "from game='holdem', PLAYER_1='AsKs', PLAYER_2='QdQh'",
    );
  });

  it('compiles a tiesHi outcome query', () => {
    const q = QueryIntent.parse({ game: 'holdem', players: [{ name: 'PLAYER_1', cards: 'AsKs' }, { name: 'PLAYER_2', cards: 'QdQh' }], metrics: ['winsHi', 'tiesHi'] });
    expect(compileIntentToPQL(q)).toBe(
      "select count(winsHi(PLAYER_1)) as p1_wins, count(tiesHi(PLAYER_1)) as p1_ties, " +
      "count(winsHi(PLAYER_2)) as p2_wins, count(tiesHi(PLAYER_2)) as p2_ties " +
      "from game='holdem', PLAYER_1='AsKs', PLAYER_2='QdQh'",
    );
  });

  it('compiles a distribution query (9 category counts for PLAYER_1)', () => {
    const q = QueryIntent.parse({ game: 'holdem', players: [{ name: 'PLAYER_1', cards: 'AsKs' }, { name: 'PLAYER_2', cards: 'QdQh' }], metrics: ['distribution'] });
    const pql = compileIntentToPQL(q);
    expect(pql).toContain("count(exactHandType(PLAYER_1, river, highcard)) as p1_highcard");
    expect(pql).toContain("count(exactHandType(PLAYER_1, river, straightflush)) as p1_straightflush");
    expect(pql).toContain("from game='holdem', PLAYER_1='AsKs', PLAYER_2='QdQh'");
    expect(pql).not.toContain('p2_pair');
  });

  it('passes a range string through unquoted in the player slot', () => {
    const q = QueryIntent.parse({ game: 'holdem', players: [{ name: 'PLAYER_1', cards: 'AsKs' }, { name: 'PLAYER_2', cards: 'QQ+' }] });
    expect(compileIntentToPQL(q)).toContain("PLAYER_2='QQ+'");
  });

  it('compiles nutHi (per-player nut frequency)', () => {
    const q = QueryIntent.parse({ game: 'holdem', players: [{ name: 'PLAYER_1', cards: 'AsKs' }, { name: 'PLAYER_2', cards: 'QdQh' }], metrics: ['nutHi'] });
    const pql = compileIntentToPQL(q);
    expect(pql).toContain('count(nutHi(PLAYER_1)) as p1_nuts');
    expect(pql).toContain('count(nutHi(PLAYER_2)) as p2_nuts');
  });

  it('compiles winningDistribution (9 winner-category counts)', () => {
    const q = QueryIntent.parse({ game: 'holdem', players: [{ name: 'PLAYER_1', cards: 'AsKs' }, { name: 'PLAYER_2', cards: 'QdQh' }], metrics: ['winningDistribution'] });
    const pql = compileIntentToPQL(q);
    expect(pql).toContain('count(winningHandType(PLAYER_1, river, flush)) as win_flush');
    expect(pql).toContain('count(winningHandType(PLAYER_1, river, straightflush)) as win_straightflush');
  });

  it('compiles a handTypeQuery threshold', () => {
    const q = QueryIntent.parse({
      game: 'omahahi5',
      players: [{ name: 'PLAYER_1', cards: 'AsAhKsKhQs' }, { name: 'PLAYER_2', cards: 'JdTd9d8d7d' }],
      handTypeQuery: { player: 'PLAYER_1', mode: 'min', category: 'flush' },
    });
    expect(compileIntentToPQL(q)).toContain('count(minHandType(PLAYER_1, river, flush)) as p1_ht');
  });

  it('compiles a draws query (one count per kind)', () => {
    const q = QueryIntent.parse({
      game: 'holdem',
      players: [{ name: 'PLAYER_1', cards: 'KhQh' }, { name: 'PLAYER_2', cards: 'AsAd' }],
      board: 'Ah7c2h',
      drawQuery: { player: 'PLAYER_1', kinds: ['flushDraw', 'oesd'] },
    });
    const pql = compileIntentToPQL(q);
    expect(pql).toContain('count(flushDraw(PLAYER_1)) as p1_flushDraw');
    expect(pql).toContain('count(oesd(PLAYER_1)) as p1_oesd');
    expect(pql).toContain("board='Ah7c2h'");
  });

  it('compiles an outs query (avg + cumulative buckets)', () => {
    const q = QueryIntent.parse({
      game: 'holdem',
      players: [{ name: 'PLAYER_1', cards: 'KhQh' }, { name: 'PLAYER_2', cards: 'AsAd' }],
      board: 'Ah7c2h',
      outsQuery: { player: 'PLAYER_1', handtype: 'flush', street: 'turn', buckets: 12 },
    });
    const pql = compileIntentToPQL(q);
    expect(pql).toContain('avg(outsToHandType(PLAYER_1, turn, flush)) as p1_avgouts');
    expect(pql).toContain('count(minOutsToHandType(PLAYER_1, turn, flush, 1)) as p1_ge1');
    expect(pql).toContain('count(minOutsToHandType(PLAYER_1, turn, flush, 12)) as p1_ge12');
  });

  it('compiles a handTypeQuery with an explicit street', () => {
    const q = QueryIntent.parse({
      game: 'holdem',
      players: [{ name: 'PLAYER_1', cards: '10%' }],
      handTypeQuery: { player: 'PLAYER_1', mode: 'min', category: 'twopair', street: 'flop' },
    });
    expect(compileIntentToPQL(q)).toContain('count(minHandType(PLAYER_1, flop, twopair)) as p1_ht');
  });

  it('defaults handTypeQuery street to river', () => {
    const q = QueryIntent.parse({
      game: 'holdem',
      players: [{ name: 'PLAYER_1', cards: '10%' }],
      handTypeQuery: { player: 'PLAYER_1', mode: 'min', category: 'twopair' },
    });
    expect(compileIntentToPQL(q)).toContain('count(minHandType(PLAYER_1, river, twopair)) as p1_ht');
  });
});
