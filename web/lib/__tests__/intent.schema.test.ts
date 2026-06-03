import { describe, it, expect } from 'vitest';
import { QueryIntent } from '@/lib/intent';

describe('QueryIntent schema', () => {
  it('accepts a valid fixed-hand intent and defaults metrics to [equity]', () => {
    const parsed = QueryIntent.parse({
      game: 'omahahi5',
      players: [
        { name: 'PLAYER_1', cards: 'AsAhKsKhQs' },
        { name: 'PLAYER_2', cards: 'JdTd9d8d7d' },
      ],
      board: '2c3c4c',
    });
    expect(parsed.metrics).toEqual(['equity']);
    expect(parsed.players).toHaveLength(2);
  });

  it('rejects an unknown game', () => {
    expect(() =>
      QueryIntent.parse({ game: 'razz', players: [{ name: 'PLAYER_1', cards: 'AsKs' }, { name: 'PLAYER_2', cards: 'QdQh' }] }),
    ).toThrow();
  });

  it('accepts a single player (single-hand/range category queries)', () => {
    const parsed = QueryIntent.parse({
      game: 'holdem', players: [{ name: 'PLAYER_1', cards: '10%' }],
      handTypeQuery: { player: 'PLAYER_1', mode: 'min', category: 'twopair' },
    });
    expect(parsed.players).toHaveLength(1);
  });

  it('rejects zero players', () => {
    expect(() => QueryIntent.parse({ game: 'holdem', players: [] })).toThrow();
  });

  it('accepts holdem with two-card hands', () => {
    const parsed = QueryIntent.parse({
      game: 'holdem',
      players: [
        { name: 'PLAYER_1', cards: 'AsKs' },
        { name: 'PLAYER_2', cards: 'QdQh' },
      ],
    });
    expect(parsed.game).toBe('holdem');
  });

  it('accepts a range as a player value', () => {
    const parsed = QueryIntent.parse({
      game: 'holdem',
      players: [{ name: 'PLAYER_1', cards: 'AsKs' }, { name: 'PLAYER_2', cards: 'QQ+' }],
    });
    expect(parsed.players[1].cards).toBe('QQ+');
  });
});
