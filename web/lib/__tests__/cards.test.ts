import { describe, it, expect } from 'vitest';
import { parseCards, isConcreteHand, rangeLabel } from '@/lib/cards';

describe('cards', () => {
  it('parseCards splits a card string into rank/suit pairs', () => {
    expect(parseCards('AsAhKsKhQs')).toEqual([
      { rank: 'A', suit: 's' }, { rank: 'A', suit: 'h' }, { rank: 'K', suit: 's' },
      { rank: 'K', suit: 'h' }, { rank: 'Q', suit: 's' },
    ]);
    expect(parseCards('')).toEqual([]);
  });

  it('isConcreteHand distinguishes hands from ranges', () => {
    expect(isConcreteHand('AsKs')).toBe(true);
    expect(isConcreteHand('AsAhKsKhQs')).toBe(true);
    expect(isConcreteHand('25%')).toBe(false);
    expect(isConcreteHand('QQ+')).toBe(false);
    expect(isConcreteHand('AAxx')).toBe(false);
  });

  it('rangeLabel humanizes a bare percent, passes others through', () => {
    expect(rangeLabel('25%')).toBe('Top 25%');
    expect(rangeLabel('QQ+')).toBe('QQ+');
    expect(rangeLabel('AAxx')).toBe('AAxx');
  });
});
