'use client';
import { parseCards } from '@/lib/cards';
import { PokerCard } from '@/components/viz/PokerCard';

export function CardRow({ cards }: { cards: string }) {
  const parsed = parseCards(cards);
  if (parsed.length === 0) return null;
  return (
    <span className="inline-flex flex-wrap gap-1 align-middle">
      {parsed.map((c, i) => <PokerCard key={i} rank={c.rank} suit={c.suit} />)}
    </span>
  );
}
