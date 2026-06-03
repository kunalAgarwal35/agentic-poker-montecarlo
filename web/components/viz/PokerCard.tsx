'use client';
import { SUIT_GLYPH, SUIT_COLOR } from '@/lib/cards';

export function PokerCard({ rank, suit }: { rank: string; suit: string }) {
  const s = suit.toLowerCase();
  return (
    <span
      className="inline-flex h-7 w-6 shrink-0 flex-col items-center justify-center rounded-[3px] border border-zinc-700 bg-zinc-800 leading-none"
      style={{ color: SUIT_COLOR[s] ?? '#e5e7eb' }}
    >
      <span className="text-xs font-semibold">{rank.toUpperCase()}</span>
      <span className="text-[10px]" aria-hidden>{SUIT_GLYPH[s] ?? '?'}</span>
    </span>
  );
}
