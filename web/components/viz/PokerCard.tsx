'use client';
import { SUIT_GLYPH, SUIT_COLOR_ON_LIGHT } from '@/lib/cards';

// A real-looking playing card on a light/near-white face so it pops on the dark
// zinc UI. The rank + suit glyph sit tight in the top-left corner (like a real
// card's index), in the 4-color suit color, with a large faint suit glyph in
// the body. `size` defaults to 'md' (~36×48px, ~1.5× the old tiny chip).
const SIZE = {
  sm: { box: 'h-9 w-7', rank: 'text-[11px]', pip: 'text-[9px]', body: 'text-xl' },
  md: { box: 'h-12 w-9', rank: 'text-sm', pip: 'text-[11px]', body: 'text-3xl' },
} as const;

export function PokerCard({
  rank,
  suit,
  size = 'md',
}: {
  rank: string;
  suit: string;
  size?: 'sm' | 'md';
}) {
  const s = suit.toLowerCase();
  const color = SUIT_COLOR_ON_LIGHT[s] ?? '#111827';
  const glyph = SUIT_GLYPH[s] ?? '?';
  const r = rank.toUpperCase();
  const dims = SIZE[size];
  return (
    <span
      className={`relative inline-flex ${dims.box} shrink-0 select-none overflow-hidden rounded-md border border-zinc-300 bg-zinc-50 align-middle shadow-sm`}
      aria-label={`${r}${glyph}`}
    >
      {/* top-left corner index: rank over suit, tight */}
      <span
        className="absolute left-1 top-0.5 flex flex-col items-center leading-none"
        style={{ color }}
      >
        <span className={`${dims.rank} font-bold`}>{r}</span>
        <span className={dims.pip} aria-hidden>{glyph}</span>
      </span>
      {/* faint body glyph, bottom-right, for that real-card weight */}
      <span
        className={`absolute bottom-0 right-0.5 ${dims.body} leading-none opacity-15`}
        style={{ color }}
        aria-hidden
      >
        {glyph}
      </span>
    </span>
  );
}
