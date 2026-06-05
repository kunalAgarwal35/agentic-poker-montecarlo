'use client';
import { parseCards } from '@/lib/cards';
import { PokerCard } from '@/components/viz/PokerCard';

// A row of cards. `fan` false (default) → a flat, gapped row for board /
// community cards. `fan` true → the cards overlap and tilt symmetrically around
// their bottom center, like a hand a player holds (2-card holdem, or a wider
// PLO fan). Board cards must NOT fan.
//
// Fan geometry per card i of n (centered on (n-1)/2):
//   angle = (i - mid) * spread   — outer cards tilt away from center
//   lift  = |i - mid| * LIFT     — outer cards rise to arc the hand
//   overlap: negative left margin after the first card so they overlap ~30%.
const SPREAD_DEG = 12; // per-card tilt for a tight 2-card hand
const MAX_TOTAL_DEG = 32; // cap the total spread so wide (PLO) fans stay tasteful
const LIFT_PX = 3;
const OVERLAP_PX = -10;

export function CardRow({ cards, fan = false }: { cards: string; fan?: boolean }) {
  const parsed = parseCards(cards);
  if (parsed.length === 0) return null;

  if (!fan) {
    return (
      <span className="inline-flex flex-wrap items-center gap-1 align-middle">
        {parsed.map((c, i) => <PokerCard key={i} rank={c.rank} suit={c.suit} />)}
      </span>
    );
  }

  const n = parsed.length;
  const mid = (n - 1) / 2;
  // Scale the per-card spread down for wider hands so the total fan stays capped.
  const spread = n > 1 ? Math.min(SPREAD_DEG, MAX_TOTAL_DEG / (n - 1)) : 0;

  return (
    // extra padding/height so the rotated outer corners don't clip
    <span className="inline-flex items-end px-2 pt-2 align-middle">
      {parsed.map((c, i) => {
        const offset = i - mid;
        const angle = offset * spread;
        const lift = Math.abs(offset) * LIFT_PX;
        return (
          <span
            key={i}
            className="inline-block"
            style={{
              transform: `rotate(${angle}deg) translateY(-${lift}px)`,
              transformOrigin: 'bottom center',
              marginLeft: i === 0 ? 0 : OVERLAP_PX,
            }}
          >
            <PokerCard rank={c.rank} suit={c.suit} />
          </span>
        );
      })}
    </span>
  );
}
