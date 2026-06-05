'use client';
import { parseCards } from '@/lib/cards';
import { PokerCard } from '@/components/viz/PokerCard';

// A row of cards. `fan` false (default) → a flat, gapped row for board /
// community cards. `fan` true → the cards overlap and tilt gently and
// symmetrically around their bottom center, like a hand a player holds
// (2-card holdem, or a 4–6 card PLO fan). Board cards must NOT fan.
//
// Fan geometry per card i of n (centered on (n-1)/2):
//   step  = min(STEP_MAX, TOTAL_CAP / (n-1)) — per-card tilt, shrinks as n grows
//   angle = (i - mid) * step                  — outer cards tilt away from center
//   lift  = |i - mid| * LIFT_PX               — a slight arc; flat baseline reads clean
//   overlap: a small negative left margin so each card's top-left index (rank +
//   suit) stays fully visible and uncovered by the next card on top of it.
//
// Conservative on purpose: a subtle held-hand tilt that stays tidy at 2 cards
// (Hold'em, ±6°) and gentler still at 4–6 cards (PLO).
const STEP_MAX = 6; // cap the per-card tilt so a 2-card hand sits at ±6°
const TOTAL_CAP = 14; // total spread budget; divided across the gaps for wider hands
const LIFT_PX = 1.5; // a faint arc — outer cards rise just slightly
const OVERLAP_PX = -8; // cards on a ~36px-wide card; keeps the top-left index readable

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
  // Per-card tilt: small and fixed for 2 cards, scaled down so wider PLO fans
  // stay gentle (4 cards ≈ ±7° total per side, 6 cards even less).
  const step = n > 1 ? Math.min(STEP_MAX, TOTAL_CAP / (n - 1)) : 0;

  return (
    // a little padding so the rotated outer corners aren't clipped; stays compact
    <span className="inline-flex items-end px-1.5 pt-1.5 align-middle">
      {parsed.map((c, i) => {
        const offset = i - mid;
        const angle = offset * step;
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
