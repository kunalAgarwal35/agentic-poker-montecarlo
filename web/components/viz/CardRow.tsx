'use client';
import { parseCards } from '@/lib/cards';
import { PokerCard } from '@/components/viz/PokerCard';

// A row of cards. `fan` false (default) → a flat, gapped row for board /
// community cards. `fan` true → the classic CSS playing-card fan: all cards are
// absolutely positioned, anchored to the SAME spot (horizontal center, bottom
// of the container) and each rotated by a few degrees around a SINGLE SHARED
// PIVOT well below the cards. They radiate from that one low point — like a hand
// gripped at the bottom — so they overlap into one cohesive arc instead of
// floating at mismatched heights. Board cards must NOT fan.
//
// ── Tuning knobs (presentation-only; the look is driven entirely by these) ──
// The coordinator screenshots the render and tweaks ORIGIN_Y_PCT / STEP_MAX if
// the arch is too tight or too wide. Keep them here, named and together.
//
// Geometry per card i of n (centered on mid = (n-1)/2):
//   step   = min(STEP_MAX, STEP_BUDGET / (n + 1))  — per-card tilt, shrinks as n grows
//   angle  = (i - mid) * step                       — outer cards tilt away from center
//   pivot  = transform-origin 50% ORIGIN_Y_PCT      — a single point below all cards
// Each card is the same absolute element (left:50%, bottom:ANCHOR_BOTTOM_PX),
// pre-centered with translateX(-50%); only the rotation differs, so rotating
// around the shared low pivot fans them out with no per-card translateY.
const STEP_MAX = 8; // cap per-card tilt (deg) so a 2-card hand sits at ±4°, not splayed
const STEP_BUDGET = 30; // spread budget divided by (n+1); wider hands fan gentler
const ORIGIN_Y_PCT = 220; // pivot Y as % of card height → ~1.2× card-height BELOW the card
const ANCHOR_BOTTOM_PX = 6; // lift the whole fan off the container floor a touch

// Card size in px (PokerCard default 'md' = w-9 h-12 = 36×48). Drives the sized
// relative container so the absolute children reserve layout space and the fan
// neither clips its rotated corners nor overlaps the row label / next element.
const CARD_W = 36;
const CARD_H = 48;
const SPREAD_X = 18; // horizontal room each extra card adds to the container width
const ARC_PAD_Y = 14; // extra height for the arc the rotated cards sweep through

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
  // Per-card tilt: gentle for 2 cards, scaled down so wider PLO fans stay tidy.
  const step = Math.min(STEP_MAX, STEP_BUDGET / (n + 1));

  // Sized relative container: fits the fan so the absolutely-positioned children
  // reserve real layout space (no clipping, no overlap of neighbours).
  const width = CARD_W + (n - 1) * SPREAD_X;
  const height = CARD_H + ARC_PAD_Y;

  return (
    <span
      className="relative inline-block align-middle"
      style={{ width, height }}
    >
      {parsed.map((c, i) => {
        const angle = (i - mid) * step;
        return (
          <span
            key={i}
            className="absolute"
            style={{
              left: '50%',
              bottom: ANCHOR_BOTTOM_PX,
              transformOrigin: `50% ${ORIGIN_Y_PCT}%`,
              transform: `translateX(-50%) rotate(${angle}deg)`,
              zIndex: i,
            }}
          >
            <PokerCard rank={c.rank} suit={c.suit} />
          </span>
        );
      })}
    </span>
  );
}
