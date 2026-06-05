'use client';
import { parseCards } from '@/lib/cards';
import { PokerCard } from '@/components/viz/PokerCard';

// A row of cards. `fan` false (default) → a flat, gapped row for board /
// community cards. `fan` true → the classic playing-card fan, the way a player
// holds hole cards: tilted, overlapping, radiating from a low grip.
//
// ── Geometry: an explicit circular arc ───────────────────────────────────────
// Earlier fans drove the horizontal spread from the rotation (rotate-around-a-
// shared-pivot), which made the spacing depend on sin(angle) and on how the
// per-card step shrank as the hand grew — so wide PLO hands kept clumping. Here
// the HORIZONTAL SPACING IS STRUCTURAL: card i sits at x = (i - mid) * GAP_X,
// dead even for every card count. Each card's tilt and its vertical dip are then
// DERIVED from that x so the cards lie on one circular arc of radius ARC_R whose
// centre is GRIP below the row — a real hand fanned at a low grip:
//   x      = (i - mid) * GAP_X              — even spacing, the invariant
//   angle  = asin(x / ARC_R)               — tilt along the arc's radius
//   dip    = ARC_R * (1 - cos(angle))      — outer cards ride a touch lower
// Right card stacks over left (zIndex i), and every card's rank+suit index lives
// in its TOP-LEFT corner — the one spot the next card never covers — so all
// indices stay legible no matter how tight the overlap. Board cards must NOT fan.
//
// Tuning knobs (presentation-only): the coordinator screenshots and tweaks these.
const CARD_W = 36; // PokerCard 'md' width — the fan always uses a fixed size so geometry is deterministic
const CARD_H = 48; // PokerCard 'md' height
const GAP_X = 19; // even horizontal step between adjacent cards (px). ~half a card → the top-left index always shows
const ARC_R = 130; // radius of the arc the cards sit on (px). Bigger = flatter arc + gentler tilt; smaller = curvier
const ARC_PAD_TOP = 6; // breathing room above the highest (centre) card
const ROT_PAD_X = 16; // side room for the rotated outer cards' corners so the container never clips them

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

  // Pre-compute each card's place on the arc from its even x position.
  const placed = parsed.map((c, i) => {
    const x = (i - mid) * GAP_X; // EVEN spacing — the structural invariant
    const angleRad = Math.asin(Math.max(-0.95, Math.min(0.95, x / ARC_R)));
    const angleDeg = (angleRad * 180) / Math.PI;
    const dip = ARC_R * (1 - Math.cos(angleRad)); // outer cards sit slightly lower
    return { c, x, angleDeg, dip };
  });
  const maxDip = Math.max(...placed.map((p) => p.dip));

  // Sized relative container so the absolutely-positioned cards reserve real
  // layout space (no clipping, no overlap of the row label or next element).
  const width = CARD_W + (n - 1) * GAP_X + 2 * ROT_PAD_X;
  const height = CARD_H + maxDip + ARC_PAD_TOP;

  return (
    <span className="relative inline-block align-middle" style={{ width, height }}>
      {placed.map(({ c, x, angleDeg, dip }, i) => (
        <span
          key={i}
          className="absolute"
          style={{
            left: '50%',
            top: ARC_PAD_TOP,
            transform: `translateX(-50%) translate(${x}px, ${dip}px) rotate(${angleDeg}deg)`,
            zIndex: i,
          }}
        >
          <PokerCard rank={c.rank} suit={c.suit} />
        </span>
      ))}
    </span>
  );
}
