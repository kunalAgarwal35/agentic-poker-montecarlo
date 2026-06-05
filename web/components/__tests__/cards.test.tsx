import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { PokerCard } from '@/components/viz/PokerCard';
import { CardRow } from '@/components/viz/CardRow';
import { SUIT_COLOR_ON_LIGHT } from '@/lib/cards';

describe('card components', () => {
  it('PokerCard renders rank + suit glyph with the 4-color (on-light) suit color', () => {
    const { container } = render(<PokerCard rank="A" suit="d" />);
    expect(container.textContent).toContain('A');
    expect(container.textContent).toContain('♦');
    // diamonds are blue in the 4-color deck (tuned for the light card face)
    expect(container.innerHTML).toContain(SUIT_COLOR_ON_LIGHT.d);
  });

  it('PokerCard spades use a dark color so they read on the light face', () => {
    const { container } = render(<PokerCard rank="A" suit="s" />);
    expect(container.textContent).toContain('♠');
    expect(container.innerHTML).toContain(SUIT_COLOR_ON_LIGHT.s);
  });

  it('CardRow renders one card per card (flat row)', () => {
    const { container } = render(<CardRow cards="AsKhQc" />);
    expect(container.textContent).toContain('A');
    expect(container.textContent).toContain('♠');
    expect(container.textContent).toContain('♥');
    expect(container.textContent).toContain('♣');
  });

  it('CardRow with fan renders all cards, each with a rotate transform', () => {
    const { container } = render(<CardRow cards="AsKh" fan />);
    expect(container.textContent).toContain('A');
    expect(container.textContent).toContain('K');
    // both cards are present and the fan applies a rotate() transform
    const transformed = Array.from(
      container.querySelectorAll<HTMLElement>('[style*="rotate"]'),
    );
    expect(transformed.length).toBe(2);
    // two-card hand: left card tilts one way, right card the other
    expect(transformed[0].style.transform).toContain('rotate(-');
    expect(transformed[1].style.transform).toContain('rotate(');
  });

  it('CardRow renders nothing for an empty string', () => {
    const { container } = render(<CardRow cards="" />);
    expect(container.firstChild).toBeNull();
  });
});
