import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { PokerCard } from '@/components/viz/PokerCard';
import { CardRow } from '@/components/viz/CardRow';
import { SUIT_COLOR } from '@/lib/cards';

describe('card components', () => {
  it('PokerCard renders rank + suit glyph with the 4-color suit color', () => {
    const { container } = render(<PokerCard rank="A" suit="d" />);
    expect(container.textContent).toContain('A');
    expect(container.textContent).toContain('♦');
    // diamonds are blue in the 4-color deck
    expect(container.innerHTML).toContain(SUIT_COLOR.d);
  });

  it('CardRow renders one chip per card', () => {
    const { container } = render(<CardRow cards="AsKhQc" />);
    expect(container.textContent).toContain('A');
    expect(container.textContent).toContain('♠');
    expect(container.textContent).toContain('♥');
    expect(container.textContent).toContain('♣');
  });

  it('CardRow renders nothing for an empty string', () => {
    const { container } = render(<CardRow cards="" />);
    expect(container.firstChild).toBeNull();
  });
});
