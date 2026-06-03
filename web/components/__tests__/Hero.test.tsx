import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Hero } from '@/components/Hero';

describe('Hero', () => {
  it('renders the headline value prop', () => {
    render(<Hero />);
    expect(screen.getByRole('heading', { name: /Ask any poker spot\. Get the math\./i })).toBeInTheDocument();
  });

  it('renders the supported-games line', () => {
    const { container } = render(<Hero />);
    expect(container.textContent).toContain("Texas Hold'em");
    expect(container.textContent).toContain('PLO');
    expect(container.textContent).toMatch(/Hi-lo, stud, and razz aren't supported/);
  });
});
