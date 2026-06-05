import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ResultCard } from '@/components/ResultCard';
import type { VizSpec } from '@/lib/types';

const viz: VizSpec = {
  kind: 'equity',
  mode: 'enumeration',
  trials: 741,
  rows: [
    { name: 'Hero', equity: 0.8367, isHero: true },
    { name: 'Villain', equity: 0.1633, isHero: false },
  ],
};

describe('ResultCard', () => {
  it('renders each player, the equity %, and the mode badge', () => {
    render(<ResultCard resolvedQuery="select avg(riverEquity(PLAYER_1)) as p1 from ..." viz={viz} />);
    expect(screen.getByText('Hero')).toBeInTheDocument();
    expect(screen.getByText('Villain')).toBeInTheDocument();
    expect(screen.getByText('83.7%')).toBeInTheDocument();
    expect(screen.getByText(/Exact · 741 runouts/)).toBeInTheDocument(); // mode badge
  });

  it('hides the resolved query until the disclosure is clicked', () => {
    render(<ResultCard resolvedQuery="select avg(riverEquity(PLAYER_1)) as p1 from ..." viz={viz} />);
    expect(screen.queryByText(/riverEquity/)).not.toBeInTheDocument(); // collapsed by default
    fireEvent.click(screen.getByText(/Show query/i));
    expect(screen.getByText(/riverEquity/)).toBeInTheDocument();        // now visible
  });

  it('renders a board row and per-player cards / range chip from context', () => {
    const viz: VizSpec = { kind: 'equity', mode: 'monte_carlo', trials: 100, rows: [
      { name: 'P1', equity: 0.6, isHero: true }, { name: 'P2', equity: 0.4, isHero: false }] };
    const context = { game: 'omahahi5', board: 'Kh7d2c', players: [
      { name: 'PLAYER_1', cards: 'AsAhKsKhQs' }, { name: 'PLAYER_2', cards: '25%' }] };
    const { container } = render(<ResultCard resolvedQuery="q" viz={viz} context={context} />);
    expect(container.textContent).toContain('Board');
    expect(container.textContent).toContain('♥');        // Kh in the board
    expect(container.textContent).toContain('Top 25%');  // range chip for P2
  });

  it('renders no setup header when context is omitted', () => {
    const viz: VizSpec = { kind: 'frequency', mode: 'monte_carlo', trials: 100, label: '≥ flush by the river', pct: 0.5 };
    const { container } = render(<ResultCard resolvedQuery="q" viz={viz} />);
    expect(container.textContent).not.toContain('Board');
  });

  it('renders an engine-checked draws line listing only the true draws', () => {
    const heroDraws = { player: 'PLAYER_1', flushDraw: true, straightDraw: false, oesd: false, gutshot: true, flushOuts: 9, straightOuts: 0 };
    const { container } = render(<ResultCard resolvedQuery="q" viz={viz} heroDraws={heroDraws} />);
    expect(container.textContent).toContain('Engine-checked draws');
    expect(container.textContent).toContain('flush draw (9 outs)');
    expect(container.textContent).toContain('gutshot');
    expect(container.textContent).not.toContain('OESD'); // oesd is false → omitted
  });

  it('omits the draws line when no draw is present', () => {
    const heroDraws = { player: 'PLAYER_1', flushDraw: false, straightDraw: false, oesd: false, gutshot: false, flushOuts: 0, straightOuts: 0 };
    const { container } = render(<ResultCard resolvedQuery="q" viz={viz} heroDraws={heroDraws} />);
    expect(container.textContent).not.toContain('Engine-checked draws');
  });
});
