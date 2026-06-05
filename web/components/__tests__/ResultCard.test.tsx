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

  it('renders a verdict headline for a heads-up equity result', () => {
    const { container } = render(<ResultCard resolvedQuery="q" viz={viz} />);
    // Hero is well ahead (83.7%) → "Ahead" verdict + rounded headline %.
    expect(container.textContent).toMatch(/ahead|behind|coinflip/i);
    expect(container.textContent).toContain('Hero ahead');
    expect(screen.getByText('84%')).toBeInTheDocument(); // rounded headline
  });

  it('shows a Behind verdict when the hero is the underdog', () => {
    const behind: VizSpec = { kind: 'equity', mode: 'monte_carlo', trials: 100, rows: [
      { name: 'Hero', equity: 0.32, isHero: true }, { name: 'Villain', equity: 0.68, isHero: false }] };
    const { container } = render(<ResultCard resolvedQuery="q" viz={behind} />);
    expect(container.textContent).toMatch(/behind/i);
  });

  it('skips the verdict for a multiway (>2 player) equity result', () => {
    const multiway: VizSpec = { kind: 'equity', mode: 'monte_carlo', trials: 100, rows: [
      { name: 'Hero', equity: 0.4, isHero: true },
      { name: 'V1', equity: 0.35, isHero: false },
      { name: 'V2', equity: 0.25, isHero: false }] };
    const { container } = render(<ResultCard resolvedQuery="q" viz={multiway} />);
    expect(container.textContent).not.toMatch(/ahead|behind|coinflip/i);
    // The viz itself still renders the players.
    expect(screen.getByText('Hero')).toBeInTheDocument();
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
    expect(container.textContent).toContain('omahahi5'); // game chip
  });

  it('renders no setup header when context is omitted', () => {
    const viz: VizSpec = { kind: 'frequency', mode: 'monte_carlo', trials: 100, label: '≥ flush by the river', pct: 0.5 };
    const { container } = render(<ResultCard resolvedQuery="q" viz={viz} />);
    expect(container.textContent).not.toContain('Board');
  });

  it('renders engine-checked draw pills listing only the true draws', () => {
    const heroDraws = { player: 'PLAYER_1', flushDraw: true, straightDraw: false, oesd: false, gutshot: true, flushOuts: 9, straightOuts: 0 };
    const { container } = render(<ResultCard resolvedQuery="q" viz={viz} heroDraws={heroDraws} />);
    expect(container.textContent).toContain('Engine-checked draws');
    expect(container.textContent).toContain('Flush draw · 9 outs');
    expect(container.textContent).toContain('Gutshot');
    expect(container.textContent).not.toContain('straight draw'); // straight is false → omitted
  });

  it('labels an open-ended straight draw distinctly', () => {
    const heroDraws = { player: 'PLAYER_1', flushDraw: false, straightDraw: true, oesd: true, gutshot: false, flushOuts: 0, straightOuts: 8 };
    const { container } = render(<ResultCard resolvedQuery="q" viz={viz} heroDraws={heroDraws} />);
    expect(container.textContent).toContain('Open-ended straight draw · 8 outs');
  });

  it('omits the draws section when no draw is present', () => {
    const heroDraws = { player: 'PLAYER_1', flushDraw: false, straightDraw: false, oesd: false, gutshot: false, flushOuts: 0, straightOuts: 0 };
    const { container } = render(<ResultCard resolvedQuery="q" viz={viz} heroDraws={heroDraws} />);
    expect(container.textContent).not.toContain('Engine-checked draws');
  });
});
