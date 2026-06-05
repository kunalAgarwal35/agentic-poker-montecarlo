import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ResultViz } from '@/components/viz/ResultViz';
import { EquityDistributionChart } from '@/components/viz/EquityDistributionChart';
import { EquityVsClassChart } from '@/components/viz/EquityVsClassChart';
import type { VizSpec } from '@/lib/types';
import type { VizSpec as VS } from '@/lib/types';

describe('ResultViz', () => {
  it('renders an equity comparison summary', () => {
    const spec: VizSpec = { kind: 'equity', mode: 'monte_carlo', trials: 5000, rows: [
      { name: 'Hero', equity: 0.46, isHero: true }, { name: 'Villain', equity: 0.54, isHero: false }] };
    render(<ResultViz spec={spec} />);
    expect(screen.getByText('Hero')).toBeInTheDocument();
    expect(screen.getByText('46.0%')).toBeInTheDocument();
  });

  it('lists the full player name + equity in the equity legend', () => {
    const spec: VizSpec = { kind: 'equity', mode: 'monte_carlo', trials: 100, rows: [
      { name: 'PLAYER_1', equity: 0.6, isHero: true }, { name: 'PLAYER_2', equity: 0.4, isHero: false }] };
    const { container } = render(<ResultViz spec={spec} />);
    // Legend keeps the full readable name + the precise %.
    expect(container.textContent).toContain('PLAYER_1');
    expect(container.textContent).toContain('60.0%');
  });

  it('renders a win-tie-loss summary', () => {
    const spec: VizSpec = { kind: 'win-tie-loss', mode: 'monte_carlo', trials: 100, rows: [
      { name: 'Hero', win: 0.8, tie: 0.04, loss: 0.16 }] };
    render(<ResultViz spec={spec} />);
    expect(screen.getByText('Hero')).toBeInTheDocument();
    expect(screen.getByText(/Win 80\.0%/)).toBeInTheDocument();
  });

  it('renders an equity distribution histogram with the mean', () => {
    const buckets = Array.from({ length: 10 }, (_, b) => ({ lo: b * 10, hi: b * 10 + 10, pct: b === 9 ? 1 : 0 }));
    const { container } = render(
      <EquityDistributionChart player="PLAYER_1" mean={0.95} combos={120} sampledCombos={null} buckets={buckets} />,
    );
    expect(container.textContent).toContain('95.0%');   // mean rendered as a percentage
  });

  it('equity-by-street renders a Y axis and tooltip layer', () => {
    // Recharts' <ResponsiveContainer> only renders its chart (and SVG axes) once it
    // observes a positive size. happy-dom reports zero-size boxes and never fires
    // ResizeObserver, so we install a fixed-size shim for the duration of this test
    // only (restored in finally) to keep the other text-based tests unaffected.
    const origRO = globalThis.ResizeObserver;
    const origW = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'offsetWidth');
    const origH = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'offsetHeight');
    class ResizeObserverMock {
      cb: ResizeObserverCallback;
      constructor(cb: ResizeObserverCallback) { this.cb = cb; }
      observe(target: Element) {
        this.cb([{ target, contentRect: { width: 400, height: 150 } } as ResizeObserverEntry],
          this as unknown as ResizeObserver);
      }
      unobserve() {}
      disconnect() {}
    }
    globalThis.ResizeObserver = ResizeObserverMock as unknown as typeof ResizeObserver;
    Object.defineProperty(HTMLElement.prototype, 'offsetWidth', { configurable: true, value: 400 });
    Object.defineProperty(HTMLElement.prototype, 'offsetHeight', { configurable: true, value: 150 });
    try {
      const spec: VizSpec = { kind: 'equity-by-street', mode: 'monte_carlo', trials: 100, series: [
        { name: 'AsKh', points: [{ street: 'flop', equity: 0.6 }, { street: 'river', equity: 1 }] }] };
      const { container } = render(<ResultViz spec={spec} />);
      // Recharts renders the cartesian Y axis as an element with class "recharts-yAxis"
      expect(container.querySelector('.recharts-yAxis')).toBeTruthy();
    } finally {
      globalThis.ResizeObserver = origRO;
      if (origW) Object.defineProperty(HTMLElement.prototype, 'offsetWidth', origW);
      else delete (HTMLElement.prototype as unknown as Record<string, unknown>).offsetWidth;
      if (origH) Object.defineProperty(HTMLElement.prototype, 'offsetHeight', origH);
      else delete (HTMLElement.prototype as unknown as Record<string, unknown>).offsetHeight;
    }
  });

  it('renders equity-vs-class rows', () => {
    const rows = [
      { category: 'pair', label: 'One Pair', equity: 0.72, freq: 0.6 },
      { category: 'flush', label: 'Flush', equity: 0.12, freq: 0.4 },
    ];
    const { container } = render(<EquityVsClassChart player="PLAYER_1" rows={rows} />);
    expect(container.textContent).toContain('One Pair');
    expect(container.textContent).toContain('72%');
  });
});

it('renders the distribution histogram summary', () => {
  const spec: VS = { kind: 'distribution', mode: 'monte_carlo', trials: 100, player: 'Hero',
    bars: [{ token: 'pair', label: 'Pair', freq: 0.5 }, { token: 'flush', label: 'Flush', freq: 0.1 }] };
  render(<ResultViz spec={spec} />);
  expect(screen.getByText(/Pair/)).toBeInTheDocument();
  expect(screen.getByText(/50\.0%/)).toBeInTheDocument();
});

it('renders a frequency stat', () => {
  const spec: VizSpec = { kind: 'frequency', mode: 'monte_carlo', trials: 100, label: 'Hero holds the nuts', pct: 0.123 };
  render(<ResultViz spec={spec} />);
  expect(screen.getByText('Hero holds the nuts')).toBeInTheDocument();
  expect(screen.getByText('12.3%')).toBeInTheDocument();
});
