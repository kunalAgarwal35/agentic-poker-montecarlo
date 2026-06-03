import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RawResultCard } from '@/components/viz/RawResultCard';

const base = {
  query: "select avg(riverEquity(PLAYER_1)) as eq, histogram(handType(PLAYER_1,river)) as dist from game='holdem', board='Ah7c2h', PLAYER_1='KhQh', PLAYER_2='QdQc' where minHandType(PLAYER_1,river,flush)",
  result: {
    values: { eq: 0.991 },
    columns: ['eq', 'dist'],
    trials: 712,
    mode: 'monte_carlo' as const,
    seed: null,
    histograms: { dist: { pairs: [{ value: 5, count: 712 }], labels: { '5': 'Flush' } } },
  },
};

describe('RawResultCard', () => {
  it('shows a scalar formatted as a percentage', () => {
    render(<RawResultCard data={base} />);
    expect(screen.getByText('eq')).toBeInTheDocument();
    expect(screen.getByText(/99\.1%/)).toBeInTheDocument();
  });

  it('renders a histogram bucket with its category label', () => {
    render(<RawResultCard data={base} />);
    expect(screen.getByText('Flush')).toBeInTheDocument();
  });

  it('notes the matching-trial count when a where was used', () => {
    render(<RawResultCard data={base} />);
    expect(screen.getByText('matching trials', { exact: false })).toBeInTheDocument();
  });

  it('shows n/a for a null scalar', () => {
    const d = { ...base, result: { ...base.result, values: { eq: null }, histograms: {} } };
    render(<RawResultCard data={d} />);
    expect(screen.getByText(/n\/a/i)).toBeInTheDocument();
  });
});
