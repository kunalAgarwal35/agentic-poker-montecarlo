import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { RecentGallery } from '@/components/RecentGallery';

const items = [
  { id: '1', ts: '2026-06-01T12:00:00.000Z', question: 'How often do I win when I flop a flush?', kind: 'raw',
    payload: { raw: { query: "select avg(riverEquity(PLAYER_1)) as eq from game='holdem', PLAYER_1='KhQh', PLAYER_2='QdQc'", result: { values: { eq: 0.99 }, columns: ['eq'], trials: 100, mode: 'monte_carlo', seed: null, histograms: {} } } } },
];

function mockFetch(data: any) {
  global.fetch = vi.fn(async () => ({ ok: true, json: async () => data })) as any;
}

beforeEach(() => {
  mockFetch({ items });
});

describe('RecentGallery', () => {
  it('fetches and lists recent questions', async () => {
    render(<RecentGallery onPick={() => {}} />);
    await waitFor(() => expect(screen.getByText(/win when I flop a flush/i)).toBeInTheDocument());
  });

  it('expands a card to show the stored result on click', async () => {
    render(<RecentGallery onPick={() => {}} />);
    await waitFor(() => screen.getByText(/win when I flop a flush/i));
    fireEvent.click(screen.getByText(/win when I flop a flush/i));
    await waitFor(() => expect(screen.getByText(/99\.0%/)).toBeInTheDocument());
  });

  it('renders a friendly placeholder when there are no items', async () => {
    mockFetch({ items: [] });
    render(<RecentGallery onPick={() => {}} />);
    await waitFor(() =>
      expect(screen.getByText(/your recent questions will appear here/i)).toBeInTheDocument(),
    );
  });
});
