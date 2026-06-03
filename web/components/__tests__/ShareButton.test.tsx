import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { track } from '@/lib/analytics';
import { ShareButton } from '@/components/ShareButton';

vi.mock('@/lib/analytics', () => ({ track: vi.fn() }));

describe('ShareButton', () => {
  const writeText = vi.fn().mockResolvedValue(undefined);

  beforeEach(() => {
    vi.clearAllMocks();
    // No navigator.share -> falls back to clipboard copy.
    delete (navigator as any).share;
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    });
  });

  afterEach(() => {
    delete (navigator as any).clipboard;
  });

  it('copies a branded summary containing the app URL and shows Copied!', async () => {
    render(<ShareButton summary="Poker equity — Hero 83.7% vs Villain 16.3%" />);
    fireEvent.click(screen.getByRole('button', { name: /share/i }));

    await waitFor(() => expect(writeText).toHaveBeenCalledOnce());
    const copied = writeText.mock.calls[0][0] as string;
    expect(copied).toContain('Poker equity — Hero 83.7% vs Villain 16.3%');
    expect(copied).toContain('https://agentic-poker-montecarlo.vercel.app');

    await waitFor(() => expect(screen.getByText('Copied!')).toBeInTheDocument());
  });

  it('fires the result_shared analytics event', async () => {
    render(<ShareButton summary="x" />);
    fireEvent.click(screen.getByRole('button', { name: /share/i }));
    await waitFor(() => expect(writeText).toHaveBeenCalled());
    expect(track).toHaveBeenCalledWith('result_shared');
  });

  it('falls back to a generic message when no summary is given', async () => {
    render(<ShareButton />);
    fireEvent.click(screen.getByRole('button', { name: /share/i }));
    await waitFor(() => expect(writeText).toHaveBeenCalled());
    expect(writeText.mock.calls[0][0]).toContain('Check out this poker equity result');
  });
});
