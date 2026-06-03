import { describe, it, expect, vi, beforeEach } from 'vitest';
import { track as vercelTrack } from '@vercel/analytics';
import { track } from '@/lib/analytics';

vi.mock('@vercel/analytics', () => ({ track: vi.fn() }));

const mockedVercelTrack = vercelTrack as unknown as ReturnType<typeof vi.fn>;

beforeEach(() => {
  mockedVercelTrack.mockReset();
});

describe('analytics track wrapper', () => {
  it('forwards event + props to the underlying vercel track', () => {
    track('feedback_given', { rating: 'up' });
    expect(mockedVercelTrack).toHaveBeenCalledTimes(1);
    expect(mockedVercelTrack).toHaveBeenCalledWith('feedback_given', { rating: 'up' });
  });

  it('forwards events with no props', () => {
    track('clarifying_shown');
    expect(mockedVercelTrack).toHaveBeenCalledWith('clarifying_shown', undefined);
  });

  it('does not throw when the underlying track throws', () => {
    mockedVercelTrack.mockImplementationOnce(() => {
      throw new Error('boom');
    });
    expect(() => track('tool_used', { tool: 'run_pql' })).not.toThrow();
  });
});
