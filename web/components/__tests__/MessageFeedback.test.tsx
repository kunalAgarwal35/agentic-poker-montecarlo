import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MessageFeedback } from '@/components/MessageFeedback';
import { track } from '@/lib/analytics';

vi.mock('@/lib/analytics', () => ({ track: vi.fn() }));

const mockedTrack = track as unknown as ReturnType<typeof vi.fn>;

function lastFeedbackBody(): any {
  const fetchMock = global.fetch as unknown as ReturnType<typeof vi.fn>;
  const calls = fetchMock.mock.calls.filter((c) => c[0] === '/api/feedback');
  const last = calls[calls.length - 1];
  return JSON.parse(last[1].body as string);
}

beforeEach(() => {
  mockedTrack.mockReset();
  global.fetch = vi.fn(async () => ({ ok: true, json: async () => ({ ok: true, id: 'x' }) })) as any;
});

describe('MessageFeedback', () => {
  it('posts rating "up" and shows a thanks message on 👍', async () => {
    render(<MessageFeedback question="q1" answerSummary="a1" />);
    fireEvent.click(screen.getByRole('button', { name: /yes, this helped/i }));

    await waitFor(() => expect(screen.getByText(/thanks/i)).toBeInTheDocument());
    expect(global.fetch).toHaveBeenCalledWith('/api/feedback', expect.objectContaining({ method: 'POST' }));
    const body = lastFeedbackBody();
    expect(body.rating).toBe('up');
    expect(body.question).toBe('q1');
    expect(body.sessionId).toBeTruthy();
    expect(mockedTrack).toHaveBeenCalledWith('feedback_given', { rating: 'up', reason: '' });
  });

  it('reveals reason chips on 👎, then posts rating "down" with reason + comment on Submit', async () => {
    render(<MessageFeedback question="q1" answerSummary="a1" />);
    fireEvent.click(screen.getByRole('button', { name: /no, this missed/i }));

    // chips revealed
    const chip = await screen.findByRole('button', { name: 'The numbers look wrong' });
    fireEvent.click(chip);
    fireEvent.change(screen.getByLabelText(/feedback details/i), { target: { value: 'eq is off' } });
    fireEvent.click(screen.getByRole('button', { name: 'Submit' }));

    await waitFor(() => expect(screen.getByText(/thanks/i)).toBeInTheDocument());
    const body = lastFeedbackBody();
    expect(body.rating).toBe('down');
    expect(body.reason).toBe('wrong_numbers');
    expect(body.comment).toBe('eq is off');
    expect(mockedTrack).toHaveBeenCalledWith('feedback_given', { rating: 'down', reason: 'wrong_numbers' });
  });

  it('"Want me to try again?" calls onRetry with (question, reason) and also posts', async () => {
    const onRetry = vi.fn();
    render(<MessageFeedback question="q1" answerSummary="a1" onRetry={onRetry} />);
    fireEvent.click(screen.getByRole('button', { name: /no, this missed/i }));

    fireEvent.click(await screen.findByRole('button', { name: 'Wrong game' }));
    fireEvent.click(screen.getByRole('button', { name: /want me to try again/i }));

    await waitFor(() => expect(onRetry).toHaveBeenCalledWith('q1', 'wrong_game'));
    const body = lastFeedbackBody();
    expect(body.rating).toBe('down');
    expect(body.reason).toBe('wrong_game');
  });
});
