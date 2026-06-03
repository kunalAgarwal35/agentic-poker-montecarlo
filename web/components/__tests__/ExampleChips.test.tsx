import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ExampleChips } from '@/components/ExampleChips';

describe('ExampleChips', () => {
  it('renders plain-English starters', () => {
    render(<ExampleChips onPick={vi.fn()} />);
    expect(screen.getByRole('button', { name: /AA vs KK preflop, who wins\?/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Flush draw on the turn/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /set/i })).toBeInTheDocument();
  });

  it('fires onPick with the full natural-language prompt when clicked', () => {
    const onPick = vi.fn();
    render(<ExampleChips onPick={onPick} />);
    fireEvent.click(screen.getByRole('button', { name: /AA vs KK preflop, who wins\?/i }));
    expect(onPick).toHaveBeenCalledOnce();
    expect(onPick.mock.calls[0][0]).toBe('AA vs KK preflop, who wins?');
  });

  it('uses no insider labels or pre-stuffed hands', () => {
    render(<ExampleChips onPick={vi.fn()} />);
    expect(screen.queryByText(/PLO5 flop/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Nut frequency/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/AsAhKsKhQs/)).not.toBeInTheDocument();
  });
});
