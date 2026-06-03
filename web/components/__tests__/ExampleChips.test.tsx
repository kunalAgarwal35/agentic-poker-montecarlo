import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ExampleChips } from '@/components/ExampleChips';

describe('ExampleChips', () => {
  it('renders chips and fires onPick with the prompt when clicked', () => {
    const onPick = vi.fn();
    render(<ExampleChips onPick={onPick} />);
    const holdem = screen.getByRole('button', { name: /Holdem preflop/i });
    fireEvent.click(holdem);
    expect(onPick).toHaveBeenCalledOnce();
    expect(onPick.mock.calls[0][0]).toMatch(/holdem/i);
  });
});
