import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ChoiceCard } from '@/components/ChoiceCard';

describe('ChoiceCard', () => {
  it('renders the question + options and fires onChoose on click', () => {
    const onChoose = vi.fn();
    render(<ChoiceCard question="Which game?" options={['PLO4', 'PLO5']} onChoose={onChoose} />);
    expect(screen.getByText('Which game?')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'PLO5' }));
    expect(onChoose).toHaveBeenCalledWith('PLO5');
  });
});
