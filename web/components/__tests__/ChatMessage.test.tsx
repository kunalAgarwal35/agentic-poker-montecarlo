import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ChatMessage } from '@/components/ChatMessage';
import type { UIMessage } from 'ai';

describe('ChatMessage', () => {
  it('renders an inline error when a tool part is in output-error state', () => {
    const message = {
      id: 'm1',
      role: 'assistant',
      parts: [
        { type: 'tool-build_query', state: 'output-error', toolCallId: 't1', input: {}, errorText: 'Engine error: Duplicate card(s) ... As' },
      ],
    } as unknown as UIMessage;
    render(<ChatMessage message={message} onChoose={() => {}} />);
    expect(screen.getByText(/Duplicate card/)).toBeInTheDocument();
  });

  it('renders a text part as a bubble', () => {
    const message = { id: 'm2', role: 'assistant', parts: [{ type: 'text', text: 'Which Omaha variant?' }] } as unknown as UIMessage;
    render(<ChatMessage message={message} onChoose={() => {}} />);
    expect(screen.getByText('Which Omaha variant?')).toBeInTheDocument();
  });
});
