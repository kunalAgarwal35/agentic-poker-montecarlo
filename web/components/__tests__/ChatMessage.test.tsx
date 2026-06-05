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

  it('renders markdown in an assistant answer (bold headline → <strong>, no literal asterisks)', () => {
    const message = {
      id: 'm3',
      role: 'assistant',
      parts: [{ type: 'text', text: "**You're behind: ~32%** vs 68%." }],
    } as unknown as UIMessage;
    const { container } = render(<ChatMessage message={message} onChoose={() => {}} />);
    const strong = container.querySelector('strong');
    expect(strong).not.toBeNull();
    expect(strong?.textContent).toBe("You're behind: ~32%");
    // The asterisks must NOT survive as literal text.
    expect(container.textContent).not.toContain('**');
  });

  it('keeps USER message text plain (does not strip/parse markdown into elements)', () => {
    const message = {
      id: 'm4',
      role: 'user',
      parts: [{ type: 'text', text: '**not bold** for me' }],
    } as unknown as UIMessage;
    const { container } = render(<ChatMessage message={message} onChoose={() => {}} />);
    expect(container.querySelector('strong')).toBeNull();
    expect(container.textContent).toContain('**not bold** for me');
  });
});
