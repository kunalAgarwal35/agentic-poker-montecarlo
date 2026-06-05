import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';

// Controllable useChat mock: tests set `mockState.messages` and inspect
// `mockState.setMessages` (a spy). Everything else is a harmless stub.
const mockState: {
  messages: any[];
  setMessages: ReturnType<typeof vi.fn>;
  sendMessage: ReturnType<typeof vi.fn>;
} = {
  messages: [],
  setMessages: vi.fn(),
  sendMessage: vi.fn(),
};

vi.mock('@ai-sdk/react', () => ({
  useChat: () => ({
    messages: mockState.messages,
    sendMessage: mockState.sendMessage,
    status: 'ready',
    error: null,
    regenerate: vi.fn(),
    setMessages: mockState.setMessages,
  }),
}));

// Avoid network/IO from the gallery + analytics in the empty state.
vi.mock('@/components/RecentGallery', () => ({ RecentGallery: () => null }));
vi.mock('@/lib/analytics', () => ({ track: vi.fn() }));

import { Chat } from '@/components/Chat';

const userMessage = {
  id: 'u1',
  role: 'user',
  parts: [{ type: 'text', text: 'AsKs vs QdQc?' }],
};

beforeEach(() => {
  mockState.messages = [];
  mockState.setMessages.mockClear();
  mockState.sendMessage.mockClear();
});

describe('Chat — New chat reset', () => {
  it('does NOT show the "New chat" button in the empty start state', () => {
    mockState.messages = [];
    render(<Chat />);
    expect(screen.queryByRole('button', { name: '+ New chat' })).toBeNull();
  });

  it('shows a "New chat" button once there are messages', () => {
    mockState.messages = [userMessage];
    render(<Chat />);
    expect(screen.getByRole('button', { name: '+ New chat' })).toBeInTheDocument();
  });

  it('clicking "New chat" resets the conversation via setMessages([])', () => {
    mockState.messages = [userMessage];
    render(<Chat />);
    fireEvent.click(screen.getByRole('button', { name: '+ New chat' }));
    expect(mockState.setMessages).toHaveBeenCalledTimes(1);
    expect(mockState.setMessages).toHaveBeenCalledWith([]);
  });

  it('clicking the brand also resets the conversation', () => {
    mockState.messages = [userMessage];
    render(<Chat />);
    fireEvent.click(screen.getByRole('button', { name: /agentic-poker-montecarlo/i }));
    expect(mockState.setMessages).toHaveBeenCalledWith([]);
  });
});
