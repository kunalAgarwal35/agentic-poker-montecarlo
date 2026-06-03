import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock `ai` so we never construct a real stream or call Opus. We only need the
// named exports the route imports; streamText is a spy we assert against.
const streamTextMock = vi.fn((..._args: any[]) => ({
  toUIMessageStreamResponse: () =>
    new Response('stream', { status: 200 }),
}));

vi.mock('ai', () => ({
  streamText: (...args: any[]) => streamTextMock(...args),
  tool: (cfg: any) => cfg,
  convertToModelMessages: async (m: any) => m,
  stepCountIs: () => ({}),
  hasToolCall: () => ({}),
}));

// Mock the rate limiter so we control ok/not-ok and can assert ordering.
const checkRateLimitMock = vi.fn(
  async (..._args: any[]): Promise<{ ok: boolean; status: number; message: string }> => ({
    ok: true,
    status: 0,
    message: '',
  }),
);
vi.mock('@/lib/ratelimit', () => ({
  checkRateLimit: (...args: any[]) => checkRateLimitMock(...args),
  clientIp: () => '1.2.3.4',
}));

// Avoid pulling in the real model/provider (which would want an API key) by
// stubbing the agent model + tool dependencies the route imports.
vi.mock('@/lib/models', () => ({ AGENT_MODEL: 'mock-model' }));
vi.mock('@/lib/prompts', () => ({ AGENT_SYSTEM_PROMPT: 'sys' }));
vi.mock('@/lib/buildQuery', () => ({ executeBuildQuery: vi.fn() }));
vi.mock('@/lib/runPqlTool', () => ({ runPqlTool: vi.fn() }));

import { POST } from '@/app/api/chat/route';

function chatReq(body: any) {
  return new Request('http://test/api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
}

function textMessage(role: string, text: string) {
  return { role, parts: [{ type: 'text', text }] };
}

describe('/api/chat input guards', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    checkRateLimitMock.mockResolvedValue({ ok: true, status: 0, message: '' });
  });

  it('rejects an over-long latest user message with 400 and never calls streamText', async () => {
    const res = await POST(
      chatReq({ messages: [textMessage('user', 'a'.repeat(2001))] }),
    );
    expect(res.status).toBe(400);
    expect(streamTextMock).not.toHaveBeenCalled();
  });

  it('rejects messages.length > 20 with 400 and never calls streamText', async () => {
    const messages = Array.from({ length: 21 }, (_, i) =>
      textMessage('user', `m${i}`),
    );
    const res = await POST(chatReq({ messages }));
    expect(res.status).toBe(400);
    expect(streamTextMock).not.toHaveBeenCalled();
  });

  it('rejects total text over 8000 chars with 400', async () => {
    // 10 messages * 1000 chars = 10000 > 8000, each under per-message limits.
    const messages = Array.from({ length: 10 }, () =>
      textMessage('assistant', 'x'.repeat(1000)),
    );
    const res = await POST(chatReq({ messages }));
    expect(res.status).toBe(400);
    expect(streamTextMock).not.toHaveBeenCalled();
  });

  it('returns the rate-limit status and does NOT call streamText when limited', async () => {
    checkRateLimitMock.mockResolvedValue({
      ok: false,
      status: 503,
      message: "The app has hit today's free usage cap — please try again tomorrow.",
    });
    const res = await POST(chatReq({ messages: [textMessage('user', 'hi')] }));
    expect(res.status).toBe(503);
    const body = await res.json();
    expect(body.error).toMatch(/usage cap/i);
    expect(streamTextMock).not.toHaveBeenCalled();
  });

  it('allows a valid, under-limit request through to streamText', async () => {
    const res = await POST(chatReq({ messages: [textMessage('user', 'hello')] }));
    expect(res.status).toBe(200);
    expect(streamTextMock).toHaveBeenCalledTimes(1);
  });
});
