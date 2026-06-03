'use client';
import { useChat } from '@ai-sdk/react';
import { useRef, useState } from 'react';
import { ChatMessage } from '@/components/ChatMessage';
import { ExampleChips } from '@/components/ExampleChips';
import { Hero } from '@/components/Hero';
import { RecentGallery } from '@/components/RecentGallery';
import { track } from '@/lib/analytics';

const REASON_LABELS: Record<string, string> = {
  misread_cards: 'Misread my hand/board',
  wrong_game: 'Wrong game',
  different_question: 'Answered a different question',
  wrong_numbers: 'The numbers look wrong',
  too_vague: 'Too vague',
  other: 'Something else',
};

function buildRecentBody(question: string, parts: any[]) {
  let body: any = null;
  for (const p of parts) {
    if (p.type === 'tool-run_pql' && p.state === 'output-available' && p.output) {
      body = { question, kind: 'raw', raw: p.output };
    } else if (p.type === 'tool-build_query' && p.state === 'output-available' && p.output?.viz) {
      const viz = p.output.viz;
      body = { question, kind: viz.kind, vizSpecs: [viz] };
    } else if (p.type === 'tool-ask_choice' && p.state === 'output-available' && p.output) {
      body = { question, kind: 'ask_choice', summary: 'Asked: ' + p.output.question };
    }
  }
  if (!body) return null;
  const txt = parts.filter((x) => x.type === 'text').map((x) => x.text).join(' ').trim();
  if (txt) body.summary = txt;
  return body;
}

async function captureRecent(question: string, parts: any[]) {
  const body = buildRecentBody(question, parts);
  if (!body) return;
  try {
    await fetch('/api/recent', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  } catch {}
}

export function Chat() {
  const [input, setInput] = useState('');
  const lastQuestionRef = useRef('');
  const { messages, sendMessage, status, error, regenerate } = useChat({
    onFinish: ({ message }) => {
      captureRecent(lastQuestionRef.current, message.parts as any[]);
      for (const p of (message.parts as any[]) ?? []) {
        if (typeof p?.type !== 'string' || !p.type.startsWith('tool-')) continue;
        const tool = p.type.replace('tool-', '');
        if (p.state === 'output-available') track('tool_used', { tool });
        else if (p.state === 'output-error') track('agent_error', { tool });
      }
    },
  });

  // The text of the most recent USER message before the assistant message at
  // index `idx`. Used to tie a feedback widget / retry to its question.
  function precedingQuestion(idx: number): string {
    for (let i = idx - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role === 'user') {
        return m.parts
          .filter((p: any) => p.type === 'text')
          .map((p: any) => p.text)
          .join(' ')
          .trim();
      }
    }
    return '';
  }

  function onRetry(question: string, reason: string) {
    const reasonLabel = reason ? REASON_LABELS[reason] ?? '' : '';
    const text = `That last answer wasn't quite right${
      reasonLabel ? ' (' + reasonLabel + ')' : ''
    }. Please try again. Original question: ${question}`;
    lastQuestionRef.current = question || text;
    sendMessage({ text });
  }

  return (
    <div className="mx-auto flex h-screen max-w-2xl flex-col p-4">
      <h1 className="mb-3 text-sm font-medium text-zinc-500">agentic-poker-montecarlo</h1>
      <div className="flex flex-1 flex-col gap-3 overflow-y-auto pb-4">
        {messages.length === 0 && (
          <>
            <Hero />
            <ExampleChips onPick={(p) => { lastQuestionRef.current = p; track('question_submitted', { source: 'chip' }); sendMessage({ text: p }); }} />
            <RecentGallery onPick={(p) => { lastQuestionRef.current = p; track('question_submitted', { source: 'recent' }); sendMessage({ text: p }); }} />
          </>
        )}
        {messages.map((m, idx) => (
          <ChatMessage
            key={m.id}
            message={m}
            onChoose={(o) => { lastQuestionRef.current = o; track('choice_selected'); sendMessage({ text: o }); }}
            question={precedingQuestion(idx)}
            onRetry={onRetry}
          />
        ))}
      </div>
      {error && (
        <div className="mb-2 rounded-lg border border-red-800 bg-red-950/40 p-2 text-sm text-red-200">
          Something went wrong contacting the agent.{' '}
          <button type="button" onClick={() => regenerate()} className="underline hover:text-red-100">Retry</button>
        </div>
      )}
      <form
        onSubmit={(e) => { e.preventDefault(); if (!input.trim()) return; lastQuestionRef.current = input; track('question_submitted', { source: 'input' }); sendMessage({ text: input }); setInput(''); }}
        className="flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.currentTarget.value)}
          placeholder="Ask a Hold'em or PLO question…"
          className="flex-1 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 caret-green-500 focus:border-zinc-500 focus:outline-none"
        />
        <button disabled={status !== 'ready'} className="rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50">
          Send
        </button>
      </form>
    </div>
  );
}
