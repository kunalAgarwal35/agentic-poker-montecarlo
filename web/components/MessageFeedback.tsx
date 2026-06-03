'use client';
import { useState } from 'react';
import { getSessionId } from '@/lib/session';
import { track } from '@/lib/analytics';

export const FEEDBACK_REASONS: { value: string; label: string }[] = [
  { value: 'misread_cards', label: 'Misread my hand/board' },
  { value: 'wrong_game', label: 'Wrong game' },
  { value: 'different_question', label: 'Answered a different question' },
  { value: 'wrong_numbers', label: 'The numbers look wrong' },
  { value: 'too_vague', label: 'Too vague' },
  { value: 'other', label: 'Something else' },
];

type Phase = 'ask' | 'reasons' | 'done';

export function MessageFeedback({
  question,
  answerSummary,
  meta,
  onRetry,
}: {
  question?: string;
  answerSummary?: string;
  meta?: Record<string, unknown>;
  onRetry?: (question: string, reason: string) => void;
}) {
  const [phase, setPhase] = useState<Phase>('ask');
  const [showNote, setShowNote] = useState(false);
  const [note, setNote] = useState('');
  const [reason, setReason] = useState<string | null>(null);
  const [doneMsg, setDoneMsg] = useState('Thanks for the feedback!');

  async function submit(rating: 'up' | 'down', r?: string | null, comment?: string) {
    try {
      await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          question,
          answerSummary,
          rating,
          reason: r ?? undefined,
          comment: comment?.trim() ? comment.trim() : undefined,
          sessionId: getSessionId(),
          meta,
        }),
      });
    } catch {
      // best-effort — never block the UI
    }
    track('feedback_given', { rating, reason: r ?? '' });
  }

  if (phase === 'done') {
    return (
      <div className="mt-2 border-t border-zinc-800 pt-2 text-xs text-zinc-500">{doneMsg}</div>
    );
  }

  if (phase === 'reasons') {
    return (
      <div className="mt-2 border-t border-zinc-800 pt-2 text-xs text-zinc-400">
        <div className="mb-1.5 text-zinc-400">Where did I go wrong?</div>
        <div className="mb-2 flex flex-wrap gap-1.5">
          {FEEDBACK_REASONS.map((opt) => {
            const active = reason === opt.value;
            return (
              <button
                key={opt.value}
                type="button"
                aria-pressed={active}
                onClick={() => setReason(active ? null : opt.value)}
                className={`rounded-full border px-2 py-0.5 text-xs transition-colors ${
                  active
                    ? 'border-zinc-500 bg-zinc-700 text-zinc-100'
                    : 'border-zinc-700 bg-zinc-800/60 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200'
                }`}
              >
                {opt.label}
              </button>
            );
          })}
        </div>
        <textarea
          value={note}
          onChange={(e) => setNote(e.currentTarget.value)}
          placeholder="Add details (optional)…"
          aria-label="Feedback details"
          rows={2}
          className="mb-2 w-full resize-none rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200 placeholder:text-zinc-600 focus:border-zinc-500 focus:outline-none"
        />
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={async () => {
              await submit('down', reason, note);
              setDoneMsg('Thanks — this helps.');
              setPhase('done');
            }}
            className="rounded-lg border border-zinc-700 bg-zinc-800 px-2.5 py-1 text-xs text-zinc-200 hover:border-zinc-600 hover:bg-zinc-700"
          >
            Submit
          </button>
          <button
            type="button"
            onClick={async () => {
              await submit('down', reason, note);
              onRetry?.(question ?? '', reason ?? '');
              setDoneMsg('Thanks — asking again…');
              setPhase('done');
            }}
            className="rounded-lg border border-green-700/60 bg-green-900/30 px-2.5 py-1 text-xs text-green-300 hover:border-green-600 hover:bg-green-900/50"
          >
            Want me to try again?
          </button>
        </div>
      </div>
    );
  }

  // phase === 'ask'
  return (
    <div className="mt-2 border-t border-zinc-800 pt-2 text-xs text-zinc-500">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-zinc-400">Did this answer your question?</span>
        <button
          type="button"
          aria-label="Yes, this helped"
          onClick={async () => {
            await submit('up', undefined, note);
            setDoneMsg('Thanks for the feedback!');
            setPhase('done');
          }}
          className="rounded-md border border-zinc-700 bg-zinc-800/60 px-2 py-0.5 text-zinc-300 hover:border-green-600 hover:text-green-400"
        >
          👍 Yes
        </button>
        <button
          type="button"
          aria-label="No, this missed"
          onClick={() => setPhase('reasons')}
          className="rounded-md border border-zinc-700 bg-zinc-800/60 px-2 py-0.5 text-zinc-300 hover:border-zinc-500 hover:text-zinc-100"
        >
          👎 No
        </button>
        <button
          type="button"
          onClick={() => setShowNote((s) => !s)}
          className="text-zinc-500 underline-offset-2 hover:text-zinc-300 hover:underline"
        >
          Add a note
        </button>
      </div>
      {showNote && (
        <textarea
          value={note}
          onChange={(e) => setNote(e.currentTarget.value)}
          placeholder="Anything you'd like to add…"
          aria-label="Add a note"
          rows={2}
          className="mt-2 w-full resize-none rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200 placeholder:text-zinc-600 focus:border-zinc-500 focus:outline-none"
        />
      )}
    </div>
  );
}
