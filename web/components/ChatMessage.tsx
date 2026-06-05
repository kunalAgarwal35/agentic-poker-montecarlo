'use client';
import { useEffect } from 'react';
import type { UIMessage } from 'ai';
import { ResultCard } from '@/components/ResultCard';
import { AnswerMarkdown, stripPreamble } from '@/components/AnswerMarkdown';
import { ChoiceCard } from '@/components/ChoiceCard';
import { RawResultCard } from '@/components/viz/RawResultCard';
import { MessageFeedback } from '@/components/MessageFeedback';
import { track } from '@/lib/analytics';
import type { BuildQueryResult } from '@/lib/buildQuery';
import type { RawResult } from '@/lib/types';

export function ChatMessage({
  message,
  onChoose,
  question,
  onRetry,
}: {
  message: UIMessage;
  onChoose: (o: string) => void;
  question?: string;
  onRetry?: (q: string, reason: string) => void;
}) {
  const isUser = message.role === 'user';

  // An assistant turn counts as an "answer" (and gets the feedback widget) when
  // it has a text answer OR a build_query / run_pql result. A turn whose only
  // meaningful part is a clarifying ask_choice is NOT an answer.
  const hasTextAnswer = message.parts.some(
    (p) => p.type === 'text' && typeof (p as any).text === 'string' && (p as any).text.trim() !== '',
  );
  const hasResult = message.parts.some(
    (p) =>
      (p.type === 'tool-build_query' || p.type === 'tool-run_pql') &&
      (p as any).state === 'output-available',
  );
  const isAnswer = !isUser && (hasTextAnswer || hasResult);

  const answerSummary = message.parts
    .filter((p) => p.type === 'text')
    .map((p) => (p as any).text)
    .join(' ')
    .trim();

  // best-effort meta: the result tool name + viz kind (when a build_query)
  const buildPart = message.parts.find(
    (p) => p.type === 'tool-build_query' && (p as any).state === 'output-available',
  ) as any;
  const runPart = message.parts.find(
    (p) => p.type === 'tool-run_pql' && (p as any).state === 'output-available',
  ) as any;
  const meta: Record<string, unknown> = {};
  if (buildPart) {
    meta.tool = 'build_query';
    meta.vizKind = buildPart.output?.viz?.kind;
  } else if (runPart) {
    meta.tool = 'run_pql';
  }

  // Fire result_rendered / clarifying_shown ONCE per distinct settled result of
  // this message — not on every React re-render. We build a stable signature
  // from the message id + the result kind, and key the effect on it, so the
  // effect body only runs when that signature first appears (or changes).
  const buildKind = buildPart?.output?.viz?.kind as string | undefined;
  const hasRun = !!runPart;
  const askPart = message.parts.find(
    (p) => p.type === 'tool-ask_choice' && (p as any).state === 'output-available',
  ) as any;
  const resultSig = `${message.id}|build:${buildKind ?? ''}|run:${hasRun ? 1 : 0}|ask:${askPart ? 1 : 0}`;
  useEffect(() => {
    if (buildKind) track('result_rendered', { kind: buildKind });
    else if (hasRun) track('result_rendered', { kind: 'raw' });
    if (askPart) track('clarifying_shown');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resultSig]);

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[85%] rounded-2xl px-3 py-1.5 text-sm ${isUser ? 'bg-blue-600 text-white' : 'bg-zinc-800 text-zinc-100'}`}>
        {message.parts.map((part, i) => {
          if (part.type === 'text') {
            // User text stays plain; assistant text is rendered as markdown so
            // the bolded headline (and lists/code) display as formatted, not as
            // literal asterisks.
            if (isUser) {
              return <span key={i} className="whitespace-pre-wrap break-words">{part.text}</span>;
            }
            // A standalone preamble part (e.g. the model emits "I'll compute…"
            // as its OWN text part before the tool call) reduces to '' after
            // stripPreamble — skip it so it renders nothing, not an empty box.
            if (stripPreamble(part.text).trim() === '') return null;
            return <AnswerMarkdown key={i} text={part.text} />;
          }
          if (part.type === 'tool-build_query' && (part as any).state === 'output-available') {
            const out = (part as any).output as BuildQueryResult;
            return <ResultCard key={i} resolvedQuery={out.resolvedQuery} viz={out.viz} context={out.context} heroDraws={out.heroDraws} />;
          }
          if (part.type === 'tool-build_query' && (part as any).state === 'output-error') {
            return (
              <div key={i} className="my-1.5 rounded-xl border border-red-800 bg-red-950/40 p-2.5 text-sm text-red-200">
                ⚠ {(part as any).errorText ?? 'Query failed.'}
              </div>
            );
          }
          if (part.type === 'tool-run_pql' && (part as any).state === 'output-available') {
            const out = (part as any).output as RawResult;
            return <RawResultCard key={i} data={out} />;
          }
          if (part.type === 'tool-run_pql' && (part as any).state === 'output-error') {
            return (
              <div key={i} className="my-1.5 rounded-xl border border-red-800 bg-red-950/40 p-2.5 text-sm text-red-200">
                ⚠ {(part as any).errorText ?? 'Query failed.'}
              </div>
            );
          }
          if (part.type === 'tool-ask_choice' && (part as any).state === 'output-available') {
            const out = (part as any).output as { question: string; options: string[] };
            return <ChoiceCard key={i} question={out.question} options={out.options} onChoose={onChoose} />;
          }
          return null;
        })}
        {isAnswer && (
          <MessageFeedback
            question={question}
            answerSummary={answerSummary || undefined}
            meta={meta}
            onRetry={onRetry}
          />
        )}
      </div>
    </div>
  );
}
