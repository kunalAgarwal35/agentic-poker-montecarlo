'use client';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Renders an assistant answer through react-markdown with a small, safe,
// theme-matched component map. Raw HTML is disabled (no rehype-raw), so any
// HTML in the model output is rendered as text, not executed. USER messages
// are NOT rendered through this — they stay plain.
const COMPONENTS = {
  p: (props: any) => <p className="text-sm leading-relaxed" {...props} />,
  strong: (props: any) => <strong className="font-semibold text-zinc-100" {...props} />,
  em: (props: any) => <em className="italic" {...props} />,
  ul: (props: any) => <ul className="my-1 list-disc space-y-0.5 pl-4 text-sm leading-relaxed" {...props} />,
  ol: (props: any) => <ol className="my-1 list-decimal space-y-0.5 pl-4 text-sm leading-relaxed" {...props} />,
  li: (props: any) => <li className="leading-relaxed" {...props} />,
  code: (props: any) => (
    <code className="rounded bg-black/40 px-1 py-0.5 text-[0.85em] text-zinc-200" {...props} />
  ),
} as const;

// The model is told to lead with the verdict (no preamble), but it occasionally
// still opens with an intent sentence like "I'll compute your equity…". Strip a
// single leading intent-sentence deterministically — but only when substantial
// answer text follows, so a short reply is never blanked.
export function stripPreamble(text: string): string {
  // Case A: the WHOLE text is just a standalone intent sentence (the model
  // emitted the preamble as its own text part, before the tool call) — e.g.
  // "I'll compute your equity and check your draws on this flop." or
  // "Let me check this spot." Blank it entirely so it renders nothing.
  const t = text.trim();
  if (/^(?:I['’]?ll|I will|Let me|Let['’]?s|First,?\s+I['’]?ll|Sure|Okay|Alright|Got it)\b[^.!?\n]*[.!?]?$/i.test(t)) return '';
  // Case B: a leading intent sentence glued to a real answer in the same
  // string — strip just the sentence, but only when substantial answer text
  // follows, so a short reply is never blanked.
  const m = /^\s*(?:I['’]?ll|I will|Let me|Let['’]?s|First,?\s+I['’]?ll|Sure[,!.]?|Okay[,!.]?|Alright[,!.]?|Got it[,!.]?)\b[^.!?\n]*[.!?]\s+/i.exec(text);
  if (m && text.length - m[0].length > 40) return text.slice(m[0].length);
  return text;
}

export function AnswerMarkdown({ text }: { text: string }) {
  return (
    <div className="break-words [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={COMPONENTS}>
        {stripPreamble(text)}
      </ReactMarkdown>
    </div>
  );
}
