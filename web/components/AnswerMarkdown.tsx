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

export function AnswerMarkdown({ text }: { text: string }) {
  return (
    <div className="break-words [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={COMPONENTS}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
