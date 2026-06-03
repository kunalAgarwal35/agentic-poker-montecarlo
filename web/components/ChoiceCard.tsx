'use client';

export function ChoiceCard({ question, options, onChoose }: { question: string; options: string[]; onChoose: (o: string) => void }) {
  return (
    <div className="my-2 rounded-xl border border-zinc-700 bg-zinc-900 p-3">
      <div className="mb-2 text-sm text-zinc-200">{question}</div>
      <div className="flex flex-col gap-2">
        {options.map((o) => (
          <button
            key={o}
            type="button"
            onClick={() => onChoose(o)}
            className="rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-left text-sm text-zinc-100 hover:border-green-600 hover:bg-zinc-700"
          >
            {o}
          </button>
        ))}
      </div>
    </div>
  );
}
