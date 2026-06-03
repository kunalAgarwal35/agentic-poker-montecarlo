'use client';

// Plain-English starters a real person would type. The chip label is a short
// friendly phrase; the picked text is the full natural-language prompt.
const EXAMPLES: { label: string; prompt: string }[] = [
  { label: 'AA vs KK preflop, who wins?', prompt: 'AA vs KK preflop, who wins?' },
  { label: 'AK vs a tight 3-bet range', prompt: "AK vs a tight 3-bet range — what's my equity?" },
  { label: 'Flush draw on the turn', prompt: 'Flush draw on the turn — what are my odds?' },
  { label: 'AKs vs QQ, equity by street', prompt: 'How does AKs equity change by street vs QQ on a Js Tc 2d flop?' },
  { label: 'How often does a pair flop a set?', prompt: 'How often does a pocket pair flop a set?' },
  { label: 'PLO: double-suited rundown vs aces', prompt: 'T9s8s7s double-suited vs aces in PLO?' },
];

export function ExampleChips({ onPick }: { onPick: (prompt: string) => void }) {
  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-zinc-500">Try one of these:</p>
      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            type="button"
            onClick={() => onPick(ex.prompt)}
            className="rounded-full border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-300 hover:border-zinc-500 hover:text-zinc-100"
          >
            {ex.label}
          </button>
        ))}
      </div>
    </div>
  );
}
