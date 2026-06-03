'use client';

const EXAMPLES: { label: string; prompt: string }[] = [
  { label: 'PLO5 flop', prompt: 'In PLO5, how does AsAhKsKhQs do against JdTd9d8d7d on a 2c3c4c flop?' },
  { label: 'Holdem preflop', prompt: 'In holdem, AsKs vs QdQh preflop' },
  { label: 'Holdem flop', prompt: 'In holdem, AsKs vs QdQh on an Ah7c2d flop' },
  { label: 'PLO4 three-way', prompt: 'PLO4 three-way: AsKsQsJs vs AdKdQdJd vs 2c3h4c5h' },
  { label: 'Turn win counts', prompt: 'On the 2c3c4c5c turn, how often does AsAhKsKhQs win outright vs JdTd9d8d7d in PLO5?' },
  { label: 'PLO6', prompt: 'PLO6: AsAhKsKhQsQh vs 2c3c4c5c6c7c preflop' },
  { label: 'Dead cards', prompt: 'In PLO5 on a 2c3c4c flop, AsAhKsKhQs vs JdTd9d8d7d with JsJhTsTh dead' },
  { label: 'Hand vs range', prompt: 'In PLO5, AsAhKsKhQs vs a top 25% range' },
  { label: 'Holdem vs range', prompt: 'In holdem, AsKs vs QQ+' },
  { label: 'Nut frequency', prompt: 'In PLO5 on a 2c3c4c flop, how often does AsAhKsKhQs make the nuts by the river vs JdTd9d8d7d?' },
  { label: 'Winning hand type', prompt: 'In holdem, AsKs vs QdQh — what hand usually wins by the river?' },
  { label: 'Draws', prompt: 'In holdem, what draws does KhQh have on the Ah7c2h flop?' },
  { label: 'Flush outs', prompt: 'In holdem on the Ah7c2h flop, how many flush outs does KhQh have by the turn?' },
  { label: 'Equity by street', prompt: 'In holdem, show the equity-by-street graph for AsKh vs QdJd on Ah7c2dKsQc' },
  { label: 'Equity distribution', prompt: 'In holdem, show the equity distribution of AsKs vs a QQ+ range on the Ah7c2d flop' },
  { label: 'Equity vs class', prompt: 'In holdem on Ah7c2d, show AsKs equity by what a QQ+ opponent makes' },
];

export function ExampleChips({ onPick }: { onPick: (prompt: string) => void }) {
  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-zinc-500">Try an example:</p>
      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            type="button"
            onClick={() => onPick(ex.prompt)}
            className="rounded-full border border-zinc-700 bg-zinc-900 px-3 py-1 text-xs text-zinc-300 hover:border-zinc-500 hover:text-zinc-100"
          >
            {ex.label}
          </button>
        ))}
      </div>
    </div>
  );
}
