export function Hero() {
  return (
    <div className="flex flex-col gap-2">
      <h2 className="text-2xl font-semibold tracking-tight text-zinc-100">
        Ask any poker spot. Get the math.
      </h2>
      <p className="text-sm text-zinc-300">
        Ask any Hold&apos;em or PLO question in plain English — equity, outs, ranges, and
        street-by-street graphs, computed by Monte-Carlo.
      </p>
      <p className="text-xs text-zinc-500">
        Texas Hold&apos;em · PLO (4/5/6 card). Hi-lo, stud, and razz aren&apos;t supported.
      </p>
    </div>
  );
}
