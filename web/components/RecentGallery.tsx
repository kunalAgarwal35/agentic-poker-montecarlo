'use client';
import { useEffect, useState } from 'react';
import type { VizSpec, RawResult } from '@/lib/types';
import { ResultViz } from '@/components/viz/ResultViz';
import { RawResultCard } from '@/components/viz/RawResultCard';
import { track } from '@/lib/analytics';

type Item = {
  id: string;
  ts?: string;
  question: string;
  kind: string;
  summary?: string;
  payload?: { vizSpecs?: VizSpec[]; raw?: RawResult };
};

export function RecentGallery({ onPick }: { onPick: (q: string) => void }) {
  const [items, setItems] = useState<Item[]>([]);
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/recent').then((r) => r.json()).then((d) => setItems(d.items ?? [])).catch(() => {});
  }, []);

  // Client-side safeguard: newest first, capped at 30.
  const sorted = [...items]
    .sort((a, b) => (b.ts ?? '').localeCompare(a.ts ?? ''))
    .slice(0, 30);

  return (
    <div className="mt-4">
      <h2 className="mb-2 text-sm font-medium text-zinc-400">Recent questions (shared)</h2>
      {sorted.length === 0 ? (
        <p className="rounded-xl border border-dashed border-zinc-800 bg-zinc-900/30 p-3 text-xs text-zinc-600">
          Your recent questions will appear here after you ask one.
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {sorted.map((it) => (
            <div key={it.id} className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-3">
              <button type="button" onClick={() => {
                  const expanding = open !== it.id;
                  setOpen(expanding ? it.id : null);
                  if (expanding) track('recent_opened', { itemId: it.id });
                }}
                className="text-left text-sm text-zinc-200 hover:text-white">
                {it.question}
              </button>
              {it.summary && <p className="mt-1 text-xs text-zinc-500">{it.summary}</p>}
              {open === it.id && (
                <div className="mt-2 min-h-[220px]">
                  {it.payload?.raw && <RawResultCard data={it.payload.raw} />}
                  {it.payload?.vizSpecs?.map((v, i) => <div key={i} className="mt-2"><ResultViz spec={v} /></div>)}
                  <button type="button" onClick={() => onPick(it.question)}
                    className="mt-2 text-xs text-green-500 hover:underline">ask this again</button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
