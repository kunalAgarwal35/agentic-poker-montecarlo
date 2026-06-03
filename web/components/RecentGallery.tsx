'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { VizSpec, RawResult } from '@/lib/types';
import { ResultViz } from '@/components/viz/ResultViz';
import { RawResultCard } from '@/components/viz/RawResultCard';
import { track } from '@/lib/analytics';

type Item = {
  id: string;
  ts?: string;
  question: string;
  kind: string;
  summary?: string | null;
  payload?: { vizSpecs?: VizSpec[]; raw?: RawResult };
};

const PAGE = 20;

export function RecentGallery({ onPick }: { onPick: (q: string) => void }) {
  const [items, setItems] = useState<Item[]>([]);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [loading, setLoading] = useState(false);
  const [openId, setOpenId] = useState<string | null>(null);

  // Refs so the IntersectionObserver callback always sees current values
  // without re-subscribing the observer on every state change.
  const loadingRef = useRef(false);
  const hasMoreRef = useRef(true);
  const offsetRef = useRef(0);
  const sentinelRef = useRef<HTMLDivElement | null>(null);

  const loadMore = useCallback(async () => {
    if (loadingRef.current || !hasMoreRef.current) return;
    loadingRef.current = true;
    setLoading(true);
    try {
      const off = offsetRef.current;
      const r = await fetch(`/api/recent?limit=${PAGE}&offset=${off}`);
      const data = await r.json();
      const incoming: Item[] = Array.isArray(data?.items) ? data.items : [];
      setItems((prev) => {
        const seen = new Set(prev.map((p) => p.id));
        const deduped = incoming.filter((it) => it && !seen.has(it.id));
        return deduped.length ? [...prev, ...deduped] : prev;
      });
      const nextOffset = off + incoming.length;
      offsetRef.current = nextOffset;
      setOffset(nextOffset);
      const more = !!data?.hasMore;
      hasMoreRef.current = more;
      setHasMore(more);
    } catch {
      // Network hiccup — stop paginating but keep what we have.
      hasMoreRef.current = false;
      setHasMore(false);
    } finally {
      loadingRef.current = false;
      setLoading(false);
    }
  }, []);

  // Initial page on mount.
  useEffect(() => {
    void loadMore();
  }, [loadMore]);

  // Infinite scroll: load older questions when the sentinel scrolls into view.
  useEffect(() => {
    const el = sentinelRef.current;
    if (!el) return;
    const observer = new IntersectionObserver((entries) => {
      if (
        entries.some((e) => e.isIntersecting) &&
        hasMoreRef.current &&
        !loadingRef.current
      ) {
        void loadMore();
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [loadMore]);

  function toggle(it: Item) {
    setOpenId((cur) => {
      const expanding = cur !== it.id;
      if (expanding) track('recent_opened', { itemId: it.id });
      return expanding ? it.id : null;
    });
  }

  return (
    <div className="mt-4">
      <h2 className="mb-2 text-sm font-medium text-zinc-400">Recent questions (shared)</h2>
      {items.length === 0 && !loading ? (
        <p className="rounded-xl border border-dashed border-zinc-800 bg-zinc-900/30 p-3 text-xs text-zinc-600">
          Your recent questions will appear here after you ask one.
        </p>
      ) : (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/40">
          {items.map((it) => (
            <div key={it.id} className="border-b border-zinc-800 last:border-b-0">
              <button
                type="button"
                onClick={() => toggle(it)}
                className="block w-full truncate px-3 py-2 text-left text-sm text-zinc-200 hover:text-white hover:bg-zinc-900/60"
              >
                {it.question}
              </button>
              {openId === it.id && (
                <div className="px-3 pb-3">
                  {it.summary && <p className="mb-2 text-xs text-zinc-500">{it.summary}</p>}
                  {it.payload?.raw && (
                    <div className="min-h-[220px]">
                      <RawResultCard data={it.payload.raw} />
                    </div>
                  )}
                  {it.payload?.vizSpecs?.map((v, i) => (
                    <div key={i} className="mt-2 min-h-[220px]">
                      <ResultViz spec={v} />
                    </div>
                  ))}
                  <button
                    type="button"
                    onClick={() => onPick(it.question)}
                    className="mt-2 text-xs text-green-500 hover:underline"
                  >
                    ask this again
                  </button>
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="px-3 py-2 text-xs text-zinc-600">Loading…</div>
          )}
          <div ref={sentinelRef} aria-hidden className="h-px" />
        </div>
      )}
    </div>
  );
}
