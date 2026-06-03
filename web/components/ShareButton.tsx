'use client';
import { useState } from 'react';
import { track } from '@/lib/analytics';

const APP_URL = 'https://agentic-poker-montecarlo.vercel.app';

export function ShareButton({ summary }: { summary?: string }) {
  const [copied, setCopied] = useState(false);

  async function onShare() {
    const text = summary?.trim()
      ? `${summary.trim()} — agentic-poker-montecarlo`
      : 'Check out this poker equity result — agentic-poker-montecarlo';
    track('result_shared');
    try {
      if (typeof navigator !== 'undefined' && typeof navigator.share === 'function') {
        await navigator.share({ title: 'agentic-poker-montecarlo', text, url: APP_URL });
        return;
      }
      if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(`${text} ${APP_URL}`);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }
    } catch {
      /* user cancelled the share sheet, or clipboard denied — no-op */
    }
  }

  return (
    <button
      type="button"
      onClick={onShare}
      className="mt-3 ml-3 text-xs text-zinc-500 hover:text-zinc-300"
    >
      {copied ? 'Copied!' : 'Share'}
    </button>
  );
}
