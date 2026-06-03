export const SUIT_GLYPH: Record<string, string> = { s: '♠', h: '♥', d: '♦', c: '♣' };

// 4-color deck (tuned for the dark theme): spades light, hearts red, diamonds blue, clubs green.
export const SUIT_COLOR: Record<string, string> = {
  s: '#e5e7eb', h: '#f87171', d: '#60a5fa', c: '#4ade80',
};

export function parseCards(s: string): { rank: string; suit: string }[] {
  const out: { rank: string; suit: string }[] = [];
  for (let i = 0; i + 1 < s.length; i += 2) {
    out.push({ rank: s[i], suit: s[i + 1] });
  }
  return out;
}

export function isConcreteHand(s: string): boolean {
  return /^([2-9TJQKA][shdc])+$/i.test(s);
}

export function rangeLabel(s: string): string {
  return /^\d+(\.\d+)?%$/.test(s) ? `Top ${s}` : s;
}
