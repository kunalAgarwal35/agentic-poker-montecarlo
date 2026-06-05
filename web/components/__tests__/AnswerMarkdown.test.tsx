import { describe, it, expect } from 'vitest';
import { stripPreamble } from '@/components/AnswerMarkdown';

describe('stripPreamble', () => {
  it('removes a leading intent sentence when real content follows', () => {
    const t = "I'll compute your equity and verify your draws on this flop. **You're behind: ~26% vs ~74%** — QQ flopped top set against your nut flush draw.";
    expect(stripPreamble(t)).toMatch(/^\*\*You're behind/);
  });

  it('handles "Let me…" and curly apostrophes', () => {
    expect(stripPreamble("Let me check this spot. QQ is well ahead at ~76% versus your overcards.")).toMatch(/^QQ is well ahead/);
    expect(stripPreamble("I’ll run the numbers here. AA wins about 85% against a random hand heads-up.")).toMatch(/^AA wins about 85/);
  });

  it('leaves a verdict-first answer untouched', () => {
    const t = '**QQ is ahead — ~54%** vs AK ~46%, the classic race.';
    expect(stripPreamble(t)).toBe(t);
  });

  it('blanks a standalone preamble that is the entire text', () => {
    // The model emits the intent sentence as its OWN text part before the tool
    // call — the whole string is just the preamble, so it should reduce to ''.
    expect(stripPreamble("I'll compute your equity and check your draws on this flop.")).toBe('');
    expect(stripPreamble('Let me check this spot.')).toBe('');
    expect(stripPreamble('Let me check.')).toBe('');
    // Tolerates surrounding whitespace and curly apostrophes.
    expect(stripPreamble('  I’ll run the numbers here.  ')).toBe('');
    expect(stripPreamble('Sure, let me take a look.')).toBe('');
  });

  it('strips a glued intent sentence but keeps the real multi-sentence answer', () => {
    // Two sentences: the first is intent, the second is the verdict. Only the
    // leading intent sentence is stripped; the verdict survives.
    const t = "I'll run the numbers. AKo is ~73% vs AQo, the classic domination.";
    expect(stripPreamble(t)).toMatch(/^AKo is ~73%/);
  });
});
