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

  it('never blanks a short reply that has no substantial follow-up', () => {
    expect(stripPreamble('Let me check.')).toBe('Let me check.');
  });
});
