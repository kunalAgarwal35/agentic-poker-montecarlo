import { describe, it, expect } from 'vitest';
import { AGENT_SYSTEM_PROMPT } from '@/lib/prompts';

describe('AGENT_SYSTEM_PROMPT', () => {
  it('documents both tools and the raw-PQL surface', () => {
    expect(AGENT_SYSTEM_PROMPT).toMatch(/run_pql/);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/build_query/);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/histogram/);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/where/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/ask_choice/);
  });
  it('keeps declining hi/lo and stud/razz', () => {
    expect(AGENT_SYSTEM_PROMPT).toMatch(/omaha8|hi.?lo|stud|razz/i);
  });
  it('still documents core build_query metrics and ranges', () => {
    expect(AGENT_SYSTEM_PROMPT).toMatch(/equity/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/distribution/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/30%!5%|percentile|range/i);
  });
  it('enforces the verdict-first + coaching output contract', () => {
    expect(AGENT_SYSTEM_PROMPT).toMatch(/verdict/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/assum/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/gutshot/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/open-ended|OESD/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/pot odds|break-even/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/rule of 2|rule-of-2/i);
  });
  it('bans preamble filler and requires the verdict in the first words', () => {
    expect(AGENT_SYSTEM_PROMPT).toMatch(/first words/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/never begin with|no preamble/i);
    // Strengthened, emphatic no-preamble rule.
    expect(AGENT_SYSTEM_PROMPT).toMatch(/zero exceptions/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/statement of intent/i);
  });

  it('makes nut-flush-draw status engine-truth and bans nuttiness/blocker speculation', () => {
    expect(AGENT_SYSTEM_PROMPT).toMatch(/nutFlushDraw|nut flush draw/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/to the nuts/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/blocker/i);
  });
  it('requires clean finished percentages, not raw arithmetic', () => {
    expect(AGENT_SYSTEM_PROMPT).toMatch(/rule of 2-and-4|resulting percentage|finished percentage/i);
  });
  it('enforces commentary accuracy (no invented board specifics)', () => {
    expect(AGENT_SYSTEM_PROMPT).toMatch(/commentary accuracy/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/invent/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/verifiabl/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/flush draw/i);
  });
  it('enforces the hard rule: no out-counts or named draws unless engine-verified this turn', () => {
    expect(AGENT_SYSTEM_PROMPT).toMatch(/HARD RULE/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/engine tool result|ran .* this turn|did not run an engine/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/backdoor/i);
  });
  it('tells the agent to use heroDraws as the engine-verified source for draws/outs', () => {
    expect(AGENT_SYSTEM_PROMPT).toMatch(/heroDraws/);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/flushOuts/);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/straightOuts/);
  });
  it('bans inventing clean/unclean out-quality caveats and self-contradiction', () => {
    expect(AGENT_SYSTEM_PROMPT).toMatch(/clean.*unclean|do not subdivide|tainted/i);
    expect(AGENT_SYSTEM_PROMPT).toMatch(/internally consistent|contradict/i);
  });
  it('forbids a redundant separate draw/outs query alongside an equity build_query', () => {
    expect(AGENT_SYSTEM_PROMPT).toMatch(/single query|do not also run a separate/i);
  });
});
