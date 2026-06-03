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
});
