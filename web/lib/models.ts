import { anthropic } from '@ai-sdk/anthropic';

// Opus drives the agent loop; Sonnet is available for any trivial sub-task.
// NOTE: confirm these model ids are enabled on the ANTHROPIC_API_KEY account.
export const AGENT_MODEL = anthropic('claude-opus-4-8');
export const SIMPLE_MODEL = anthropic('claude-sonnet-4-6');
