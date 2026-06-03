import { streamText, tool, convertToModelMessages, stepCountIs, hasToolCall, type UIMessage } from 'ai';
import { z } from 'zod';
import { AGENT_MODEL } from '@/lib/models';
import { AGENT_SYSTEM_PROMPT } from '@/lib/prompts';
import { QueryIntent } from '@/lib/intent';
import { executeBuildQuery } from '@/lib/buildQuery';
import { runPqlTool } from '@/lib/runPqlTool';

// The Anthropic SDK + streamText() need the Node.js runtime; Edge would break streaming.
export const runtime = 'nodejs';
export const maxDuration = 60;

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = streamText({
    model: AGENT_MODEL,
    system: AGENT_SYSTEM_PROMPT,
    messages: await convertToModelMessages(messages),
    stopWhen: [stepCountIs(5), hasToolCall('ask_choice')],
    tools: {
      build_query: tool({
        description:
          'Run an Omaha-hi equity query. Provide a complete QueryIntent (game, >=2 players with exact hole cards, optional board/dead, metrics). Returns resolved PQL, the engine result, and a viz spec.',
        inputSchema: QueryIntent,
        execute: async (intent) => executeBuildQuery(intent),
      }),
      ask_choice: tool({
        description:
          'Ask the user to choose between discrete options — a clarification or a fallback. Prefer this over listing options in prose. Give a short question and 2-6 options.',
        inputSchema: z.object({
          question: z.string(),
          options: z.array(z.string()).min(2).max(6),
        }),
        execute: async ({ question, options }) => ({ question, options }),
      }),
      run_pql: tool({
        description:
          'Run a RAW PQL query for advanced questions the structured build_query cannot express: conditional "where" filters (e.g. equity GIVEN a flush), comparisons, arithmetic, sum, and histogram distributions. Provide ONE complete PQL string with a full FROM clause. Use build_query for ordinary equity/draw/graph questions.',
        inputSchema: z.object({ query: z.string() }),
        execute: async ({ query }) => runPqlTool({ query }),
      }),
    },
  });

  return result.toUIMessageStreamResponse({
    onError: (error) => {
      console.error('[/api/chat] error:', error);
      return error instanceof Error ? error.message : 'An error occurred.';
    },
  });
}
