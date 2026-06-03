import { runPql } from '@/lib/engine';
import type { RawResult } from '@/lib/types';

export async function runPqlTool({ query }: { query: string }): Promise<RawResult> {
  return { query, result: await runPql(query) };
}
