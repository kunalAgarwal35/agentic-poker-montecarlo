import { track as vercelTrack } from '@vercel/analytics';

export type AnalyticsEvent =
  | 'question_submitted'
  | 'tool_used'
  | 'result_rendered'
  | 'clarifying_shown'
  | 'recent_opened'
  | 'choice_selected'
  | 'new_chat'
  | 'feedback_given'
  | 'result_shared'
  | 'agent_error';

export function track(
  event: AnalyticsEvent,
  props?: Record<string, string | number | boolean | null>,
) {
  try {
    vercelTrack(event, props as any);
  } catch {
    /* no-op: dev/SSR/test */
  }
}
