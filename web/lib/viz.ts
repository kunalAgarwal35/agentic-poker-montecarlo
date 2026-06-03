import type { QueryIntent } from '@/lib/intent';
import type { PQLResult, VizSpec } from '@/lib/types';
import { CATEGORY_ORDER, CATEGORY_LABELS } from '@/components/viz/theme';

const DRAW_LABELS: Record<string, string> = {
  flushDraw: 'Flush draw', straightDraw: 'Straight draw', oesd: 'Open-ended', gutshot: 'Gut-shot',
};

export function pickViz(intent: QueryIntent, result: PQLResult): VizSpec {
  const metrics = intent.metrics;
  if (intent.drawQuery) {
    const { player, kinds } = intent.drawQuery;
    const i = (intent.players.findIndex((p) => p.name === player) + 1) || 1;
    return {
      kind: 'draws', mode: result.mode, trials: result.trials, player: `P${i}`,
      bars: kinds.map((k) => ({
        label: DRAW_LABELS[k] ?? k,
        pct: result.trials ? (result.values[`p${i}_${k}`] ?? 0) / result.trials : 0,
      })),
    };
  }
  if (intent.outsQuery) {
    const { player, handtype, street, buckets } = intent.outsQuery;
    const i = (intent.players.findIndex((p) => p.name === player) + 1) || 1;
    const ge = (k: number) => (k <= 0 ? result.trials : (result.values[`p${i}_ge${k}`] ?? 0));
    const bars: { outs: number; prob: number }[] = [];
    for (let k = 0; k < buckets; k++) {
      bars.push({ outs: k, prob: result.trials ? (ge(k) - ge(k + 1)) / result.trials : 0 });
    }
    bars.push({ outs: buckets, prob: result.trials ? ge(buckets) / result.trials : 0 }); // tail >= buckets
    return {
      kind: 'outs-distribution', mode: result.mode, trials: result.trials, player: `P${i}`,
      handtype, street, avg: result.values[`p${i}_avgouts`] ?? 0, bars,
    };
  }
  if (intent.handTypeQuery) {
    const htq = intent.handTypeQuery;
    const i = (intent.players.findIndex((p) => p.name === htq.player) + 1) || 1;
    const count = result.values[`p${i}_ht`] ?? 0;
    const sym = htq.mode === 'min' ? '≥' : htq.mode === 'max' ? '≤' : '=';
    return { kind: 'frequency', mode: result.mode, trials: result.trials,
      label: `${sym} ${htq.category} by the ${htq.street}`, pct: result.trials ? count / result.trials : 0 };
  }
  if (intent.metrics.includes('nutHi')) {
    const count = result.values['p1_nuts'] ?? 0;
    return { kind: 'frequency', mode: result.mode, trials: result.trials,
      label: `P1 holds the nuts`, pct: result.trials ? count / result.trials : 0 };
  }
  if (intent.metrics.includes('winningDistribution')) {
    return { kind: 'distribution', mode: result.mode, trials: result.trials, player: 'Winner',
      bars: CATEGORY_ORDER.map((token) => ({ token, label: CATEGORY_LABELS[token],
        freq: result.trials ? (result.values[`win_${token}`] ?? 0) / result.trials : 0 })) };
  }
  if (metrics.includes('distribution')) {
    const p = 1; // distribution is single-player (PLAYER_1) in this slice
    return {
      kind: 'distribution',
      mode: result.mode,
      trials: result.trials,
      player: 'P1',
      bars: CATEGORY_ORDER.map((token) => ({
        token,
        label: CATEGORY_LABELS[token],
        freq: result.trials ? (result.values[`p${p}_${token}`] ?? 0) / result.trials : 0,
      })),
    };
  }
  if (metrics.includes('winsHi') && metrics.includes('tiesHi')) {
    return {
      kind: 'win-tie-loss',
      mode: result.mode,
      trials: result.trials,
      rows: intent.players.map((pl, idx) => {
        const win = (result.values[`p${idx + 1}_wins`] ?? 0) / result.trials;
        const tie = (result.values[`p${idx + 1}_ties`] ?? 0) / result.trials;
        const loss = Math.max(0, Math.round((1 - win - tie) * 1e10) / 1e10);
        return { name: `P${idx + 1}`, win, tie, loss };
      }),
    };
  }
  return {
    kind: 'equity',
    mode: result.mode,
    trials: result.trials,
    rows: intent.players.map((pl, idx) => ({
      name: `P${idx + 1}`,
      equity: result.values[`p${idx + 1}`] ?? 0,
      isHero: idx === 0,
    })),
  };
}
