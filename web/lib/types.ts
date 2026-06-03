export type HistogramBucket = { value: number; count: number };
export type Histogram = { pairs: HistogramBucket[]; labels: Record<string, string> | null };

export type PQLResult = {
  values: Record<string, number | null>;
  columns: string[];
  trials: number;
  mode: 'enumeration' | 'monte_carlo';
  seed: number | null;
  histograms?: Record<string, Histogram>;
};

export type RawResult = { query: string; result: PQLResult };

export type EquityRow = { name: string; equity: number; isHero: boolean };
export type WinTieLossRow = { name: string; win: number; tie: number; loss: number };
export type DistributionBar = { token: string; label: string; freq: number };

export type VizSpec =
  | { kind: 'equity'; mode: PQLResult['mode']; trials: number; rows: EquityRow[] }
  | { kind: 'win-tie-loss'; mode: PQLResult['mode']; trials: number; rows: WinTieLossRow[] }
  | { kind: 'distribution'; mode: PQLResult['mode']; trials: number; player: string; bars: DistributionBar[] }
  | { kind: 'equity-by-street'; mode: PQLResult['mode']; trials: number; series: { name: string; points: { street: string; equity: number }[] }[] }
  | { kind: 'equity-distribution'; mode: PQLResult['mode']; trials: number; player: string; mean: number; combos: number; sampledCombos: number | null; buckets: { lo: number; hi: number; pct: number }[] }
  | { kind: 'equity-vs-class'; mode: PQLResult['mode']; trials: number; player: string; rows: { category: string; label: string; equity: number; freq: number }[] }
  | { kind: 'frequency'; mode: PQLResult['mode']; trials: number; label: string; pct: number }
  | { kind: 'draws'; mode: PQLResult['mode']; trials: number; player: string; bars: { label: string; pct: number }[] }
  | { kind: 'outs-distribution'; mode: PQLResult['mode']; trials: number; player: string; handtype: string; street: string; avg: number; bars: { outs: number; prob: number }[] };
