import type { RecentItem } from '@/lib/db';

// Curated example exchanges so the gallery and homepage look alive on first
// visit and whenever the DB is empty. Payload shapes mirror RawResult and
// VizSpec from web/lib/types.ts.
export const SEED_EXAMPLES: RecentItem[] = [
  {
    id: 'seed-conditional-flush',
    ts: '2026-06-01T12:00:00.000Z',
    question:
      'When I flop a flush with KhQh against QdQc, how often do I win by the river?',
    kind: 'raw',
    summary:
      'Given a flopped flush, hero is ~99% to win by the river against pocket queens.',
    payload: {
      raw: {
        query:
          "select avg(riverEquity(PLAYER_1)) as eq from game='holdem', PLAYER_1='KhQh', PLAYER_2='QdQc', board='2h7hTh' where flushCard(flop, PLAYER_1)",
        result: {
          values: { eq: 0.99 },
          columns: ['eq'],
          trials: 100000,
          mode: 'monte_carlo',
          seed: null,
          histograms: {},
        },
      },
    },
  },
  {
    id: 'seed-equity-aks-vs-qq',
    ts: '2026-06-01T11:30:00.000Z',
    question: 'AsKs versus QcQd preflop — what is the equity?',
    kind: 'equity',
    summary: 'AsKs is roughly a coin flip against QcQd preflop (~46%).',
    payload: {
      vizSpecs: [
        {
          kind: 'equity',
          mode: 'monte_carlo',
          trials: 100000,
          rows: [
            { name: 'AsKs', equity: 0.4625, isHero: true },
            { name: 'QcQd', equity: 0.5375, isHero: false },
          ],
        },
      ],
    },
  },
  {
    id: 'seed-equity-aces-vs-two',
    ts: '2026-06-01T11:00:00.000Z',
    question: 'How do pocket aces do against two random hands?',
    kind: 'equity',
    summary: 'AhAd holds about 73% equity against two random hands.',
    payload: {
      vizSpecs: [
        {
          kind: 'equity',
          mode: 'monte_carlo',
          trials: 100000,
          rows: [
            { name: 'AhAd', equity: 0.7312, isHero: true },
            { name: 'Random', equity: 0.1389, isHero: false },
            { name: 'Random', equity: 0.1299, isHero: false },
          ],
        },
      ],
    },
  },
];
