import { ImageResponse } from 'next/og';
import { dbConfigured } from '@/lib/db';
import { bank } from '@/lib/trainerBank';
import { CODE_RE, ensureChallengeSchema, getChallenge, getEntries, rankEntries } from '@/lib/trainerChallenges';

// Link-preview image for PLO Trainer challenge links (train.magicflop.com/c/CODE).
// GET /api/trainer/og?code=CODE  -> 1200x630 PNG; no/unknown code -> generic card.
export const runtime = 'nodejs';

const MODE_LABEL: Record<string, string> = {
  range_reads: 'Range Reads',
  card_removal: 'Card Removal',
  range_vs_range: 'Range vs Range',
  hero_equity: 'Hero vs Range Equity',
  mixed: 'Mixed modes',
};
const SUIT: Record<string, { s: string; c: string }> = {
  s: { s: '♠', c: '#111827' },
  h: { s: '♥', c: '#dc2626' },
  d: { s: '♦', c: '#2563eb' },
  c: { s: '♣', c: '#16a34a' },
};

function PlayingCard({ card }: { card: string }) {
  const suit = SUIT[card[1]];
  return (
    <div
      style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        width: 120, height: 168, borderRadius: 16, background: '#f8fafc', color: suit.c,
        fontSize: 64, fontWeight: 700, marginRight: 18, boxShadow: '0 10px 30px rgba(0,0,0,.45)',
      }}
    >
      <div style={{ display: 'flex' }}>{card[0]}</div>
      <div style={{ display: 'flex', fontSize: 54, marginTop: -6 }}>{suit.s}</div>
    </div>
  );
}

type Info = { title: string; subtitle: string; footer: string; board: string[] };

async function info(code: string | null): Promise<Info> {
  const generic: Info = {
    title: 'Train your PLO reads',
    subtitle: 'Range reads · card removal · range vs range · equity',
    footer: 'Challenge your friends — free, no sign-up',
    board: ['As', 'Kh', 'Td'],
  };
  if (!code || !CODE_RE.test(code) || !dbConfigured()) return generic;
  try {
    await ensureChallengeSchema();
    const c = await getChallenge(code);
    if (!c) return generic;
    const entries = rankEntries(await getEntries(code)).filter((e) => e.finished);
    const first = bank().byId.get(c.question_ids[0]);
    const days = Math.max(0, Math.ceil((new Date(c.expires_at).getTime() - Date.now()) / 86_400_000));
    const game = c.game === 'mixed' ? 'PLO4 + PLO5' : c.game;
    return {
      title: `${c.creator_name} challenged you`,
      subtitle: `${MODE_LABEL[c.mode] ?? c.mode} · ${game} · ${c.question_ids.length} questions`,
      footer: entries.length
        ? `Score to beat: ${entries[0].total} (${entries[0].name}) · ${entries.length} played · ${days ? `ends in ${days}d` : 'ended'}`
        : `Be the first to play · ${days ? `ends in ${days}d` : 'ended'}`,
      board: first?.board ?? generic.board,
    };
  } catch {
    return generic;
  }
}

export async function GET(req: Request) {
  const code = new URL(req.url).searchParams.get('code')?.toUpperCase() ?? null;
  const i = await info(code);
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%', height: '100%', display: 'flex', padding: '64px 72px',
          background: 'linear-gradient(135deg, #064e3b 0%, #0b1f17 55%, #0a0a0a 100%)',
          color: '#f8fafc', fontFamily: 'sans-serif',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', flex: 1, justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', fontSize: 30, letterSpacing: 6, color: '#6ee7b7' }}>PLO TRAINER</div>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', fontSize: 72, fontWeight: 800, lineHeight: 1.05 }}>{i.title}</div>
            <div style={{ display: 'flex', fontSize: 36, marginTop: 22, color: '#d1fae5' }}>{i.subtitle}</div>
          </div>
          <div style={{ display: 'flex', fontSize: 30, color: '#fde68a' }}>{i.footer}</div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          {i.board.map((c) => (
            <PlayingCard key={c} card={c} />
          ))}
        </div>
      </div>
    ),
    {
      width: 1200,
      height: 630,
      headers: { 'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=3600' },
    },
  );
}
