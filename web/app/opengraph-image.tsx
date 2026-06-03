import { ImageResponse } from 'next/og';

export const alt = 'agentic-poker-montecarlo — Ask any poker spot. Get the math.';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export default function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          padding: '80px',
          background: '#09090b',
          color: '#fafafa',
          fontFamily: 'sans-serif',
        }}
      >
        <div
          style={{
            display: 'flex',
            fontSize: 40,
            letterSpacing: 4,
            color: '#22c55e',
          }}
        >
          <span style={{ color: '#3f3f46' }}>♣</span>
          <span style={{ color: '#fafafa' }}>♠</span>
          <span style={{ color: '#ef4444' }}>♥</span>
          <span style={{ color: '#ef4444' }}>♦</span>
        </div>
        <div
          style={{
            display: 'flex',
            fontSize: 84,
            fontWeight: 700,
            marginTop: 32,
            lineHeight: 1.1,
          }}
        >
          Ask any poker spot. Get the math.
        </div>
        <div
          style={{
            display: 'flex',
            fontSize: 36,
            marginTop: 28,
            color: '#a1a1aa',
          }}
        >
          Hold&apos;em + PLO equity · Monte-Carlo · plain English
        </div>
        <div
          style={{
            display: 'flex',
            fontSize: 28,
            marginTop: 'auto',
            color: '#52525b',
          }}
        >
          agentic-poker-montecarlo.vercel.app
        </div>
      </div>
    ),
    { ...size },
  );
}
