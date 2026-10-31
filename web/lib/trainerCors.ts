// CORS for the PLO Trainer's cross-origin calls (train.magicflop.com -> this app).

const DEFAULT_ORIGINS = [
  'https://train.magicflop.com',
];

function allowedOrigins(): string[] {
  const env = process.env.TRAINER_ALLOWED_ORIGINS;
  if (!env) return DEFAULT_ORIGINS;
  return env.split(',').map((s) => s.trim()).filter(Boolean);
}

export function corsHeaders(req: Request): Record<string, string> {
  const origin = req.headers.get('origin') ?? '';
  const allowed = allowedOrigins();
  const ok =
    allowed.includes('*') ||
    allowed.includes(origin) ||
    // Vercel preview deployments of the trainer project, and local dev on any port.
    /^https:\/\/plotrainer-frontend[a-z0-9-]*\.vercel\.app$/.test(origin) ||
    /^http:\/\/localhost:\d{2,5}$/.test(origin);
  const headers: Record<string, string> = {
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
    Vary: 'Origin',
  };
  if (ok && origin) headers['Access-Control-Allow-Origin'] = allowed.includes('*') ? '*' : origin;
  return headers;
}
