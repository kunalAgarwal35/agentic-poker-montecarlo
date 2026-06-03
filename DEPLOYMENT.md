# Deployment

The app deploys as three independent pieces:

1. **Neon** — Postgres database for the shared "recent questions" gallery and user feedback.
2. **Railway** — the slim Python Monte-Carlo engine (`server.py`), built from the repo `Dockerfile`.
3. **Vercel** — the Next.js web app (`web/`), which talks to Claude, Neon, and the engine.

Recommended order: **Neon → Railway → Vercel**, because the Vercel app needs both the engine URL and the database connection string as environment variables.

---

## 1. Neon (database)

Create a Postgres database one of two ways:

- **Via Vercel** (simplest): in your Vercel project, open the **Storage** tab → add **Postgres / Neon**. Vercel auto-injects the connection string (`DATABASE_URL` / `POSTGRES_URL`) into the project's environment variables.
- **Via [neon.tech](https://neon.tech)** directly: create a project and copy its connection string into the Vercel env vars yourself (see step 3).

Initialize the schema once by running `web/lib/schema.sql` against the database (it is idempotent — `CREATE TABLE IF NOT EXISTS …`). You can paste it into the Neon SQL editor, or pipe it via `psql "$DATABASE_URL" -f web/lib/schema.sql`.

> The app also calls `ensureSchema()` lazily on first DB access, so the tables are created on demand even if you skip the manual step — but running the schema once up front avoids a first-request latency hit.

---

## 2. Railway (engine)

1. Create a **new project** on [Railway](https://railway.app) from this repo.
2. Set the service **root directory to the repo root** (where the `Dockerfile` lives). Railway detects the `Dockerfile` and builds the engine image.
3. Railway injects a `$PORT` env var at runtime; the container's start command (`waitress-serve --port=${PORT} server:app`) binds to it automatically — no manual port config needed.
4. Once deployed, copy the service's **public URL** (e.g. `https://<your-engine>.up.railway.app`). You'll set this as `ENGINE_URL` in Vercel.

Sanity check: `GET <public-url>/health` should return `ok`.

---

## 3. Vercel (web)

1. **Import the repo** into Vercel.
2. Set **Root Directory** to `web` (the Next.js app is a subdirectory of the monorepo). A `web/.vercelignore` is included as a belt-and-suspenders exclude.
3. Add the following **Environment Variables**:

| Variable | Value |
| --- | --- |
| `ANTHROPIC_API_KEY` | Your Anthropic API key — must have access to `claude-opus-4-8`. |
| `ENGINE_URL` | The Railway engine public URL from step 2 (no trailing slash). |
| `DATABASE_URL` _or_ `POSTGRES_URL` | Neon connection string. **Auto-injected** if you added Neon via the Vercel Storage tab; otherwise paste it manually. |

> **`maxDuration`:** the `/api/chat` route sets `export const maxDuration = 60` (60 seconds), since an agent turn may make several tool calls and an engine round-trip. This requires a Vercel plan that permits a 60s function duration (the Hobby plan caps lower than this on some configurations — verify your plan's limit).

Deploy. The build runs `next build` from `web/`.

---

## 4. Smoke checklist

After deploying, run through these against the live URL:

- [ ] **Conditional query** → ask something like "equity for KhQh vs QdQc on Ah7c2h **given** I make a flush" → returns a raw/result card.
- [ ] **Distribution** → ask for an equity distribution or histogram → renders a **chart**.
- [ ] **Ambiguous question** → ask something underspecified (e.g. omit the game or a hand) → the agent shows a **clarifying choice** prompt.
- [ ] **Feedback** → submit a 👍 on one answer and a 👎 (with a reason) on another.
- [ ] **Gallery persistence** → reload in a **fresh browser / incognito window** and confirm the "recent questions" gallery still shows prior entries (proves Neon persistence, not local state).
- [ ] **Analytics** → confirm **Vercel Analytics** (and Speed Insights) receive events in the Vercel dashboard.

---

## Environment variables reference

| Variable | Used by | Where it's set | Notes |
| --- | --- | --- | --- |
| `ANTHROPIC_API_KEY` | Web (`/api/chat` → Claude) | Vercel env vars; local `web/.env.local` | Needs `claude-opus-4-8` access. |
| `ENGINE_URL` | Web (`web/lib/engine.ts`) | Vercel env vars; local `web/.env.local` | Railway engine URL in prod; `http://localhost:5050` locally (default if unset). |
| `DATABASE_URL` / `POSTGRES_URL` | Web (`web/lib/db.ts`, reads `DATABASE_URL ?? POSTGRES_URL`) | Vercel env vars (auto-injected by Neon integration) | Neon Postgres. If unset, gallery + feedback degrade gracefully. |
| `PORT` | Engine (`server.py`, Dockerfile) | Set automatically by Railway | Local default `5050`; container default `8080`. |
