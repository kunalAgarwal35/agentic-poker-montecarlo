# Poker Equity — NL Agent UI

Conversational UI over the native PQL engine. Ask Omaha-hi equity questions in plain English; the agent (Claude Opus) clarifies, compiles a PQL query, runs it against the local engine, and shows an equity card.

## Run locally

1. **Engine** — from the repo root: `python api_host.py` (serves `http://localhost:5050`, exposes `POST /pql`).
2. **Web** — from `web/`:
   ```bash
   cp .env.local.example .env.local      # then set ANTHROPIC_API_KEY to a real key
   npm install
   npm run dev                            # http://localhost:3000
   ```

`.env.local` keys: `ANTHROPIC_API_KEY` (required for the agent) and `ENGINE_URL` (defaults to `http://localhost:5050`).

## How it works

`Browser → /api/chat (Claude Opus + build_query tool) → executeBuildQuery → compileIntentToPQL → POST {ENGINE_URL}/pql → equity ResultCard`.

The browser only talks to `/api/chat` (same origin); the engine is called server-side from the route handler, so there's no CORS. The LLM never writes PQL — it emits a structured `QueryIntent` (Zod) that the server validates and compiles deterministically.

## Scope

Fixed-hand omaha-hi (PLO4/5/6) equity + outright-win counts. **Ranges, holdem, and hi/lo are not supported yet** — the agent says so and asks for specific hands / a supported game. Those arrive with the parallel engine-range track.

## Tests

`npm test` (Vitest). Covers the deterministic core: `QueryIntent` schema, PQL compilation, engine client, viz selection, build-query orchestration, and the `ResultCard` component. The LLM path is smoke-tested manually (ask a real question with the dev server + engine running).
