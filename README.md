# agentic-poker-montecarlo

An AI agent that answers natural-language poker-equity questions by compiling them to PQL and running them on a Monte-Carlo equity engine — with adaptive charts and a shared example gallery.

## Live demo

> Live at: _(coming soon)_

## What it is

Ask a poker question in plain English ("What's my equity with AsKs against QdQc?", "How does the nut-flush draw run out by street?"). An agent built on the Vercel AI SDK + Claude turns the question into a [PQL](https://www.propokertools.com/pql) query, runs it against a pure-Python Monte-Carlo engine, and returns equities, draws, outs, and distribution graphs.

The agent asks clarifying questions when a request is ambiguous (e.g. which game, or missing hole cards) and picks the result visualization that fits the answer — a single equity card, a win/tie/loss split, a by-street line chart, or an equity-distribution histogram. Notable example queries are saved to a shared gallery so visitors can see what the agent can do.

## Architecture

```
                    ┌──────────────────────────────────────────────┐
   Browser  ───────▶│  Vercel — Next.js app (web/)                  │
                    │                                              │
                    │  /api/chat     ──▶ Claude (Vercel AI SDK)    │
                    │                    └─ tools: build_query,     │
                    │                       run_pql, ask_choice     │
                    │                                              │
                    │  /api/recent   ──▶ Neon Postgres  (gallery)  │
                    │  /api/feedback ──▶ Neon Postgres  (👍 / 👎)   │
                    │                                              │
                    │  ENGINE_URL ─────────────┐                   │
                    └──────────────────────────┼───────────────────┘
                                               │
                                               ▼
                    ┌──────────────────────────────────────────────┐
                    │  Railway — slim Python engine (server.py)     │
                    │    POST /pql        run a PQL query           │
                    │    POST /pql-graph  street / distribution /   │
                    │                     vsclass equity graphs     │
                    │    GET  /health                              │
                    │                                              │
                    │  pure Python + numba (JIT Monte-Carlo).       │
                    │  No JVM / no Java at runtime.                 │
                    └──────────────────────────────────────────────┘
```

The engine is pure Python (numba-accelerated Monte-Carlo / enumeration). There is **no Java in the runtime path** — see [Note on the Java oracle](#note-on-the-java-oracle).

## Repo structure

| Path | What it is |
| --- | --- |
| `web/` | Next.js app — the agent UI, `/api/chat`, `/api/recent`, `/api/feedback`, charts, gallery |
| `server.py` | Slim Flask server exposing the engine (`/health`, `/pql`, `/pql-graph`) |
| `Dockerfile` | Container image for the engine (deployed to Railway) |
| `pql/` | The PQL parser + Monte-Carlo equity engine (numba) |
| `tests/` | Python engine tests (parser, executor, ranges, graphs, slim server, …) |
| `docs/` | Design specs and implementation plans |

## Local development

### a) Engine (Python)

```bash
pip install -r requirements-engine.txt
python server.py        # serves on http://localhost:5050
```

`server.py` listens on port **5050** by default (override with `PORT`). The first `/pql` or `/pql-graph` request triggers a one-time numba JIT warmup that can take ~30–90s.

### b) Web (Next.js)

```bash
cd web
npm install
npm run dev             # serves on http://localhost:3000
```

### c) Required environment

Create `web/.env.local` (there is a `web/.env.local.example` to copy from):

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | Anthropic key with `claude-opus-4-8` access (drives the agent) |
| `ENGINE_URL` | Engine base URL — set to `http://localhost:5050` for local dev (default if unset) |
| `DATABASE_URL` | Neon Postgres connection string for the gallery + feedback (`POSTGRES_URL` also accepted) |

`web/lib/db.ts` reads `DATABASE_URL ?? POSTGRES_URL`; if neither is set the gallery/feedback routes degrade gracefully and the rest of the app still works. `web/lib/engine.ts` reads `ENGINE_URL`.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for deploying web → Vercel, engine → Railway, database → Neon.

## Note on the Java oracle

The ProPokerTools `java_files/*.jar` is a **licensed, dev-only differential-testing oracle**. It is used only by optional dev-time tests (`tests/test_differential_java.py`) to cross-check engine output, is **gitignored and not distributed**, and is **not part of the runtime**. Those tests auto-skip when the jar (or Java) is absent, so the full suite runs without it.

## License

MIT — see [LICENSE](LICENSE).
