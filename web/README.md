# HeavDen Nexus web app (Next.js)

The public interface (Plan.md §13): **Patients** (ward list, patient charts, reasons, what-if), **Analytics**, **Assistant** and **Behind the scenes**. All data is synthetic.

## Run it locally

Two terminals, from the repo root:

```bash
# 1. the API (needs the demo snapshot: uv run python backend/scripts/build_demo_snapshot.py)
uv run uvicorn heavden_api.app:app --port 8000

# 2. the web app
cd web
npm install
npm run dev            # http://localhost:3000
```

Open **http://localhost:3000** (not 127.0.0.1: Next's dev server blocks its own scripts for other origins, so the page wouldn't load its data).

The browser only calls `/api/*`; `next.config.ts` forwards it to the API at `BACKEND_URL` (default `http://127.0.0.1:8000`). On Vercel, `BACKEND_URL` points at the Render service, so its address and any secrets never reach the browser.

## Checks

```bash
npm run typecheck
npm run build
```

CI runs both on every pull request.

## Design

The visual language is a paper **NEWS2 observation chart**: values plotted as dots joined by ink-blue lines on chart paper, over the NEWS2 scoring zones (pale yellow, amber, red), which appear only where they mean a score. One typeface, **Atkinson Hyperlegible**, designed so characters can't be confused (0/O, 1/l): useful when a number is a vital sign. Red is reserved for the High band and the alert threshold.

- `components/ObsChart.tsx`: the observation chart used for every time series
- `components/SizedChart.tsx`: measures its own width, so charts size reliably in any layout
- `lib/news2.ts`: the NEWS2 zones per vital sign
- `lib/api.ts`: `useApi` / `postJson` against `/api/*`
