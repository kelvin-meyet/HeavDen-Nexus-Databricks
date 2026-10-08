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

The visual language is a **NEWS2 observation chart**, made lively: values plotted as ink-blue dots on chart paper over the NEWS2 scoring zones (yellow, amber, red), which appear only where they mean a score.

- **Layout:** an app shell with a deep-ink sidebar (brand, navigation, live ward time) and content that fills the rest of the screen; on phones the sidebar becomes a top bar.
- **Colour carries meaning:** ink blue for data, red only for the High band and the alert threshold, amber for alert load, green for things going well (escalations flagged in advance). Headline figures carry the colour of what they measure.
- **Type:** Bricolage Grotesque for headings and figures; Atkinson Hyperlegible for text and numbers, designed so characters can't be confused (0/O, 1/l), which matters when a number is a vital sign.
- **Motion:** used once, with purpose: the hero charts draw themselves in, and a live dot pulses beside the ward time. Everything respects "reduce motion".

Code:

- `components/Shell.tsx`: the sidebar app shell
- `components/ObsChart.tsx`: the observation chart used for every time series
- `components/SizedChart.tsx`: measures its own width, so charts size reliably in any layout
- `lib/news2.ts`: the NEWS2 zones per vital sign
- `lib/api.ts`, `lib/stream.ts`: `/api/*` calls and the assistant's streamed answers
