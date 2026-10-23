# HeavDen Nexus web app (Next.js)

The public website: the **Overview**, **Patients** (ward list, patient charts, reasons, what-if), **Analytics**, **Nexus AI** (the chat), **Behind the scenes**, **The hospital and its data** and **About me**. All data is synthetic.

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

The browser only calls `/api/*`; the routes in `app/api/` forward it to the API at `BACKEND_URL` (default `http://127.0.0.1:8000`, see `lib/backend.ts`), so its address and its secret never reach the browser.

## On Vercel

Project root `web`, Node.js 22.x. Environment variables (Production and Preview):

- `BACKEND_URL`: the API on Azure Container Apps (Terraform output `api_url`).
- `ORIGIN_SECRET` (mark it Sensitive): the API's URL is public, so it answers only requests carrying this secret, sent by `lib/backend.ts`. Copy it from Terraform without printing it: `terraform output -raw origin_secret | clip`, then paste. The release's smoke test needs the same value as the secret `ORIGIN_SECRET` of the GitHub environment `prod-deploy`.

GET answers through `/api/*` are kept by Vercel's CDN for 5 minutes and then refreshed in the background (`app/api/[...path]/route.ts`): repeat visits never wake the API or the SQL warehouse. A new deployment starts with an empty cache.

## Checks

```bash
npm run typecheck
npm run build
```

CI runs both on every pull request.

## Design

The visual language is a **NEWS2 observation chart**, made lively: values plotted as ink-blue dots on chart paper over the NEWS2 scoring zones (yellow, amber, red), which appear only where they mean a score.

- **Layout:** a sticky deep-ink top bar (brand, navigation, a highlighted "About me" button, the simulated ward time) over a dark synthetic-data strip. Every page opens with a full-width dark hero (`components/PageHero.tsx`: the title and introduction take about 70% of the width, an optional panel about 20%), and content runs to 5% from each edge of the screen (`--gutter` in `globals.css`). Full-width bands: `.nightBand` (dark) and `.whiteBand` in `globals.css`. Brand and name lines use `.lockup` (name plus a sentence-case descriptor), not an all-caps eyebrow. Below 1180 px the links move behind a Menu button (closes on Escape or when a link is chosen).
- **Wording:** a personal portfolio for a job search. The About page shows what I've built and the roles I'm open to and offers no services (Vercel's Hobby plan is for non-commercial use). The code is shared on request: no page links to the repository (`CODE_PAGE` in `lib/site.ts`).
- **Colour carries meaning:** ink blue for data, red only for the High band and the alert threshold, amber for alert load, green for things going well (escalations flagged in advance). Headline figures carry the colour of what they measure.
- **Type:** Schibsted Grotesk for headings and big figures (a newsroom grotesk made for confident headlines), with proportional figures so full stops and commas don't take a digit's width; Atkinson Hyperlegible for text and numbers, designed so characters can't be confused (0/O, 1/l), which matters when a number is a vital sign.
- **Motion:** used sparingly, with purpose: the hero charts draw themselves in, a live dot pulses beside the ward time, a heartbeat trace shows what is waking up while the API or the SQL warehouse is cold (`WakeNotice` in `components/bits.tsx`), and the tech stack's cards cascade in once. Everything respects "reduce motion".

Code:

- `components/Shell.tsx`: the top-bar app shell
- `components/PageHero.tsx`: every page's full-width dark hero
- `components/ProofTiles.tsx`: the project's measured results (Behind the scenes)
- `components/TechStack.tsx` and `lib/stack.ts`: the tech stacks (this project's, and the About page's)
- `lib/site.ts`: name, title, contact and the code-on-request links
- `components/ObsChart.tsx`: the observation chart used for every time series
- `components/SizedChart.tsx`: measures its own width, so charts size reliably in any layout
- `lib/news2.ts`: the NEWS2 zones per vital sign
- `lib/api.ts`, `lib/stream.ts`: `/api/*` calls and the assistant's streamed answers
