# HeavDen-Nexus API (FastAPI)

The backend for the public app (Plan.md §13). It serves the Analytics, Patient Risk and document search features to the Next.js frontend. **All data is synthetic.**

| Mode | Data | Model | Documents |
|---|---|---|---|
| `MODE=demo` (default, built) | Parquet snapshot of the gold tables, queried with DuckDB | the champion, loaded from the snapshot | FAISS index from the snapshot |
| `MODE=live` (Phase 5) | Databricks SQL warehouse (service principal, OAuth M2M) | Model Serving endpoint | Vector Search |

Both modes run the **same SQL** (`heavden_api/queries.py`) against `gold.<table>`, so the demo is a faithful stand-in for live.

## Run it locally

```bash
uv sync --all-groups
uv run python backend/scripts/build_demo_snapshot.py   # ~20 s; writes data/demo_snapshot/
uv run uvicorn heavden_api.app:app --reload            # http://127.0.0.1:8000/docs
```

The snapshot builder needs the Synthea output (notebook 01), the champion in the local MLflow registry (notebook 06) and the document index (notebook 08). Document search downloads the embedding model (~130 MB) into `data/models/` on first use.

## Endpoints

| Endpoint | What it returns |
|---|---|
| `GET /health` | mode, snapshot time, model version, risk-band cut-offs, row counts. The web app calls it on page load to wake Render. |
| `GET /analytics/summary?site_id=` | census, patients per band, alerts and escalations in the last 24 h, 7-day alert precision and warning time |
| `GET /analytics/census?site_id=&hours=72` | hourly census, bands, alerts and escalations per site |
| `GET /analytics/units?site_id=` | patients per band on each unit now |
| `GET /analytics/alerts?site_id=&days=7` | alerts per day, split by outcome |
| `GET /analytics/devices?site_id=&days=7` | device uptime, outages, stuck sensors, SpO2 and firmware per day |
| `GET /risk/board?site_id=&unit_id=&band=&limit=50` | patients on the ward now, highest risk first, with reasons and the 6-hour change |
| `GET /risk/patients/{encounter_id}?hours=24` | one stay: header, hourly risk and vitals, past alerts |
| `POST /risk/score` | what-if: `{"encounter_id": "...", "changes": {"spo2": 88, "acvpu": "C"}}` gives risk, band and NEWS2 before and after, plus reasons |
| `GET /documents/search?q=&k=4&mode=hybrid&as_of=` | passages with citations; `covered: false` when the documents probably don't answer |
| `GET /chat/examples` | suggested questions, and whether live answers are on |
| `POST /chat` | the assistant: `{"message": "...", "history": [...]}` gives the answer, the tool steps (SQL with results, document searches, patient lookups) and numbered citations |

## The assistant (`/chat`)

One tool-calling agent with three tools, mirroring the Databricks design (Plan.md §11):

| Tool | Demo mode | Live mode (Phase 4-5) |
|---|---|---|
| `query_gold` | LLM-written SQL in a locked DuckDB sandbox | Genie space |
| `search_documents` | FAISS + BM25 over the snapshot's document chunks | Vector Search |
| `explain_patient` | latest risk, band, reasons and alerts | UC function `explain_patient_risk` |

- **LLM:** OpenAI (`LLM_MODEL`, default `gpt-5.5`), only when `OPENAI_API_KEY` is set. Without a key, or if the LLM call fails, `/chat` replays **recorded** answers (`heavden_api/chat/recordings.json`) and suggests the recorded questions. Re-record with `uv run python backend/scripts/record_chats.py` and **review the answers before committing**.
- **SQL safety, two independent layers:** `heavden.agent.sql_guard` parses the SQL and allows only one SELECT over allow-listed `gold` tables (no table functions, no file or environment functions); the sandbox (`chat/sandbox.py`) holds in-memory copies of the gold tables without internal ids, with external access disabled and configuration locked, and stops queries after 5 s. Rows are capped at 200.
- **Abuse and cost limits** (`chat/limits.py`): messages up to 1,000 characters and 10 history turns; `CHAT_REQUESTS_PER_HOUR` per visitor (default 30); `CHAT_LIVE_ANSWERS_PER_DAY` live LLM answers across **all** visitors (default 300, then recordings are replayed); and a **monthly spending cap on the OpenAI key** as the backstop, since in-memory counters reset when the server restarts.
- **Who is the visitor?** `X-Forwarded-For` is appended to by each proxy, so only entries added by our own proxies can be trusted; anything to their left can be forged by the caller. The visitor is the entry `TRUSTED_PROXY_HOPS` places from the right: 0 locally (header ignored), 1 on Render alone, **2 when Vercel proxies `/api/*` to Render**.
- **Model choice:** `gpt-5.4-mini` was tested first and made reasoning mistakes (wrong arithmetic, accepting false premises, inventing its own metrics); `gpt-5.5` answered all ten example questions correctly. With Chat Completions, function tools require `reasoning_effort="none"` for these models.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODE` | `demo` | `demo` or `live` |
| `SNAPSHOT_DIR` | `data/demo_snapshot` | snapshot to serve in demo mode |
| `EMBED_CACHE_DIR` | `data/models` | where the embedding model is cached |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | comma-separated origins for CORS (local dev; in production Vercel proxies `/api/*`) |
| `OPENAI_API_KEY` | unset | enables live assistant answers. Locally from `.env` (git-ignored); on Render, set it in the dashboard, never in git |
| `LLM_MODEL` | `gpt-5.5` | OpenAI model for the assistant |
| `CHAT_REQUESTS_PER_HOUR` | `30` | per-visitor limit on `/chat` |
| `CHAT_LIVE_ANSWERS_PER_DAY` | `300` | live LLM answers per UTC day across all visitors; then recordings |
| `TRUSTED_PROXY_HOPS` | `0` | proxies we control in front of the API (0 local, 1 Render, 2 Vercel → Render) |

## Snapshot layout

```
manifest.json            as_of ("now"), model name and version, risk-band cut-offs, row counts
gold/<table>.parquet     patient_hour_features, risk_scores, alerts_fact, site_kpis_hourly,
                         device_health_daily, encounters (masked patient labels, no names)
model/model.joblib       the champion; model/background.parquet for explanations
rag/                     chunks.jsonl, vectors.npy, meta.json
```

In Phase 6 the same layout is exported from Databricks.
