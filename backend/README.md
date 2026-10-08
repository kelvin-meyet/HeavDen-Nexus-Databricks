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

`/chat` (the assistant) comes next.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODE` | `demo` | `demo` or `live` |
| `SNAPSHOT_DIR` | `data/demo_snapshot` | snapshot to serve in demo mode |
| `EMBED_CACHE_DIR` | `data/models` | where the embedding model is cached |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | comma-separated origins for CORS (local dev; in production Vercel proxies `/api/*`) |

## Snapshot layout

```
manifest.json            as_of ("now"), model name and version, risk-band cut-offs, row counts
gold/<table>.parquet     patient_hour_features, risk_scores, alerts_fact, site_kpis_hourly,
                         device_health_daily, encounters (masked patient labels, no names)
model/model.joblib       the champion; model/background.parquet for explanations
rag/                     chunks.jsonl, vectors.npy, meta.json
```

In Phase 6 the same layout is exported from Databricks.
