# HeavDen-Nexus API (FastAPI)

The backend for the public app (Plan.md §13). It serves the Analytics, Patient Risk and document search features to the Next.js frontend. **All data is synthetic.**

| Mode | Pages | What-if model | Documents | Assistant's numbers |
|---|---|---|---|---|
| `MODE=demo` (default; the website from 1 Nov 2026) | Parquet snapshot of the gold tables, queried with DuckDB | the champion, from the snapshot | FAISS + BM25 | the model's own SQL in a locked DuckDB sandbox |
| `MODE=live` (the website until 31 Oct 2026) | prod's Gold through the Databricks SQL warehouse, cached per query | the champion, from the snapshot | FAISS + BM25 (built into the image) | prod's Genie space; one patient via `ml.explain_patient_risk` |

Both modes run the **same SQL** (`heavden_api/queries.py`) against `gold.<table>`, so the demo is a faithful stand-in for live. On Azure Container Apps the API signs in to Databricks, Blob Storage, Table Storage and Azure OpenAI as its own managed identity (no stored credentials; its one secret is the origin secret it checks on every request), and downloads its snapshot from Blob Storage at every start (`SNAPSHOT_URL`). The whole deployment is described in [`docs/Phase-5-README.md`](../docs/Phase-5-README.md).

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
| `GET /health` | mode, snapshot time, model version, risk-band cut-offs, row counts. The web app calls it on page load, which also wakes the API when it has scaled to zero. It never wakes the SQL warehouse: in live mode "now" is prod's last answer, or the snapshot's hour before any page has asked prod. `live_data` says whether the pages read prod live right now (false in demo mode and once the day's warehouse allowance is used up; the web header then shows a note). |
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
| `POST /chat` | the assistant: `{"message": "...", "conversation_id": "..."}` gives the answer, the tool steps (SQL with results, document searches, patient lookups), numbered citations and the `conversation_id` to send with a follow-up |
| `POST /chat/stream` | the same answer as server-sent events while it is produced: `meta`, `step_start`, `step`, `delta`, `reset`, `redact`, `done` (see `chat/agent.py`) |

## The assistant (`/chat`)

One tool-calling agent with three tools, mirroring the Databricks design (Plan.md §11):

| Tool | Demo mode | Live mode |
|---|---|---|
| `query_gold` | LLM-written SQL in a locked DuckDB sandbox | prod's Genie space |
| `search_documents` | FAISS + BM25 over the document chunks | the same (Vector Search is an option of the Databricks assistant) |
| `explain_patient` | latest risk, band, reasons and alerts | UC function `explain_patient_risk` |

- **LLM:** OpenAI (`LLM_MODEL`, default `gpt-5.5`), Azure OpenAI in Microsoft Foundry with `LLM_PROVIDER=azure`, or a model served by Databricks with `LLM_PROVIDER=databricks` (no keys; see *Choosing the LLM provider*). Only the app's list in `chat/models.yaml` is accepted: `gpt-5.5` (the main model), `gpt-5-mini` (the fallback, at its default reasoning) and `databricks-gpt-oss-120b` (for comparison: it failed the app's scorecard); the Databricks assistant has its own list. Live answers need the chosen provider's key (or, for Databricks, an identity). Without a key, or if the LLM call fails, `/chat` replays **recorded** answers (`heavden_api/chat/recordings.json`) and suggests the recorded questions. Re-record with `uv run python backend/scripts/record_chats.py` and **review the answers before committing**.
- **SQL safety, two independent layers:** `heavden.agent.sql_guard` parses the SQL and allows only one SELECT over allow-listed `gold` tables (no table functions, no file or environment functions); the sandbox (`chat/sandbox.py`) holds in-memory copies of the gold tables without internal ids, with external access disabled and configuration locked, and stops queries after 5 s. Rows are capped at 200.
- **Guardrails, three independent layers** (`chat/guardrails.py`): (1) before the LLM, obvious attempts to extract the instructions, configuration, keys or source code, or to override the rules, get a fixed refusal at no cost (code requests are pointed to GitHub); (2) the system prompt sets the scope, forbids revealing or paraphrasing its instructions, treats document and query text as data, and redirects anything about real patients to their clinical team; (3) every answer, also while streaming, is checked for a per-process canary hidden in the prompt, distinctive prompt phrases and key patterns, and replaced if it leaks. The prompt contains nothing secret, so a paraphrase that slips through reveals no credentials. `backend/scripts/redteam_chat.py` sends 19 prompts (extraction attempts, off-topic requests, a real-patient emergency, and misspelt in-scope questions that must not be refused) through `/chat` to the live model; the checks are in `chat/redteam.py`. On 2026-10-17 `gpt-5.5` and `gpt-5-mini` failed none of the checks, and GPT-OSS 120B one (it showed a unit code). Until that day the internal-code check matched nothing (its pattern held control characters), so earlier results didn't cover it.
- **Streaming:** `/chat/stream` sends server-sent events. In the web app a dedicated route (`web/app/api/chat/stream/route.ts`) passes them through, because the generic `/api` route (`web/app/api/[...path]/route.ts`) waits for whole responses.
- **Conversation memory** (`chat/memory.py`): kept on the server under a random `conversation_id`: the last 6 turns, each with a compact note of the SQL and rows, documents and patient behind it, so follow-ups like "that unit" or "why is she flagged?" resolve. Idle conversations expire after 30 minutes; at most 500 are kept. The client never sends history, so it can't be forged; refused and redacted turns aren't remembered.
- **Abuse and cost limits** (`chat/limits.py`): messages up to 1,000 characters and 10 history turns; `CHAT_REQUESTS_PER_HOUR` per visitor (default 30); `CHAT_LIVE_ANSWERS_PER_DAY` live LLM answers across **all** visitors (default 100, then recordings are replayed). The per-visitor counts live in memory (the API stores no visitor data). The daily count lives in memory locally, and on Container Apps in the Azure Table named by `CHAT_BUDGET_TABLE_URL`, because the API is stopped whenever nobody visits. If that table can't be reached, no live answers are given (it fails closed). In live mode, `WarehouseAllowance` caps the SQL warehouse time of the pages and the chat together (`WAREHOUSE_SECONDS_PER_DAY`, 2 hours): before each statement or Genie question the API reserves the time it keeps the warehouse on (the query plus the 5 minutes until it stops), counted once when queries overlap, in the same table. Past it, until midnight UTC, pages run the same SQL on the snapshot, and the chat stays live on the documents: its data and patient tools refuse at once and tell the model to say numbers are paused (`ALLOWANCE_HINT`). It fails closed too. It counts time, not requests, because a slow trickle of queries keeps the warehouse on all day. Patient ids that aren't the generator's form (`E-` and digits) are refused without a query. Checked on the live website: with the API awake, one address gets exactly 30 an hour (the 31st was refused, 22 Oct 2026), even with a different forged `X-Forwarded-For` on each request, while a phone on another network was still answered (21 Oct). A first test let 31 through when its first request also woke the API from zero; that request was most likely counted under another proxy address (likely, not proven: addresses aren't logged).
- **Who is the visitor?** `X-Forwarded-For` is appended to by each proxy, so only entries added by our own proxies can be trusted; anything to their left can be forged by the caller. The visitor is the entry `TRUSTED_PROXY_HOPS` places from the right: 0 locally (header ignored), 1 behind Container Apps' ingress alone, **2 when Vercel proxies `/api/*` to Container Apps** (Vercel replaces the header with the visitor's address; the ingress appends its own entry).
- **Only the website reaches the API:** the API's Container Apps URL is public, so with `ORIGIN_SECRET` set it refuses (403) every request without that value in the `x-origin-secret` header. Only the website's proxy (`web/lib/backend.ts`) and the release's smoke test send it. Callers who skipped the website would also skip its visitor address and its CDN cache.
- **Model choice:** `gpt-5.4-mini` was tested first and made reasoning mistakes (wrong arithmetic, accepting false premises, inventing its own metrics); `gpt-5.5` answered all ten example questions correctly, with `reasoning_effort="none"`. Tool calls also work at the models' default reasoning (checked 16 Oct 2026); each model's setting is in `chat/models.yaml`. The quality test (`backend/scripts/assistant_quality.py`, 23 questions; results and how the model lists were chosen in `docs/quality/README.md`) judges each assistant's list at the settings it ships with: on 17 Oct `gpt-5.5` at `none` passed the app's scorecard, where the model writes the SQL itself.

### The Databricks assistant

The same loop, instructions and guardrails, with the Databricks tools (`chat/databricks_tools.py`), as an MLflow `ResponsesAgent` (`chat/databricks_assistant.py`) served by Model Serving on `heavden-assistant-<target>` (scale to zero). The bundle job `assistant_deploy` (`chat/assistant_deploy.py`) logs it with the packaged document index and the resources it calls, registers it as `<catalog>.ml.assistant`, deploys it, and asks it one question. In staging, CI then red-teams the endpoint and runs the quality test on every model in the `databricks:` list (`assistant-staging` in `.github/workflows/ci.yml`), only when its code, documents or settings changed.

- **Its model** is the bundle variable `assistant_llm_endpoint` (`databricks/databricks.yml`; gpt-5-mini since 18 Oct): change it and merge, and CI deploys it. Each model is a serving endpoint: GPT-OSS is hosted by Databricks, and gpt-5.5 and gpt-5-mini go through the AI Gateway endpoints in `infra/terraform/llm_gateway.tf`.
- **Its answers** carry `custom_outputs` (`model`, `steps`, `citations`, `redacted`, `guardrail`) beside the text, for the live app. A site or unit code the model writes anyway becomes its name before the answer is sent (`guardrails.plain_names`).
- **Its traces** (every LLM and tool call of each question, tagged with the `model` that answered) go to the experiment `heavden-databricks-assistant` in the deploying identity's folder, which keeps them itself (MLflow UI, Review App feedback; up to 100,000 traces). Not in Unity Catalog: a serving endpoint writes traces there only with a token or a service principal secret of its own, and this project gives its identities no tokens or service-principal secrets. Every request and reply, with its full trace, is also in the endpoint's inference table `<catalog>.ml.assistant_payload` (SQL). The deploy fails if the trace of its test question doesn't arrive within 10 minutes or doesn't show both the model's and the tools' calls. The quality test's traces do go to Unity Catalog (`<catalog>.assistant`): it runs from a laptop or CI, which can write there.
- **Compare the models in dev**, which answers from staging's fixed data, from any branch: `databricks bundle deploy -t dev`, `databricks bundle run assistant_deploy -t dev --params models=all`. That deploys one version per model in the `databricks:` list, each checked like any deploy and labelled with its model in Unity Catalog (alias such as `gpt_5_mini`, tag `llm_endpoint`, description); the default model is deployed last and keeps the traffic, and the versions the set replaces are removed from the endpoint and the registry. Without `models=all`, every environment deploys only the default model and removes nothing. The Review App talks to the endpoint (the version with the traffic), not to a chosen version: a model picker there needs one endpoint per model (not built).
- **Red-team it:** `uv run python backend/scripts/redteam_chat.py --endpoint heavden-assistant-staging`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODE` | `demo` | `demo` or `live` |
| `SNAPSHOT_DIR` | `data/demo_snapshot` | snapshot to serve (with `SNAPSHOT_URL`: where it is downloaded to) |
| `SNAPSHOT_URL` | unset | the snapshot folder in Azure Blob Storage, downloaded and checked at every start (needs `MANAGED_IDENTITY_CLIENT_ID`) |
| `MANAGED_IDENTITY_CLIENT_ID` | unset | the API's managed identity on Container Apps: signs in to Databricks, Blob and Table Storage, and Azure OpenAI without secrets |
| `DATABRICKS_HOST`, `LIVE_CATALOG`, `WAREHOUSE_ID`, `GENIE_TITLE` | unset | live mode: the workspace, the catalog (`heavden_prod`), the SQL warehouse, and the Genie space's title |
| `EMBED_CACHE_DIR` | `data/models` | where the embedding model is cached |
| `RAG_DIR` | unset | a prebuilt document index (the image builds one from the documents in git) |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | comma-separated origins for CORS (local dev; in production Vercel proxies `/api/*`) |
| `LLM_PROVIDER` | `openai` | `openai`, `azure` or `databricks`: which service answers the assistant's questions |
| `OPENAI_API_KEY` | unset | enables live assistant answers with `LLM_PROVIDER=openai`. Locally from `.env` (git-ignored), never in git |
| `AZURE_OPENAI_ENDPOINT` | unset | with `LLM_PROVIDER=azure`: the resource's endpoint as the portal shows it, e.g. `https://<resource>.openai.azure.com` |
| `AZURE_OPENAI_API_KEY` | unset | with `LLM_PROVIDER=azure`: the resource's key (`AZURE_OPENAI_KEY` is read too), locally from `.env`. On Container Apps there is no key: the managed identity signs in |
| `DATABRICKS_HOST` | unset | the workspace address, for live mode and for `LLM_PROVIDER=databricks`. The identity is the managed identity on Container Apps, or the Databricks SDK's own variables locally (`DATABRICKS_CONFIG_PROFILE`) |
| `LLM_MODEL` | `gpt-5.5` | OpenAI: the model name. Azure: the name of **your deployment**. Databricks: the serving endpoint. Must be listed in `chat/models.yaml` |
| `LLM_REASONING_EFFORT` | the model's entry in `models.yaml` | overrides it when set; empty leaves the option out |
| `CHAT_REQUESTS_PER_HOUR` | `30` | per-visitor limit on `/chat` |
| `CHAT_LIVE_ANSWERS_PER_DAY` | `100` | live LLM answers per UTC day across all visitors; then recordings |
| `CHAT_BUDGET_TABLE_URL` | unset | the Azure Table that keeps the daily counts: chat answers and warehouse time (Container Apps; needs the managed identity) |
| `WAREHOUSE_SECONDS_PER_DAY` | `7200` | live mode: SQL warehouse time per UTC day for pages and chat together, all visitors combined; past it, pages come from the snapshot and the chat answers from the documents only |
| `TRUSTED_PROXY_HOPS` | `0` | proxies we control in front of the API (0 local, 1 Container Apps alone, 2 Vercel → Container Apps) |
| `ORIGIN_SECRET` | unset | when set, only requests carrying it in `x-origin-secret` are answered (Container Apps: a secret from Terraform) |

### Choosing the LLM provider

Both providers speak the same Chat Completions API, so the assistant's code is the same; only the address and the key differ. Azure's v1 API is served at `<endpoint>/openai/v1/` and needs no `api-version`.

To answer with a model deployed in Microsoft Foundry (Azure OpenAI):

1. In Foundry, deploy a chat model that supports tool calling (**Models + endpoints → Deploy model**) and note the **deployment name** you gave it.
2. In `.env`:

   ```
   LLM_PROVIDER=azure
   AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
   AZURE_OPENAI_API_KEY=<key>
   LLM_MODEL=<deployment name>
   ```

3. Restart the API. To go back, set `LLM_PROVIDER=openai` and `LLM_MODEL` to an OpenAI model name; both keys can stay in `.env`.

Each deployment has its own rate quota and is billed per token to the Azure subscription. The recorded answers and the red-team results in this README were produced with OpenAI's `gpt-5.5`; re-run `backend/scripts/redteam_chat.py` before relying on another model.

To answer with a model **served by Databricks** (no keys: billed as Databricks usage on the Azure bill):

```
LLM_PROVIDER=databricks
DATABRICKS_HOST=https://adb-<workspace-id>.<n>.azuredatabricks.net
DATABRICKS_CONFIG_PROFILE=<your CLI profile>      # locally; on Render a service principal instead
LLM_MODEL=databricks-gpt-oss-120b                 # a databricks entry in the app's list
```

The app's list holds one Databricks-served model, GPT-OSS 120B, for comparison only: it failed the app's scorecard (quality run 3), where the model writes the SQL itself, and passed the Databricks assistant's, where Genie writes it. Another model goes into the list in `chat/models.yaml` after the quality test.

The token is fetched from the Databricks SDK before each request (Databricks tokens expire). Every listed model is called the same way; the differences are recorded per model in `chat/models.yaml`: Databricks endpoints take `max_tokens` (they reject `max_completion_tokens`, which OpenAI and Azure require for gpt-5 models), and some take a `reasoning_effort`. Reasoning models served by Databricks (GPT-OSS, Qwen) return their reasoning and their answer as separate parts; only the answer is shown (`chat/agent.text_of`). On 16 Oct 2026 tool calling worked with GPT-OSS 120B, Qwen3.5 122B and Llama 4 Maverick; Anthropic's Claude models were refused for this account (they need the account upgraded or approved). The model comparison behind the final lists is in `docs/quality/README.md`.

## Snapshot layout

```
manifest.json            as_of ("now"), model name and version, risk-band cut-offs, row counts
gold/<table>.parquet     patient_hour_features, risk_scores, alerts_fact, site_kpis_hourly,
                         device_health_daily, encounters (masked patient labels, no names)
model/model.joblib       the champion; model/background.parquet for explanations
rag/                     chunks.jsonl, vectors.npy, meta.json
```

The website's snapshot is exported from prod on Databricks (`heavden_api/export.py`: the last 28 days of Gold, the model in use and its check hours) into Azure Blob Storage, with a pointer file naming the snapshot to serve.
