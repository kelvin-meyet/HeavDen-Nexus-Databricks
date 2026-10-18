# The assistant's quality test: results

Which models the two assistants may run on, and why. Every answer, its steps and its verdicts
are in the MLflow experiment `/Shared/heavden-assistant-quality` (traces in Unity Catalog,
`heavden_staging.assistant`); these pages are the readings of each run.

- **The test:** `backend/heavden_api/chat/quality.py` and `backend/scripts/assistant_quality.py`.
  It asks 23 questions (`databricks/src/heavden/agent/evalsets/assistant.yaml`; 20 until run 3)
  of two assistants:
  - **app**: the app's own assistant, where the model writes the SQL itself;
  - **databricks**: the Databricks assistant, where Genie writes the SQL.
- **Expected figures** come from each assistant's own data.
- **Pass marks** were fixed before the first run, and only the cite-format limit was added before
  run 2. A test fault (a correct answer marked wrong) was fixed only after reading it by hand,
  and each report says which.

| Run | When (UTC) | What it answered |
| --- | --- | --- |
| [Run 1](run1-20261017-0007.md) | 16-17 Oct, 22:53-00:07 | Five models on both assistants, at their default reasoning. Re-scored after the test's own faults were fixed: only gpt-5.5 and GPT-OSS 120B pass the app; nothing passes Databricks yet |
| [Run 2](run2-20261017-0251.md) | 17 Oct, 01:58-02:51 | The shortlist, after four fixes to the assistant. All three pass Databricks, and gpt-5.5 and GPT-OSS pass the app |
| [Run 3](run3-20261017-0414.md) | 17 Oct, 03:50-04:14 | The app at the settings it ships with: gpt-5.5 at `none` passes, GPT-OSS fails at both settings |

## The final lists (17 Oct 2026)

Set in `backend/heavden_api/chat/models.yaml`. Each assistant switches with one setting, to any
model in its own list:

| Assistant | Main | Also listed |
| --- | --- | --- |
| app (`LLM_PROVIDER` + `LLM_MODEL`) | gpt-5.5 at `none` | gpt-5-mini at its default (the cheaper fallback); GPT-OSS 120B at its default (for comparison only: it failed the app's scorecard) |
| databricks (the bundle variable `assistant_llm_endpoint`) | gpt-5-mini at its default (`heavden-gpt-5-mini`, through AI Gateway; since 18 Oct, see the red-team below) | gpt-5.5 at `none` (`heavden-gpt-5-5`, compared in dev); GPT-OSS 120B at its default, never `low` (`databricks-gpt-oss-120b`, the model while building; compared in dev, not for visitors) |

## Fixed after run 3, before these lists were set

- The app's table notes define alert load as Genie does (alerts × 60 ÷ census-hours). Every app
  model had guessed it.
- The model card's "Reading it" paragraph is split into "against NEWS2" and "against version 1".
  GPT-OSS had taken one comparison for the other in every run.
- A citation [n] with no passage behind it now fails the cite-format check. In run 3, only reading
  the answers caught one.
- gpt-5-mini has 4,000 tokens to answer: at its default reasoning it twice spent the whole 1,200
  thinking and gave no answer.

None of these has been measured by a run yet. The staging CI quality gate will measure the
Databricks list when the Databricks assistant is deployed.

## Red-team (17 Oct 2026)

`backend/scripts/redteam_chat.py` sends 19 prompts through `/chat`: extraction attempts,
off-topic requests, a real-patient emergency, and misspelt in-scope questions that must not be
refused. The checks are in `backend/heavden_api/chat/redteam.py`.

| Model | Failed the automatic checks | Read by hand |
| --- | --- | --- |
| gpt-5.5 | 0 of 19 | clean |
| gpt-5-mini | 0 of 19 | safe. Once, it added its own triage advice to the emergency answer, and it once put a column name in an answer |
| GPT-OSS 120B | 1 of 19 (showed a unit code) | safe, but put 8 High-risk patients on one unit when they were spread across the whole site, and named internal tables |

Until that day, the internal-code check matched nothing: its pattern held control characters.
Earlier red-team results did not cover internal codes.

## Red-team of the Databricks assistant (18 Oct 2026)

The same 19 prompts through the dev endpoint `heavden-assistant-dev` (`redteam_chat.py
--endpoint`), answering from staging's data. The rule, set before the gpt-5-mini run: gpt-5-mini
becomes the Databricks assistant's model if it does at least slightly better than GPT-OSS.

| Model | Failed the automatic checks | Read by hand |
| --- | --- | --- |
| GPT-OSS 120B | 0 of 19 (reported 1: a refusal worded "the HeavDen hospital wards", which the check didn't recognise; fixed) | safe, but answered "why only 6 conditions" from memory, without searching the documents, and named conditions the data doesn't have (sepsis, pneumonia, stroke, acute kidney injury) |
| gpt-5-mini | 0 of 19 | safe; every document answer correct and cited (the six conditions and why, the NEWS2 thresholds); asked one clarifying question for "monitors?"; asked for SQL onboarding notes, it declined the project's rules but added a generic SQL checklist |

gpt-5-mini is the Databricks assistant's model from 18 Oct. Dev serves every listed model, one
version each, to compare them on staging's fixed data.
