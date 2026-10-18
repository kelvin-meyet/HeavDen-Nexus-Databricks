"""Run the assistant's quality test: the questions, each model, two targets (Plan.md §11.2).

Every answer, its steps and its verdicts go to the MLflow experiment `--experiment`, whose traces
are stored in Unity Catalog (`<catalog>.assistant`); the two scorecards are printed and saved.
By default each target runs its own list from chat/models.yaml as shipped; `--models` picks from
that list, as `name@effort` (e.g. gpt-5.5@none), `name@shipped` or `name@default` (the model's
own default reasoning, for comparing models). Keys for OpenAI / Azure models come from `.env`;
Databricks-hosted models, Genie, the warehouse and the judge use the CLI profile.

    uv run --group agent python backend/scripts/assistant_quality.py
    uv run --group agent python backend/scripts/assistant_quality.py --targets app \
        --models gpt-5-mini@shipped gpt-5.5@default

Writes to Databricks: the experiment (created once), its runs, and the trace tables MLflow
creates in `<catalog>.assistant`. Reads staging's Gold through Genie and the SQL warehouse.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def make_llm(spec: str, host: str, assistant: str):
    """`spec` = a name from `assistant`'s list in chat/models.yaml, at its default reasoning;
    `name@effort` sends that reasoning effort, and `name@shipped` the one models.yaml gives it
    (what the assistant uses). Returns the model and the reasoning label recorded with the run."""
    from heavden_api.chat.agent import OpenAILLM
    from heavden_api.chat.models import load_models
    from heavden_api.config import Settings

    name, _, effort = spec.partition("@")
    listed = load_models(assistant)
    if name not in listed:
        raise SystemExit(f"{name} is not in the {assistant} list of chat/models.yaml")
    model = listed[name]
    if effort == "shipped":
        effort = model.reasoning_effort or "default"
    reasoning = None if effort in ("", "default") else effort
    connection = Settings(
        llm_provider=model.provider, llm_model=name, databricks_host=host
    ).llm_connection()
    if connection is None:
        raise SystemExit(f"{name}: no key for {model.provider} in .env")
    limit = {"max_tokens": model.max_tokens} if model.max_tokens else {}
    llm = OpenAILLM(
        connection.api_key, name, reasoning, connection.base_url, model.token_limit, **limit
    )
    return llm, reasoning or "default"


def experiment(name: str, catalog: str) -> str:
    """The experiment's id; created once with its traces in `<catalog>.assistant`."""
    import mlflow
    from mlflow.entities.trace_location import UnityCatalog

    location = UnityCatalog(catalog_name=catalog, schema_name="assistant")
    found = mlflow.get_experiment_by_name(name)
    if found is None:
        experiment_id = mlflow.create_experiment(name, trace_location=location)
    else:
        bound = found.trace_location
        if not isinstance(bound, UnityCatalog) or (bound.catalog_name, bound.schema_name) != (
            catalog,
            "assistant",
        ):
            raise SystemExit(f"{name} stores its traces in {bound}, not {catalog}.assistant")
        experiment_id = found.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def main(argv: list[str] | None = None) -> None:
    from heavden_api.chat.quality import CONTESTANTS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets", nargs="+", default=["app", "databricks"])
    parser.add_argument(
        "--models", nargs="+", help="from each target's list (default: the whole list, as shipped)"
    )
    parser.add_argument("--profile", default="azure-heavden")
    parser.add_argument("--catalog", default="heavden_staging")
    parser.add_argument("--warehouse", default="0a14d0d391ce8af6")
    parser.add_argument("--genie-title", default="HeavDen Ward Data (staging)")
    parser.add_argument("--vector-search-index", help="else the packaged document index")
    parser.add_argument("--experiment", default="/Shared/heavden-assistant-quality")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "quality")
    parser.add_argument("--gate", action="store_true", help="exit 1 if any scorecard fails (CI)")
    args = parser.parse_args(argv)

    import mlflow
    from databricks.sdk import WorkspaceClient

    from heavden.agent.genie import find_space
    from heavden_api.chat import quality
    from heavden_api.chat.databricks_tools import (
        VectorSearchDocuments,
        build_document_index,
        packaged_documents,
    )
    from heavden_api.config import Settings, load_dotenv
    from heavden_api.deps import build_state

    load_dotenv()
    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = args.warehouse
    os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"  # Genie and the rate limits, one at a time
    client = WorkspaceClient(profile=args.profile)
    api, host = client.api_client, client.config.host.rstrip("/")
    mlflow.set_tracking_uri(f"databricks://{args.profile}")
    experiment_id = experiment(args.experiment, args.catalog)
    mlflow.openai.autolog()  # every LLM call becomes a span in the question's trace

    targets = {}
    if "app" in args.targets:
        # the app's data and tools without its own LLM (each run brings one), whatever .env picks
        no_llm = Settings(llm_provider="openai", openai_api_key=None)
        targets["app"] = quality.app_target(build_state(no_llm))
    if "databricks" in args.targets:
        space = find_space(api, args.genie_title)
        if space is None:
            raise SystemExit(f"no Genie space titled {args.genie_title!r}")
        if args.vector_search_index:
            from heavden.agent.corpus import chunk_corpus, load_corpus
            from heavden.agent.publish import corpus_dir

            chunks = chunk_corpus(load_corpus(corpus_dir()))
            documents = VectorSearchDocuments(api, args.vector_search_index, chunks)
        else:
            index = ROOT / "data" / "agent_docs"
            if not (index / "meta.json").exists():
                print(build_document_index(api, index))
            documents = packaged_documents(api, index)
        targets["databricks"] = quality.databricks_target(
            api, args.catalog, args.warehouse, space, documents
        )

    cards = []
    for target in targets.values():
        for model in args.models or CONTESTANTS[target.name]:
            start = time.monotonic()
            print(f"{target.name} · {model} ...", flush=True)
            llm, reasoning = make_llm(model, host, target.name)
            label = model.partition("@")[0] + ("" if reasoning == "default" else f"@{reasoning}")
            card = quality.evaluate(target, label, llm, reasoning=reasoning)
            cards.append(card)
            print(f"  {card.row()}  ({time.monotonic() - start:.0f} s)", flush=True)

    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    report = [
        f"# Assistant quality test, {stamp}",
        f"Experiment: {host}/ml/experiments/{experiment_id}",
    ]
    for name, target in targets.items():
        report += ["", f"## {name} (now = {target.now})", "", quality.header()]
        report += [c.row() for c in cards if c.target == name]
    report += [
        "",
        "Misses allowed: " + ", ".join(f"{k} {v}" for k, v in quality.MAX_MISSES.items()),
    ]
    text = "\n".join(report)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"quality-{datetime.now(UTC):%Y%m%d-%H%M}.md"
    path.write_text(text + "\n", encoding="utf-8")
    print("\n" + text + f"\n\nSaved to {path}")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(summary).write_text(text + "\n", encoding="utf-8")
    if args.gate and not all(c.passed for c in cards):
        failed = [c.model for c in cards if not c.passed]
        raise SystemExit(f"quality gate failed: {', '.join(failed)}")


if __name__ == "__main__":
    sys.exit(main())
