"""Log, register and deploy the Databricks assistant (bundle job `assistant_deploy`,
`databricks/resources/assistant_deploy.job.yml`), then ask it one question.

1. Find the Genie space by its title; refuse a model not in the `databricks:` list.
2. Build the packaged document index (every chunk, embedded by the endpoint Vector Search uses).
3. For each model to deploy (`--models`): the default one, or every listed model with the default
   last (`all`, dev's comparison set), one version each:
   a. Log the model from code (`assistant_model.py`) with the index as an artifact, the code of
      `heavden` and `heavden_api`, pinned requirements (the versions this job runs, so the example
      answered while logging is answered by the same libraries), and the resources it calls, from
      which Model Serving gives the endpoint its credentials. Register it as
      `<catalog>.ml.assistant` and label the version with its model (alias, tag, description).
   b. `agents.deploy` to the endpoint (scale to zero), with the settings as environment variables
      and the experiment that receives its traces.
   c. Wait until it serves, ask that version one question, read its trace in the experiment, print
      `ASSISTANT_DEPLOY_RESULT {json}`; fail, before any other model, if it answered without its
      tools or its trace doesn't show the model and tool calls.
4. With `all`: remove the versions the set replaces from the endpoint and the registry.

The experiment keeps its traces itself, not in Unity Catalog: a serving endpoint writes traces to
Unity Catalog tables only with a token or a service principal's secret of its own, while the
experiment's own store is what `agents.deploy` sets up (no secret; up to 100,000 traces). Every
reply, with its trace, is also in the endpoint's inference table `<catalog>.ml.assistant_payload`.

Two catalogs: the data the assistant reads (`--data-catalog`) and where its model lives
(`--catalog`). They are the same in staging and prod; dev reads staging's data (dev has no scores).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import UTC, datetime, timedelta
from importlib import metadata
from pathlib import Path

from heavden_api.chat.databricks_assistant import SETTINGS, Settings
from heavden_api.chat.endpoint import ask
from heavden_api.chat.models import load_models, profile

MODEL_FILE = Path(__file__).with_name("assistant_model.py")
MODEL_CALL, TOOL_CALL = "CHAT_MODEL", "TOOL"  # the span types a trace must show
QUESTION = "How many patients are in the High band at each site right now?"
# What the served model installs, at the versions of the job that logs it
SERVED_PACKAGES = [
    "mlflow",
    "databricks-sdk",
    "openai",
    "duckdb",
    "faiss-cpu",
    "numpy",
    "pandas",
    "pyyaml",
]
# Gold tables it reads besides Genie's: "now" and the patient function
OWN_TABLES = {"risk_scores", "encounters"}


def requirements() -> list[str]:
    def installed(name: str) -> str:
        if name == "mlflow":  # jobs have mlflow-skinny; the endpoint gets full MLflow, same version
            return metadata.version("mlflow-skinny")
        return metadata.version(name)

    return [f"{name}=={installed(name)}" for name in SERVED_PACKAGES]


def environment(settings: Settings) -> dict[str, str]:
    """The settings as the endpoint's environment variables (`databricks_assistant.SETTINGS`)."""
    values = {k: getattr(settings, k) for k in SETTINGS}
    return {SETTINGS[k]: v for k, v in values.items() if v}


def resources(settings: Settings, tables: list[str]) -> list:
    """Everything the endpoint calls, so Model Serving can give it credentials: every model in the
    `databricks:` list (switching LLM_ENDPOINT then needs no new model version), the embedding
    endpoint, the Genie space and its tables, the warehouse, the patient function, and the Vector
    Search index when there is one."""
    from mlflow.models.resources import (
        DatabricksFunction,
        DatabricksGenieSpace,
        DatabricksServingEndpoint,
        DatabricksSQLWarehouse,
        DatabricksTable,
        DatabricksVectorSearchIndex,
    )

    from heavden_api.chat.databricks_tools import EMBEDDING_ENDPOINT

    endpoints = [*load_models("databricks"), EMBEDDING_ENDPOINT]
    found = [DatabricksServingEndpoint(endpoint_name=e) for e in endpoints]
    found += [
        DatabricksGenieSpace(genie_space_id=settings.genie_space),
        DatabricksSQLWarehouse(warehouse_id=settings.warehouse_id),
        DatabricksFunction(function_name=f"{settings.catalog}.ml.explain_patient_risk"),
        *[DatabricksTable(table_name=t) for t in tables],
    ]
    if settings.vector_index:
        found.append(DatabricksVectorSearchIndex(index_name=settings.vector_index))
    return found


def endpoint_experiment(name: str) -> str:
    """The id of the experiment for the model's runs and the endpoint's traces, created once and
    keeping its traces itself: one bound to Unity Catalog would never receive them."""
    import mlflow

    found = mlflow.get_experiment_by_name(name)
    if found is not None and found.trace_location is not None:
        raise SystemExit(
            f"{name} keeps its traces in {found.trace_location}, where the endpoint can't write "
            "them; deploy with another experiment name"
        )
    experiment_id = found.experiment_id if found else mlflow.create_experiment(name)
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def models_to_deploy(choice: str, default: str) -> list[str]:
    """One version per model: the default only, or (`all`, dev's comparison set) every model in
    the `databricks:` list, the default last so it keeps the endpoint's traffic."""
    if choice == "default":
        return [default]
    if choice != "all":
        raise SystemExit(f"--models is default or all, not {choice!r}")
    return [m for m in load_models("databricks") if m != default] + [default]


def alias_for(llm_endpoint: str) -> str:
    """The alias naming a version's model: heavden-gpt-5-mini -> gpt_5_mini."""
    return re.sub(r"^(databricks|heavden)-", "", llm_endpoint).replace("-", "_")


def label_version(client, name: str, version: str, llm_endpoint: str) -> None:
    """Which model a version answers with, where Unity Catalog shows it: an alias named after the
    model (moved to this version, the newest on it), a tag and the description."""
    client.set_registered_model_alias(name, alias_for(llm_endpoint), version)
    client.set_model_version_tag(name, version, "llm_endpoint", llm_endpoint)
    client.update_model_version(
        name, version, description=f"Databricks assistant on {llm_endpoint}"
    )


def stale_versions(served_entities, model_name: str) -> list[int]:
    """Versions of `model_name` on the endpoint with a newer version on the same model
    (`LLM_ENDPOINT`): what a comparison set replaces."""
    served = [
        (int(e.entity_version), (e.environment_vars or {}).get("LLM_ENDPOINT", ""))
        for e in served_entities
        if e.entity_name == model_name
    ]
    newest: dict[str, int] = {}
    for version, llm in served:
        newest[llm] = max(version, newest.get(llm, 0))
    return sorted(version for version, llm in served if version < newest[llm])


def wait_until_serving(workspace, endpoint: str, timeout: float = 2700.0) -> float:
    """Seconds until the endpoint serves its new configuration (ready, no update in progress)."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        state = workspace.serving_endpoints.get(endpoint).state
        ready = getattr(state.ready, "value", state.ready)
        update = getattr(state.config_update, "value", state.config_update)
        if update == "UPDATE_FAILED":
            raise RuntimeError(f"{endpoint}: the new configuration failed; see its events and logs")
        if ready == "READY" and update == "NOT_UPDATING":
            return time.monotonic() - start
        time.sleep(30)
    raise TimeoutError(f"{endpoint} not serving after {timeout / 60:.0f} min")


def trace_steps(experiment_id: str, since: datetime, timeout: float = 600.0) -> list[tuple]:
    """(span type, name) of the model and tool calls, in order, in the newest trace the experiment
    received since `since` (a retried question leaves two; the newest is the retry's), waiting up
    to `timeout` for one whose spans can be read; empty if none came."""
    import mlflow

    since_ms = int(since.timestamp() * 1000)
    deadline = time.monotonic() + timeout
    while True:
        latest = mlflow.search_traces(
            locations=[experiment_id], max_results=100, return_type="list", include_spans=False
        )  # newest first
        found = [t for t in latest if t.info.timestamp_ms >= since_ms]
        # a trace is listed before its spans can be read (None until then: 18 Oct, version 6)
        trace = mlflow.get_trace(found[0].info.trace_id) if found else None
        if trace is not None:
            in_order = sorted(trace.data.spans, key=lambda s: s.start_time_ns)
            return [
                (s.span_type, s.name) for s in in_order if s.span_type in (MODEL_CALL, TOOL_CALL)
            ]
        if time.monotonic() >= deadline:
            return []
        time.sleep(30)


def check_reply(llm_endpoint: str, answer: str, tools: list, steps: list[tuple]) -> None:
    """Stop the deploy unless the version answered with its tools and its trace shows both the
    model's calls and the tools' (each a span: `databricks_assistant.TracedToolbox`)."""
    if not answer or not tools:
        raise SystemExit(f"{llm_endpoint}: the endpoint answered without using its tools")
    if not steps:
        raise SystemExit(f"{llm_endpoint}: no trace of its answer reached the experiment")
    if not {MODEL_CALL, TOOL_CALL} <= {kind for kind, _ in steps}:
        names = [name for _, name in steps]
        raise SystemExit(f"{llm_endpoint}: its trace lacks the model or tool calls: {names}")


def prune(workspace, client, name: str, endpoint: str) -> list[int]:
    """Remove the versions a comparison set replaced from the endpoint, one endpoint update at a
    time, and from the registry."""
    from databricks import agents

    stale = stale_versions(workspace.serving_endpoints.get(endpoint).config.served_entities, name)
    for version in stale:
        agents.delete_deployment(name, version)
        wait_until_serving(workspace, endpoint)
        client.delete_model_version(name, str(version))
    return stale


def deploy_model(
    workspace,
    client,
    settings: Settings,
    name: str,
    endpoint: str,
    experiment_id: str,
    documents: str,
    tables: list[str],
) -> dict:
    """Log, register and label one version answering with `settings.llm_endpoint`, deploy it, ask
    it one question and check its answer and trace (`check_reply`)."""
    import mlflow

    import heavden
    import heavden_api
    from databricks import agents

    llm = settings.llm_endpoint
    env = environment(settings)
    os.environ.update(env)  # the example answered while logging goes through the real assistant
    with mlflow.start_run(run_name=f"assistant · {llm}"):
        info = mlflow.pyfunc.log_model(
            name="assistant",
            python_model=str(MODEL_FILE),
            artifacts={"documents": documents},
            code_paths=[str(Path(p.__file__).parent) for p in (heavden, heavden_api)],
            resources=resources(settings, tables),
            pip_requirements=requirements(),
            input_example={"input": [{"role": "user", "content": QUESTION}]},
            registered_model_name=name,
        )
    version = info.registered_model_version
    label_version(client, name, version, llm)
    print(f"{name} version {version} logged, registered and labelled: {llm} ({alias_for(llm)})")

    # The experiment named explicitly: for an existing endpoint, agents.deploy would otherwise
    # keep the one in the endpoint's MONITOR_EXPERIMENT_ID tag
    tags = {"project": "heavden", "component": "assistant"}
    agents.deploy(
        name,
        int(version),
        endpoint_name=endpoint,
        scale_to_zero=True,
        environment_vars={**env, "MLFLOW_EXPERIMENT_ID": experiment_id},
        tags={**tags, "MONITOR_EXPERIMENT_ID": experiment_id},
    )
    waited = wait_until_serving(workspace, endpoint)
    served = workspace.serving_endpoints.get(endpoint).config.served_entities
    served_model = next(e.name for e in served if e.entity_version == str(version))
    asked_at = datetime.now(UTC) - timedelta(minutes=1)  # after the trace of logging's example
    start = time.monotonic()
    answer, outputs = ask(workspace.api_client, endpoint, QUESTION, served_model)
    seconds = round(time.monotonic() - start, 1)
    steps = trace_steps(experiment_id, asked_at)
    result = {
        "endpoint": endpoint,
        "model": name,
        "version": version,
        "llm_endpoint": llm,
        "minutes_to_serve": round(waited / 60, 1),
        "seconds_to_answer": seconds,
        "question": QUESTION,
        "answer": answer,
        "tools": [s.get("tool") for s in outputs.get("steps", [])],
        "trace": [step for _, step in steps],
        "experiment": experiment_id,
    }
    print("ASSISTANT_DEPLOY_RESULT " + json.dumps(result))
    check_reply(llm, answer, result["tools"], steps)
    return result


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True, help="the model's: heavden_<env>")
    parser.add_argument("--data-catalog", required=True, help="the Gold it answers from")
    parser.add_argument("--warehouse-id", required=True)
    parser.add_argument("--genie-title", required=True, help='e.g. "HeavDen Ward Data (staging)"')
    parser.add_argument("--endpoint-name", required=True, help="e.g. heavden-assistant-staging")
    parser.add_argument("--llm-endpoint", required=True, help="the default `databricks:` model")
    parser.add_argument(
        "--models", default="default", help="default, or all: one version per listed model"
    )
    parser.add_argument(
        "--experiment", required=True, help="the model's runs and the endpoint's traces"
    )
    parser.add_argument("--vector-index", default="", help="else the packaged document index")
    args = parser.parse_args(argv)

    import mlflow
    from databricks.sdk import WorkspaceClient
    from mlflow import MlflowClient

    from heavden.agent import genie
    from heavden_api.chat.databricks_tools import build_document_index

    workspace = WorkspaceClient()
    space = genie.find_space(workspace.api_client, args.genie_title)
    if space is None:
        raise SystemExit(f"no Genie space titled {args.genie_title!r}")
    profile(args.llm_endpoint, "databricks", assistant="databricks")  # refuses an unlisted model
    models = models_to_deploy(args.models, args.llm_endpoint)
    gold = sorted(set(genie.load_spec()["tables"]) | OWN_TABLES)
    tables = [f"{args.data_catalog}.gold.{t}" for t in gold]

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    experiment_id = endpoint_experiment(args.experiment)
    name = f"{args.catalog}.ml.assistant"
    deployed = {}
    with tempfile.TemporaryDirectory() as folder:
        print(build_document_index(workspace.api_client, folder))
        for llm in models:
            settings = Settings(
                llm, args.data_catalog, args.warehouse_id, space, args.vector_index or None
            )
            result = deploy_model(
                workspace, client, settings, name, args.endpoint_name, experiment_id, folder, tables
            )
            deployed[result["version"]] = llm
    if args.models == "all":
        removed = prune(workspace, client, name, args.endpoint_name)
        print(f"Removed from {args.endpoint_name} and the registry: versions {removed or 'none'}")
    print(
        f"On {args.endpoint_name}: "
        + ", ".join(f"version {v} = {m}" for v, m in deployed.items())
        + f"; {args.llm_endpoint} has the traffic"
    )
