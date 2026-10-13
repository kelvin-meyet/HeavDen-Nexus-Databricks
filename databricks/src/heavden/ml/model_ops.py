"""Model operations with an audit trail: promote a challenger, roll the champion back, and
record each decision in `<catalog>.ml.ops_audit` (Plan.md §10.4).

The registry's aliases are the deployment: batch scoring and the drift check load `@champion`,
so moving that alias is the release, and moving it back is the rollback. Nothing is copied or
rebuilt, and hours already scored keep the version that scored them.

    @champion           the model in use
    @challenger         a retrained model that passed its gates and waits for a decision
    @previous_champion  the champion before the last promotion or rollback (the rollback target)

Decisions are made on GitHub (an approval or a rejection in a protected environment, with a
comment) and carried out here by the job `model_ops`, started by the workflow that was approved:

    promote            needs a version that passed its gates **against the current champion**
    rollback           back to @previous_champion (or a named version that passed its gates)
    record-rejections  rejected requests, read from GitHub by the next approved prod run. A
                       rejected request never gets credentials for prod, so it can't write its
                       own row; that is by design
    status, selftest   read the aliases; check that the registry accepts alias changes

`ml.ops_audit` has one row per event: what was asked (`event`), what happened (`outcome`), which
versions, who decided and why, the drift check that led to it, and the GitHub run. The training
job writes the `retrain` rows itself (`heavden.ml.training`).
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import uuid
from datetime import UTC, datetime

RESULT_MARKER = "MODEL_OPS_RESULT"
ALIASES = ("champion", "challenger", "previous_champion")
ACTIONS = ("status", "selftest", "promote", "rollback", "record-rejections")
SELFTEST_ALIAS = "selftest_alias"
# How registries say "this alias isn't set" (the local store and Unity Catalog differ).
NOT_FOUND = ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE", "NOT_FOUND")

# Explicit, so rows with empty columns (no check, no reason) still write typed columns.
AUDIT_SCHEMA = (
    "event_id string, event_ts timestamp, recorded_at timestamp, event string, outcome string, "
    "model string, version string, from_version string, actor string, reason string, "
    "check_id string, episode string, run_url string, details string"
)
AUDIT_COLUMNS = [field.split()[0] for field in AUDIT_SCHEMA.split(", ")]

# Requests carry what they are about in their run name on GitHub (set by the workflows).
REQUEST_TITLE = re.compile(r"^(Retrain|Promotion|Rollback) request:")
_CHECK = re.compile(r"check ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
_EPISODE = re.compile(r"episode ([\w-]+)")
_VERSION = re.compile(r"version (\d+)")


def to_json(value) -> str:
    """JSON with NaN and infinity written as null (plain `json.dumps` writes NaN, which strict
    readers such as jq and Spark's JSON functions reject)."""

    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list | tuple):
            return [clean(x) for x in v]
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    return json.dumps(clean(value))


def encode_context(context: dict) -> str:
    """A dict as one word of base64: job parameters travel as `key=value,key=value`, which
    free text (a review comment with commas or quotes) would break."""
    return base64.b64encode(json.dumps(context).encode()).decode()


def decode_context(text: str | None) -> dict:
    return json.loads(base64.b64decode(text)) if text else {}


def aliases(client, name: str) -> dict[str, str]:
    """Which version each known alias points at (aliases that aren't set are left out).

    Only "not found" counts as not set. Any other failure is raised: read as a missing champion,
    a permission or network error would let a promotion skip its check against the champion."""
    from mlflow.exceptions import MlflowException

    found = {}
    for alias in ALIASES:
        try:
            found[alias] = str(client.get_model_version_by_alias(name, alias).version)
        except MlflowException as e:
            if e.error_code not in NOT_FOUND:
                raise
    return found


def _passed_gates(client, name: str, version: str):
    """The version, if it passed its gates; a version that failed them is never put in use."""
    mv = client.get_model_version(name, version)
    if mv.tags.get("passed_gates") != "true":
        raise ValueError(f"version {version} did not pass its gates")
    return mv


def promote(client, name: str, version: str | None = None) -> dict:
    """Point `@champion` at a challenger. Returns the versions involved and its gate results.

    The version must have passed its gates in a comparison with the champion that is in use
    now: a challenger that beat an earlier champion proves nothing about the current one."""
    now = aliases(client, name)
    target = str(version or now.get("challenger") or "")
    champion = now.get("champion")
    if not target:
        raise ValueError("nothing to promote: no version given and no @challenger")
    if target == champion:
        raise ValueError(f"version {target} is already @champion")
    run = client.get_run(_passed_gates(client, name, target).run_id)
    compared = run.data.tags.get("compared_with_version")
    if champion is not None and compared != champion:
        beaten = f"version {compared}" if compared else "no champion (it was not a retrain)"
        raise ValueError(
            f"version {target} was compared with {beaten}, but @champion is version {champion}: "
            "retrain, so it is judged against the model in use"
        )
    client.set_registered_model_alias(name, "champion", target)
    if champion is not None:
        client.set_registered_model_alias(name, "previous_champion", champion)
    if now.get("challenger") == target:
        client.delete_registered_model_alias(name, "challenger")
    prefix = "vs_champion_"
    gates = {k[len(prefix) :]: v for k, v in run.data.metrics.items() if k.startswith(prefix)}
    return {"version": target, "from_version": champion, "gates": gates}


def rollback(client, name: str, version: str | None = None) -> dict:
    """Point `@champion` back at `@previous_champion` (or at `version`). The version rolled
    back from becomes `@previous_champion`, so a rollback can itself be undone."""
    now = aliases(client, name)
    target = str(version or now.get("previous_champion") or "")
    champion = now.get("champion")
    if not target:
        raise ValueError("nothing to roll back to: no version given and no @previous_champion")
    if target == champion:
        raise ValueError(f"version {target} is already @champion")
    _passed_gates(client, name, target)
    client.set_registered_model_alias(name, "champion", target)
    if champion is not None:
        client.set_registered_model_alias(name, "previous_champion", champion)
    return {"version": target, "from_version": champion}


def status(client, name: str, newest: int = 20) -> dict:
    """The aliases, and the newest versions with whether each passed its gates."""
    numbers = sorted(int(v.version) for v in client.search_model_versions(f"name = '{name}'"))
    versions = []
    for number in numbers[-newest:]:
        # each version is fetched by itself: a search doesn't return tags in every registry
        tags = client.get_model_version(name, str(number)).tags
        versions.append({"version": str(number), "passed_gates": tags.get("passed_gates")})
    return {"aliases": aliases(client, name), "versions": versions}


def selftest(client, name: str) -> dict:
    """Try, where it is safe, everything a promotion relies on: reading the aliases whether set
    or not, reading a version's `passed_gates` tag and its run's tags, and setting, reading and
    removing a throwaway alias. Run in staging after every merge, so the first real promotion
    in prod is not also the first time the registry is asked."""
    current = status(client, name)
    if not current["versions"]:
        raise LookupError(f"{name} has no versions: run the training job first")
    newest = current["versions"][-1]
    if newest["passed_gates"] not in ("true", "false"):
        raise RuntimeError(f"version {newest['version']}: the passed_gates tag can't be read")
    run_id = client.get_model_version(name, newest["version"]).run_id
    run_tags = client.get_run(run_id).data.tags
    client.set_registered_model_alias(name, SELFTEST_ALIAS, newest["version"])
    read = str(client.get_model_version_by_alias(name, SELFTEST_ALIAS).version)
    client.delete_registered_model_alias(name, SELFTEST_ALIAS)
    if read != newest["version"]:
        raise RuntimeError(f"alias read back as version {read}, set on {newest['version']}")
    checked = {"alias_round_trip": "ok", "on_version": read, "run_mode": run_tags.get("mode")}
    return checked | current


# --- The audit trail -------------------------------------------------------------------------


def audit_row(
    event: str,
    outcome: str,
    model: str,
    version: str | None,
    from_version: str | None,
    context: dict,
    details: dict | None = None,
    event_ts: datetime | None = None,
    event_id: str | None = None,
) -> dict:
    """One row of `ml.ops_audit`. `context` is what the approved workflow passed along: who
    decided (`actor`), their comment (`reason`), the GitHub run (`run_url`) and, when the
    request came from a drift check, its `check_id` and `episode`."""
    now = datetime.now(UTC)
    return {
        "event_id": event_id or str(uuid.uuid4()),
        "event_ts": event_ts or now,
        "recorded_at": now,
        "event": event,
        "outcome": outcome,
        "model": model,
        "version": None if version is None else str(version),
        "from_version": None if from_version is None else str(from_version),
        "actor": context.get("actor") or None,
        "reason": context.get("reason") or None,
        "check_id": context.get("check_id") or None,
        "episode": context.get("episode") or None,
        "run_url": context.get("run_url") or None,
        "details": to_json(details) if details else None,
    }


def rejection_rows(model: str, reviews: list[dict], known: set[str]) -> list[dict]:
    """Audit rows for requests rejected on GitHub that the table doesn't hold yet.

    Each review is `{workflow, run_id, title, run_url, decided_at, environment, actor,
    comment}`. The run's title says what was asked (the workflows set it): the drift check and
    episode of a retrain request, the version of a promotion request. Runs with another title
    are not requests (e.g. tests from before the titles existed) and are left out."""
    rows = []
    for r in reviews:
        event_id = f"github-{r['run_id']}-{r['environment']}"
        title = r["title"]
        if event_id in known or not REQUEST_TITLE.match(title):
            continue
        context = {
            "actor": r["actor"],
            "reason": r.get("comment"),
            "run_url": r["run_url"],
            "check_id": _found(_CHECK, title),
            "episode": _found(_EPISODE, title),
        }
        rows.append(
            audit_row(
                r["workflow"],
                "rejected",
                model,
                _found(_VERSION, title),
                None,
                context,
                {"title": title, "environment": r["environment"]},
                datetime.fromisoformat(r["decided_at"].replace("Z", "+00:00")),
                event_id,
            )
        )
    return rows


def _found(pattern: re.Pattern, text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def audit_frame(spark, rows: list[dict]):
    """The rows as a Spark DataFrame with the table's schema."""
    return spark.createDataFrame(
        [tuple(row[c] for c in AUDIT_COLUMNS) for row in rows], AUDIT_SCHEMA
    )


def write_audit(spark, catalog: str, rows: list[dict]) -> None:
    """Append rows to `<catalog>.ml.ops_audit` (created on first use)."""
    if rows:
        audit_frame(spark, rows).write.mode("append").saveAsTable(f"{catalog}.ml.ops_audit")


def known_event_ids(spark, catalog: str) -> set[str]:
    table = f"{catalog}.ml.ops_audit"
    if not spark.catalog.tableExists(table):
        return set()
    return {row.event_id for row in spark.read.table(table).select("event_id").collect()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True, help="e.g. heavden_prod")
    parser.add_argument("--action", choices=ACTIONS, default="status")
    # these two may arrive empty: the job parameters default to ""
    parser.add_argument("--version", nargs="?", default="", const="", help="default: the alias")
    parser.add_argument("--context", nargs="?", default="", const="", help="base64 JSON")
    args = parser.parse_args(argv)

    import mlflow
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient()
    name = f"{args.catalog}.ml.deterioration_risk"
    context = decode_context(args.context)

    if args.action in ("promote", "rollback"):
        change = (promote if args.action == "promote" else rollback)(
            client, name, args.version or None
        )
        outcome = "promoted" if args.action == "promote" else "rolled_back"
        row = audit_row(
            args.action,
            outcome,
            name,
            change["version"],
            change["from_version"],
            context,
            {"gates": change["gates"]} if change.get("gates") else None,
        )
        write_audit(spark, args.catalog, [row])
        result = change | {"outcome": outcome, "actor": row["actor"]}
    elif args.action == "record-rejections":
        rows = rejection_rows(
            name, context.get("reviews", []), known_event_ids(spark, args.catalog)
        )
        write_audit(spark, args.catalog, rows)
        result = {"recorded": [row["event_id"] for row in rows]}
    else:
        result = (selftest if args.action == "selftest" else status)(client, name)
    result = {"action": args.action, "model": name} | result | {"aliases": aliases(client, name)}
    print(f"{name}: {args.action} -> {result}")
    print(f"{RESULT_MARKER} {to_json(result)}")
