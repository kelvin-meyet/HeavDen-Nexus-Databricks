"""Train, evaluate and register the deterioration-risk model (Plan.md §8).

The steps of notebook 06 as one function, so the Databricks job (`databricks/src/jobs/
train_model.py`) and the tests run the same code:

1. split the labelled patient-hours by time (train / validation / test, with 6 h gaps);
2. fit the candidates (logistic regression, LightGBM), calibrate them on validation, and score
   them and NEWS2 on validation at the same alert budget;
3. choose: the best validation AUPRC among candidates that pass the gates (beat NEWS2 on AUPRC
   and flag at least as many escalations); if none passes, the best one is still registered,
   without an alias, so the registry records every run;
4. score the chosen model and NEWS2 once on test;
5. log every candidate to MLflow and register the chosen one (`tracking.register_version`).

A **retrain** (`--mode retrain`, started by an approved retrain request) differs in two ways:
the split is counted back from the newest labelled hour (`train.RecentSplit`), and the chosen
model must also beat the current champion on the test block, at the same alert budget
(`compare_with_champion`), to become `@challenger`. Promotion to `@champion` stays a separate,
approved step.

On Databricks `main` reads `<catalog>.gold.training_set` with Spark, logs to the workspace
experiment and registers in Unity Catalog as `<catalog>.ml.deterioration_risk`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import pandas as pd

import heavden
from heavden.ml import evaluate, tracking, train

# What the logged model needs to load: our code travels inside the model (code_paths), these
# libraries are pinned to the versions it was trained with.
MODEL_LIBRARIES = ("cloudpickle", "lightgbm", "numpy", "pandas", "scikit-learn", "scipy")


RESULT_MARKER = "TRAINING_RESULT"
BETTER_SHARE = 0.75  # of paired resamples: flags at least as many escalations; higher AUROC
NOT_WORSE_SHARE = 0.10  # of paired resamples in which its AUPRC is the higher (a guard)
BRIER_SLACK = 1.10  # its Brier score may be at most 10% worse than the champion's


@dataclass
class Champion:
    """The model a retrain has to beat. Its deployed alert `threshold` isn't used in the
    comparison: both models get a threshold fitted on the same recent validation days."""

    model: object
    threshold: float
    version: str


@dataclass
class TrainingOutcome:
    chosen: train.CandidateResult
    passed_gates: bool  # every gate that applied: NEWS2, and the champion in a retrain
    news2_valid: dict[str, float]
    test: dict[str, float]
    news2_test: dict[str, float]
    version: int
    aliases: list[str]
    rows: dict[str, int]
    vs_champion: dict[str, float] | None = None


def compare_with_champion(
    test: pd.DataFrame,
    scores: dict[str, object],
    thresholds: dict[str, float],
    budget: evaluate.AlertBudget,
    n_boot: int = 300,
) -> dict[str, float]:
    """The new model ("challenger") against the champion on the same test block, both at the
    same alert budget (`thresholds` fitted on the same validation days).

    A recent test block holds only 20-30 escalations, so a single comparison proves little.
    Both models are scored on the same resamples of whole stays (a paired bootstrap). The new
    model must, in at least `BETTER_SHARE` of them, flag at least as many escalations and have
    the higher AUROC; its AUPRC must not be clearly worse (higher in at least
    `NOT_WORSE_SHARE`); and its Brier score at most `BRIER_SLACK` times the champion's.

    Why these and not "higher AUPRC": measured offline after the protocol change (3 seeds, 5 to
    17 days of new data), a retrain on all history flagged more escalations at the same alert
    budget and had the higher AUROC in nearly every run, while AUPRC moved either way. AUPRC is
    set by the top of the ranking, where both models place the classic deteriorations; what the
    retrain learns (the new protocol's reviews of mildly abnormal patients) sits lower down.
    Re-fitting only the champion's threshold gave none of the gain. On the same runs these
    gates passed 13 of 15 retrains on all history (3 of 3 at 7.5 days) and stopped all 9
    retrains on post-change data alone at 5.5, 7.5 and 13.5 days.

    The limits were chosen on that rehearsal, after seeing its results (`NOT_WORSE_SHARE` with
    one passing run at 0.13), so its pass rates describe the design, not a test of it. The
    test is prod's retrain, on an episode whose seed no rehearsal used."""
    point = {k: evaluate.evaluate(test, s, thresholds[k], True, budget) for k, s in scores.items()}
    replicates = evaluate.cluster_bootstrap(test, scores, thresholds, n_boot=n_boot, budget=budget)
    wide = replicates.pivot_table(index=["rep", "metric"], columns="model", values="value")

    def share(metric: str, strictly: bool) -> float:
        new, old = (
            wide.xs(metric, level="metric")["challenger"],
            wide.xs(metric, level="metric")["champion"],
        )
        return float((new > old).mean() if strictly else (new >= old).mean())

    new, old = point["challenger"], point["champion"]
    out = {
        "escalations": new["escalations"],
        "escalations_flagged": new["escalations_flagged"],
        "champion_escalations_flagged": old["escalations_flagged"],
        "flagged_at_least_share": share("escalations_flagged", strictly=False),
        "auroc": new["auroc"],
        "champion_auroc": old["auroc"],
        "auroc_better_share": share("auroc", strictly=True),
        "auprc": new["auprc"],
        "champion_auprc": old["auprc"],
        "auprc_better_share": share("auprc", strictly=True),
        "brier": new["brier"],
        "champion_brier": old["brier"],
        "alerts_per_nurse_shift": new["alerts_per_nurse_shift"],
        "champion_alerts_per_nurse_shift": old["alerts_per_nurse_shift"],
    }
    out["beats_champion"] = float(
        out["flagged_at_least_share"] >= BETTER_SHARE
        and out["auroc_better_share"] >= BETTER_SHARE
        and out["auprc_better_share"] >= NOT_WORSE_SHARE
        and new["brier"] <= old["brier"] * BRIER_SLACK
    )
    return out


EXCLUSIONS_PATH = Path(__file__).with_name("training_exclusions.yaml")


def load_exclusions(catalog: str, path: Path = EXCLUSIONS_PATH) -> list[dict]:
    """The catalog's entries in `training_exclusions.yaml`: data faults the model must not
    learn from."""
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")).get(catalog) or []


def drop_excluded(table: pd.DataFrame, exclusions: list[dict]) -> pd.DataFrame:
    """`table` without the patient-hours of known data faults (`site_id`, `from`, `to`)."""
    keep = pd.Series(True, index=table.index)
    for e in exclusions:
        hours = table["prediction_ts"].between(
            pd.Timestamp(e["from"]).tz_convert("UTC"), pd.Timestamp(e["to"]).tz_convert("UTC")
        )
        keep &= ~(hours & ((e["site_id"] == "all") | (table["site_id"] == e["site_id"])))
    return table[keep]


def block_windows(blocks: dict[str, pd.DataFrame]) -> dict[str, str]:
    """First and last hour of each block, as run tags: a model version carries its own data
    window, so scoring and the drift check can find its training rows later."""
    tags = {}
    for name, rows in blocks.items():
        tags[f"{name}_from"] = rows["prediction_ts"].min().isoformat()
        tags[f"{name}_to"] = rows["prediction_ts"].max().isoformat()
    return tags


def model_requirements() -> list[str]:
    return [f"{lib}=={version(lib)}" for lib in MODEL_LIBRARIES]


def run_training(
    table: pd.DataFrame,
    model_name: str,
    split: train.Split | train.RecentSplit | None = None,
    budget: evaluate.AlertBudget | None = None,
    tags: dict[str, str] | None = None,
    champion: Champion | None = None,
) -> TrainingOutcome:
    """Train, evaluate, log and register. The MLflow tracking and registry URIs and the
    experiment must be set by the caller. With a `champion`, the chosen model must beat it on
    the test block too (a retrain)."""
    budget = budget or evaluate.AlertBudget()
    train_df, valid, test = train.split_blocks(table, split)
    rows = {"train": len(train_df), "valid": len(valid), "test": len(test)}
    if min(rows.values()) == 0 or valid["label"].sum() == 0:
        raise ValueError(f"not enough labelled data for this split: {rows}")

    results, news2_valid = train.train_candidates(train_df, valid, budget)
    chosen = train.select_champion(results, news2_valid)
    passed = chosen is not None
    chosen = chosen or max(results, key=lambda r: r.valid_metrics["auprc"])

    # Test is scored once, for the chosen model, at its validation threshold (and NEWS2 at its own).
    test_scores = chosen.model.predict_proba(test)[:, 1]
    test_metrics = evaluate.evaluate(
        test, test_scores, chosen.valid_metrics["threshold"], True, budget
    )
    news2_test = evaluate.evaluate(
        test, train.news2_scores(test), news2_valid["threshold"], False, budget
    )

    vs_champion = None
    if champion is not None:
        vs_champion = compare_with_champion(
            test,
            {"challenger": test_scores, "champion": champion.model.predict_proba(test)[:, 1]},
            {
                "challenger": chosen.valid_metrics["threshold"],
                "champion": evaluate.threshold_for_alert_budget(
                    valid, champion.model.predict_proba(valid)[:, 1], budget
                ),
            },
            budget,
        )
        passed = passed and bool(vs_champion["beats_champion"])

    common_tags = (tags or {}) | {f"rows_{k}": str(v) for k, v in rows.items()}
    common_tags |= block_windows({"train": train_df, "valid": valid, "test": test})
    if champion is not None:
        common_tags["compared_with_version"] = champion.version
    code = [str(Path(heavden.__file__).parent)]
    requirements = model_requirements()
    logged = {}
    for r in results:
        metrics = {f"news2_valid_{k}": v for k, v in news2_valid.items()}
        if r is chosen:
            metrics |= {f"test_{k}": v for k, v in test_metrics.items()}
            metrics |= {f"news2_test_{k}": v for k, v in news2_test.items()}
            metrics |= {f"vs_champion_{k}": v for k, v in (vs_champion or {}).items()}
        run_tags = common_tags | {
            "chosen": str(r is chosen).lower(),
            "passed_gates": str(train.passes_gates(r.valid_metrics, news2_valid)).lower(),
        }
        logged[r.name] = tracking.log_candidate(
            r, valid, metrics, run_tags, code_paths=code, pip_requirements=requirements
        )

    description = (
        f"{chosen.name}; validation AUPRC {chosen.valid_metrics['auprc']:.3f} "
        f"(NEWS2 {news2_valid['auprc']:.3f}), escalations flagged "
        f"{chosen.valid_metrics['escalations_flagged']:.0%} (NEWS2 "
        f"{news2_valid['escalations_flagged']:.0%}); test AUPRC {test_metrics['auprc']:.3f} "
        f"(NEWS2 {news2_test['auprc']:.3f})."
    )
    if vs_champion is not None:
        description += (
            f" Against champion v{champion.version} on the same test block and alert budget: "
            f"escalations flagged {vs_champion['escalations_flagged']:.0%} vs "
            f"{vs_champion['champion_escalations_flagged']:.0%} (at least as many in "
            f"{vs_champion['flagged_at_least_share']:.0%} of resamples), AUROC "
            f"{vs_champion['auroc']:.3f} vs {vs_champion['champion_auroc']:.3f} (higher in "
            f"{vs_champion['auroc_better_share']:.0%}), AUPRC {vs_champion['auprc']:.3f} vs "
            f"{vs_champion['champion_auprc']:.3f}."
        )
    description += f" Passed gates: {passed}."
    version_number, aliases = tracking.register_version(
        logged[chosen.name], model_name, passed_gates=passed, description=description
    )
    return TrainingOutcome(
        chosen,
        passed,
        news2_valid,
        test_metrics,
        news2_test,
        version_number,
        aliases,
        rows,
        vs_champion,
    )


def result_line(model_name: str, outcome: TrainingOutcome) -> str:
    """The run on one line, as JSON after `RESULT_MARKER`: what the retrain workflow reads from
    the job's output to report the gates."""
    payload = {
        "model": model_name,
        "version": outcome.version,
        "model_type": outcome.chosen.name,
        "passed_gates": outcome.passed_gates,
        "aliases": outcome.aliases,
        "rows": outcome.rows,
        "valid_auprc": outcome.chosen.valid_metrics["auprc"],
        "news2_valid_auprc": outcome.news2_valid["auprc"],
        "test_auprc": outcome.test["auprc"],
        "news2_test_auprc": outcome.news2_test["auprc"],
        "vs_champion": outcome.vs_champion,
    }
    return f"{RESULT_MARKER} {json.dumps(payload)}"


def to_utc(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Spark hands timestamps to pandas without a time zone; the session is UTC."""
    for c in columns:
        df[c] = pd.to_datetime(df[c]).dt.tz_localize("UTC")
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True, help="e.g. heavden_prod")
    parser.add_argument("--experiment-id", required=True, help="MLflow experiment (bundle)")
    parser.add_argument("--train-days", type=float, default=train.Split.train_days)
    parser.add_argument("--valid-days", type=float, default=train.Split.valid_days)
    parser.add_argument("--mode", choices=("initial", "retrain"), default="initial")
    parser.add_argument(
        "--model-alias", default="champion", help="the model in use, which a retrain must beat"
    )
    parser.add_argument("--retrain-valid-days", type=float, default=train.RecentSplit.valid_days)
    parser.add_argument("--retrain-test-days", type=float, default=train.RecentSplit.test_days)
    args = parser.parse_args(argv)

    import mlflow
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    source = f"{args.catalog}.gold.training_set"
    table = to_utc(spark.read.table(source).toPandas(), ("prediction_ts", "label_known_at"))
    exclusions = load_exclusions(args.catalog)
    rows_read = len(table)
    table = drop_excluded(table, exclusions)

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment_id=args.experiment_id)
    model_name = f"{args.catalog}.ml.deterioration_risk"
    split: train.Split | train.RecentSplit = train.Split(args.train_days, args.valid_days)
    champion = None
    if args.mode == "retrain":
        from heavden.ml import batch_scoring  # imports this module

        split = train.RecentSplit(args.retrain_valid_days, args.retrain_test_days)
        champion = Champion(*batch_scoring.load_registered_model(model_name, args.model_alias))
    outcome = run_training(
        table,
        model_name,
        split,
        tags={
            "mode": args.mode,
            "excluded_rows": str(rows_read - len(table)),
            "exclusions": "; ".join(e["reason"] for e in exclusions) or "none",
            "source_table": source,
            "data_as_of": str(table["label_known_at"].max()),
            "heavden_version": version("heavden-nexus-databricks"),
        },
        champion=champion,
    )
    print(
        f"{model_name} v{outcome.version} = {outcome.chosen.name}, rows {outcome.rows}, "
        f"passed gates: {outcome.passed_gates}, aliases: {outcome.aliases or 'none'}"
    )
    print(
        f"validation AUPRC {outcome.chosen.valid_metrics['auprc']:.3f} "
        f"vs NEWS2 {outcome.news2_valid['auprc']:.3f}; "
        f"test AUPRC {outcome.test['auprc']:.3f} vs NEWS2 {outcome.news2_test['auprc']:.3f}"
    )
    print(result_line(model_name, outcome))
