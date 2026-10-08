"""Model evaluation: ranking, calibration and alert-budget metrics (ml_model.md §10).

The alert budget turns "how good is the ranking?" into the question a ward cares about:
*if nurses can handle only so many alerts, how many deteriorations do those alerts catch?*
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class AlertBudget:
    """About 2 alerts per nurse per 12-hour shift, with one nurse per 5 patients."""

    alerts_per_nurse_per_shift: float = 2.0
    patients_per_nurse: float = 5.0
    shift_hours: float = 12.0

    @property
    def fraction(self) -> float:
        """Share of patient-hours that may raise an alert."""
        return self.alerts_per_nurse_per_shift / (self.patients_per_nurse * self.shift_hours)


def threshold_for_budget(scores: np.ndarray, budget: AlertBudget | float) -> float:
    """Score above which `budget` of patient-hours alert (fixed on validation, used on test)."""
    fraction = budget.fraction if isinstance(budget, AlertBudget) else budget
    return float(np.quantile(np.asarray(scores, dtype=float), 1 - fraction))


def alert_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    alerts = np.asarray(scores) >= threshold
    y = np.asarray(y)
    caught = (alerts & (y == 1)).sum()
    return {
        "alert_rate": float(alerts.mean()),
        "precision_at_budget": float(caught / alerts.sum()) if alerts.any() else float("nan"),
        "recall_at_budget": float(caught / (y == 1).sum()) if (y == 1).any() else float("nan"),
    }


def top_k_metrics(y: np.ndarray, scores: np.ndarray, fraction: float) -> dict[str, float]:
    """Precision and recall when exactly the top `fraction` of rows alert.

    Use this to compare two scores at the **same number of alerts**; thresholds fixed on
    validation can drift to different alert rates on new data.
    """
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    k = max(1, int(round(fraction * len(y))))
    caught = y[np.argsort(-scores, kind="stable")[:k]].sum()
    return {
        "precision_at_k": float(caught / k),
        "recall_at_k": float(caught / y.sum()) if y.sum() else float("nan"),
    }


def ranking_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "positive_rate": float(y.mean()),
    }


def calibration_table(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed rate per bin of predictions (equal-count bins)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(probs, dtype=float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    table = df.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return table.reset_index(drop=True)


def evaluate(
    y: np.ndarray, scores: np.ndarray, threshold: float, probabilities: bool = True
) -> dict[str, float]:
    """All headline metrics. Brier score only when `scores` are probabilities."""
    out = ranking_metrics(y, scores) | alert_metrics(y, scores, threshold)
    if probabilities:
        out["brier"] = float(brier_score_loss(np.asarray(y), np.asarray(scores, dtype=float)))
    return out


# --- Deep-dive evaluation (notebook 07) --------------------------------------------------------

AGE_BANDS = (18, 50, 70, 85, 200)
AGE_LABELS = ("18-49", "50-69", "70-84", "85+")


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, AGE_BANDS, right=False, labels=AGE_LABELS).astype(str)


BOOTSTRAP_METRICS = (
    "auroc",
    "auprc",
    "alert_rate",
    "precision_at_budget",
    "recall_at_budget",
    "precision_at_k",
    "recall_at_k",
)


def _group_positions(groups: np.ndarray) -> list[np.ndarray]:
    """Row positions of each group (e.g. each encounter), for the cluster bootstrap."""
    codes, _ = pd.factorize(np.asarray(groups))
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    return np.split(order, bounds)


def cluster_bootstrap(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    fraction: float | None = None,
) -> pd.DataFrame:
    """Bootstrap replicates of the headline metrics, resampling whole **groups** (encounters).

    Hours of the same patient are strongly correlated, so resampling single rows would
    understate the uncertainty. Every model is scored on the same resample in each replicate,
    so differences between models can be read off paired. Thresholds stay fixed (from
    validation). `*_at_k` metrics compare models at the same number of alerts (`fraction`,
    default the alert budget). Returns one row per (replicate, model, metric).
    """
    fraction = AlertBudget().fraction if fraction is None else fraction
    y = np.asarray(y)
    positions = _group_positions(groups)
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(n_boot):
        idx = np.concatenate(
            [positions[i] for i in rng.integers(0, len(positions), len(positions))]
        )
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for name, s in scores.items():
            sb = np.asarray(s)[idx]
            metrics = (
                ranking_metrics(yb, sb)
                | alert_metrics(yb, sb, thresholds[name])
                | top_k_metrics(yb, sb, fraction)
            )
            for metric in BOOTSTRAP_METRICS:
                rows.append((rep, name, metric, metrics[metric]))
    return pd.DataFrame(rows, columns=["rep", "model", "metric", "value"])


def confidence_intervals(replicates: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """Percentile intervals per model and metric."""
    tail = (1 - level) / 2
    g = replicates.groupby(["model", "metric"])["value"]
    return pd.DataFrame({"low": g.quantile(tail), "high": g.quantile(1 - tail)}).unstack("metric")


def paired_difference(
    replicates: pd.DataFrame, model: str, reference: str, level: float = 0.95
) -> pd.DataFrame:
    """CI of (model - reference) per metric, and how often the model wins across replicates."""
    wide = replicates.pivot_table(index=["rep", "metric"], columns="model", values="value")
    diff = (wide[model] - wide[reference]).groupby(level="metric")
    tail = (1 - level) / 2
    return pd.DataFrame(
        {
            "mean_diff": diff.mean(),
            "low": diff.quantile(tail),
            "high": diff.quantile(1 - tail),
            "share_model_better": diff.apply(lambda d: (d > 0).mean()),
        }
    )


def slice_metrics(
    y: np.ndarray, scores: np.ndarray, threshold: float, by: pd.Series
) -> pd.DataFrame:
    """Metrics per slice (site, age band, sex...) with one global threshold.

    The threshold is not re-tuned per slice: a ward uses one rule for everyone, so differing
    alert rates between slices are part of what this table should show. Ranking metrics are
    NaN for slices without both classes.
    """
    df = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(scores, dtype=float)})
    df["slice"] = np.asarray(by)
    rows = {}
    for key, part in df.groupby("slice"):
        both = part["y"].nunique() == 2
        rows[key] = {
            "patient_hours": len(part),
            "positives": int(part["y"].sum()),
            **(
                ranking_metrics(part["y"], part["s"])
                if both
                else {"auroc": np.nan, "auprc": np.nan, "positive_rate": part["y"].mean()}
            ),
            **alert_metrics(part["y"], part["s"], threshold),
        }
    return pd.DataFrame(rows).T


def warning_times(
    df: pd.DataFrame, scores: np.ndarray, threshold: float, step: pd.Timedelta | None = None
) -> pd.DataFrame:
    """For each escalation, was it caught, and how many hours of warning did the first alert give?

    An escalation shows up as a run of consecutive positive patient-hours in one encounter (the
    hours whose 6-hour look-ahead window contains the event). The event happened during the hour
    after the run's last row, so an alert at row time `t` gave between `lead_hours - 1` and
    `lead_hours` hours of warning, with `lead_hours = (last - t) / step + 1`.
    """
    step = step or pd.Timedelta(hours=1)
    d = df[["encounter_id", "prediction_ts"]].copy()
    d["y"], d["alert"] = df["label"].to_numpy(), np.asarray(scores) >= threshold
    d = d[d["y"] == 1].sort_values(["encounter_id", "prediction_ts"])
    new_run = (d["encounter_id"] != d["encounter_id"].shift()) | (d["prediction_ts"].diff() != step)
    d["episode"] = new_run.cumsum()
    out = []
    for _, run in d.groupby("episode"):
        last = run["prediction_ts"].max()
        alerted = run.loc[run["alert"], "prediction_ts"]
        out.append(
            {
                "encounter_id": run["encounter_id"].iloc[0],
                "last_hour_before_event": last,
                "window_hours": len(run),
                "caught": not alerted.empty,
                "lead_hours": (last - alerted.min()) / step + 1 if not alerted.empty else np.nan,
            }
        )
    return pd.DataFrame(out)
