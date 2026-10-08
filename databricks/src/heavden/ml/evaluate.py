"""Model evaluation: ranking, calibration and alert metrics (ml_model.md §10).

Two levels, answering different questions:

* **Per patient-hour (ranking):** AUROC, AUPRC, Brier score, calibration. *Does the model rank
  the hours before an escalation above the others?*
* **Per alert (what a nurse experiences):** an alert fires when a patient **enters** the High
  band, and the same patient doesn't alert again for 6 hours (alarm-fatigue suppression; the
  gold `alerts_fact` table uses the same rule). The alert threshold is set so alerts stay within
  the **alert budget** of about 2 per nurse per 12-hour shift. Then:
  - `alert_precision`: share of alerts followed by an escalation within 6 hours;
  - `escalations_flagged`: share of escalations where the patient was in the High band at some
    point in the 6 hours before;
  - `median_warning_hours`: how early the patient was first flagged.

Compare models **at the same alert load**: a model that flickers in and out of High raises more
alerts than one whose High periods are steady, even with the same share of High hours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

SUPPRESS = pd.Timedelta(hours=6)  # no repeat alert for the same stay within this time
HOUR = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class AlertBudget:
    """About 2 alerts per nurse per 12-hour shift, with one nurse per 5 patients."""

    alerts_per_nurse_per_shift: float = 2.0
    patients_per_nurse: float = 5.0
    shift_hours: float = 12.0

    @property
    def alerts_per_patient_hour(self) -> float:
        return self.alerts_per_nurse_per_shift / (self.patients_per_nurse * self.shift_hours)

    def per_nurse_shift(self, alerts: float, patient_hours: float) -> float:
        """Convert an alert count over some patient-hours into alerts per nurse per shift."""
        return alerts / patient_hours * self.patients_per_nurse * self.shift_hours


# --- Per patient-hour ------------------------------------------------------------------------


def threshold_for_fraction(scores: np.ndarray, fraction: float) -> float:
    """Score above which the top `fraction` of patient-hours lie."""
    return float(np.quantile(np.asarray(scores, dtype=float), 1 - fraction))


def top_k_metrics(y: np.ndarray, scores: np.ndarray, fraction: float) -> dict[str, float]:
    """Per-hour precision and recall when exactly the top `fraction` of patient-hours are High."""
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


# --- Per alert -------------------------------------------------------------------------------


def _naive_utc(series: pd.Series) -> np.ndarray:
    """datetime64[ns] values in UTC (tz-aware columns otherwise become object arrays)."""
    return pd.to_datetime(series, utc=True).dt.tz_localize(None).to_numpy()


def alert_onsets(
    df: pd.DataFrame, high: np.ndarray, suppress: pd.Timedelta = SUPPRESS
) -> np.ndarray:
    """Which rows raise an alert: the stay enters High (was not High in its previous hour) and
    hasn't alerted within `suppress`. Returned in `df`'s row order."""
    high = np.asarray(high, dtype=bool)
    order = np.lexsort((_naive_utc(df["prediction_ts"]), df["encounter_id"].to_numpy()))
    enc = df["encounter_id"].to_numpy()[order]
    ts = _naive_utc(df["prediction_ts"])[order]
    hi = high[order]
    same_stay = np.r_[False, enc[1:] == enc[:-1]]
    contiguous = same_stay & np.r_[False, (ts[1:] - ts[:-1]) == HOUR.to_timedelta64()]
    entering = hi & ~(contiguous & np.r_[False, hi[:-1]])

    onset = np.zeros(len(df), dtype=bool)
    last: dict = {}
    window = suppress.to_timedelta64()
    for i in np.flatnonzero(entering):
        if enc[i] not in last or ts[i] - last[enc[i]] >= window:
            onset[i] = True
            last[enc[i]] = ts[i]
    out = np.zeros(len(df), dtype=bool)
    out[order] = onset
    return out


def warning_times(
    df: pd.DataFrame, scores: np.ndarray, threshold: float, step: pd.Timedelta = HOUR
) -> pd.DataFrame:
    """For each escalation: was the patient flagged (High) in the 6 hours before, and how early?

    An escalation shows up as a run of consecutive positive patient-hours in one encounter (the
    hours whose 6-hour look-ahead window contains the event). The event happened during the hour
    after the run's last row, so a High row at time `t` gave between `lead_hours - 1` and
    `lead_hours` hours of warning, with `lead_hours = (last - t) / step + 1`.
    """
    d = df[["encounter_id", "prediction_ts"]].copy()
    d["y"], d["high"] = df["label"].to_numpy(), np.asarray(scores) >= threshold
    d = d[d["y"] == 1].sort_values(["encounter_id", "prediction_ts"])
    new_run = (d["encounter_id"] != d["encounter_id"].shift()) | (d["prediction_ts"].diff() != step)
    d["episode"] = new_run.cumsum()
    out = []
    for _, run in d.groupby("episode"):
        last = run["prediction_ts"].max()
        flagged = run.loc[run["high"], "prediction_ts"]
        out.append(
            {
                "encounter_id": run["encounter_id"].iloc[0],
                "last_hour_before_event": last,
                "last_row": run.index[-1],
                "window_hours": len(run),
                "caught": not flagged.empty,
                "lead_hours": (last - flagged.min()) / step + 1 if not flagged.empty else np.nan,
            }
        )
    columns = ["encounter_id", "last_hour_before_event", "last_row", "window_hours", "caught"]
    return pd.DataFrame(out, columns=[*columns, "lead_hours"])


def alert_metrics(
    df: pd.DataFrame, scores: np.ndarray, threshold: float, budget: AlertBudget | None = None
) -> dict[str, float]:
    """Alert load, alert precision, escalations flagged and warning time for one threshold.

    `df` needs `encounter_id`, `prediction_ts` and `label` (rows with a known label).
    """
    budget = budget or AlertBudget()
    high = np.asarray(scores) >= threshold
    onset = alert_onsets(df, high)
    y = df["label"].to_numpy()
    w = warning_times(df, scores, threshold)
    return {
        "alerts": int(onset.sum()),
        "alerts_per_nurse_shift": budget.per_nurse_shift(onset.sum(), len(df)),
        "alert_precision": float(y[onset].mean()) if onset.any() else float("nan"),
        "escalations": len(w),
        "escalations_flagged": float(w["caught"].mean()) if len(w) else float("nan"),
        "median_warning_hours": float(w["lead_hours"].median()) if len(w) else float("nan"),
        "high_hour_share": float(high.mean()),
    }


def threshold_for_alert_budget(
    df: pd.DataFrame,
    scores: np.ndarray,
    budget: AlertBudget | None = None,
    grid: int = 300,
) -> float:
    """The lowest threshold whose alerts stay within the budget (fixed on validation).

    Lowering the threshold raises more alerts until almost everyone is High (then entries
    become rare again), so candidates are scanned from the top down and the scan stops at the
    first one over budget.
    """
    budget = budget or AlertBudget()
    scores = np.asarray(scores, dtype=float)
    allowed = budget.alerts_per_patient_hour * len(df)
    candidates = np.unique(np.quantile(scores, np.linspace(0.5, 0.9999, grid)))[::-1]
    chosen = candidates[0]
    for threshold in candidates:
        if alert_onsets(df, scores >= threshold).sum() > allowed:
            break
        chosen = threshold
    return float(chosen)


def evaluate(
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    probabilities: bool = True,
    budget: AlertBudget | None = None,
) -> dict[str, float]:
    """All headline metrics: per-hour ranking, per-alert metrics, and Brier score when `scores`
    are probabilities."""
    y = df["label"].to_numpy()
    out = ranking_metrics(y, scores) | alert_metrics(df, scores, threshold, budget)
    if probabilities:
        out["brier"] = float(brier_score_loss(y, np.asarray(scores, dtype=float)))
    return out


# --- Deep-dive evaluation (notebook 07) --------------------------------------------------------

AGE_BANDS = (18, 50, 70, 85, 200)
AGE_LABELS = ("18-49", "50-69", "70-84", "85+")
BOOTSTRAP_METRICS = (
    "auroc",
    "auprc",
    "alerts_per_nurse_shift",
    "alert_precision",
    "escalations_flagged",
)


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, AGE_BANDS, right=False, labels=AGE_LABELS).astype(str)


def _per_encounter_alert_counts(
    df: pd.DataFrame, scores: np.ndarray, threshold: float, codes: np.ndarray, n: int
) -> dict[str, np.ndarray]:
    """Alert and escalation counts per encounter (alerts never span encounters, so a
    bootstrap over encounters can just add these up)."""
    onset = alert_onsets(df, np.asarray(scores) >= threshold)
    y = df["label"].to_numpy()
    w = warning_times(df, scores, threshold)
    enc_code = dict(zip(df["encounter_id"].to_numpy(), codes, strict=True))
    w_codes = w["encounter_id"].map(enc_code).to_numpy(int)
    return {
        "alerts": np.bincount(codes, weights=onset, minlength=n),
        "true_alerts": np.bincount(codes, weights=onset & (y == 1), minlength=n),
        "episodes": np.bincount(w_codes, minlength=n),
        "flagged": np.bincount(w_codes, weights=w["caught"].to_numpy(float), minlength=n),
    }


def cluster_bootstrap(
    df: pd.DataFrame,
    scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    n_boot: int = 1000,
    seed: int = 0,
    budget: AlertBudget | None = None,
) -> pd.DataFrame:
    """Bootstrap replicates of the headline metrics, resampling whole **encounters**.

    Hours of the same patient are strongly correlated, so resampling single rows would
    understate the uncertainty. Every model is scored on the same resample in each replicate,
    so differences between models can be read off paired. Thresholds stay fixed. Returns one
    row per (replicate, model, metric).
    """
    budget = budget or AlertBudget()
    y = df["label"].to_numpy()
    codes, uniques = pd.factorize(df["encounter_id"])
    n = len(uniques)
    order = np.argsort(codes, kind="stable")
    positions = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
    hours = np.bincount(codes, minlength=n)
    counts = {
        name: _per_encounter_alert_counts(df, s, thresholds[name], codes, n)
        for name, s in scores.items()
    }
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(n_boot):
        pick = rng.integers(0, n, n)
        weight = np.bincount(pick, minlength=n)
        idx = np.concatenate([positions[i] for i in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for name, s in scores.items():
            c = {k: (v * weight).sum() for k, v in counts[name].items()}
            metrics = ranking_metrics(yb, np.asarray(s)[idx]) | {
                "alerts_per_nurse_shift": budget.per_nurse_shift(
                    c["alerts"], (hours * weight).sum()
                ),
                "alert_precision": c["true_alerts"] / c["alerts"] if c["alerts"] else np.nan,
                "escalations_flagged": c["flagged"] / c["episodes"] if c["episodes"] else np.nan,
            }
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
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    by: pd.Series,
    budget: AlertBudget | None = None,
) -> pd.DataFrame:
    """Metrics per slice (site, age band, sex...) with one shared threshold.

    The threshold is not re-tuned per slice: a ward uses one rule for everyone, so differing
    alert loads between slices are part of what this table should show. An escalation belongs
    to the slice of its last hour before the event. Ranking metrics are NaN for slices without
    both classes.
    """
    budget = budget or AlertBudget()
    scores = np.asarray(scores, dtype=float)
    y = df["label"].to_numpy()
    onset = alert_onsets(df, scores >= threshold)
    by = pd.Series(np.asarray(by), index=df.index)
    w = warning_times(df, scores, threshold)
    w["slice"] = by.loc[w["last_row"]].to_numpy() if len(w) else []
    rows = {}
    for key in sorted(by.unique()):
        part = (by == key).to_numpy()
        ep = w[w["slice"] == key]
        both = len(np.unique(y[part])) == 2
        rows[key] = {
            "patient_hours": int(part.sum()),
            "positives": int(y[part].sum()),
            **(
                ranking_metrics(y[part], scores[part])
                if both
                else {"auroc": np.nan, "auprc": np.nan, "positive_rate": float(y[part].mean())}
            ),
            "alerts_per_nurse_shift": budget.per_nurse_shift(onset[part].sum(), part.sum()),
            "alert_precision": float(y[part & onset].mean()) if (part & onset).any() else np.nan,
            "escalations": len(ep),
            "escalations_flagged": float(ep["caught"].mean()) if len(ep) else np.nan,
        }
    return pd.DataFrame(rows).T
