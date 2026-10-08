"""Batch scoring, risk bands and what-if scoring (Plan.md §9; Model Card §3).

* `RiskBands`: cut-offs fixed on validation. **High** = the alert threshold: entering High
  raises an alert, and the threshold keeps alerts within the budget of about 2 per nurse per
  shift. **Medium** = up to the top 10% of patient-hours. **Low** = the rest.
* `score_table`: the gold `risk_scores` rows: risk, band and the top grouped SHAP factors.
* `what_if`: re-score one patient-hour with some vitals or nurse observations changed, for the
  app's what-if sliders. Only the latest hour changes; longer windows and trends shift
  accordingly, and NEWS2 is recomputed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from heavden.ml import evaluate, explain, news2
from heavden.ml.train import CalibratedModel, to_matrix

BANDS = ("Low", "Medium", "High")
MEDIUM_FRACTION = 0.10  # Medium = patient-hours in the top 10% (High included)


@dataclass(frozen=True)
class RiskBands:
    medium: float  # risk at or above this is at least Medium
    high: float  # risk at or above this is High (= alert)

    @classmethod
    def fit(
        cls,
        valid: pd.DataFrame,
        valid_scores: np.ndarray,
        budget: evaluate.AlertBudget | None = None,
        medium_fraction: float = MEDIUM_FRACTION,
    ) -> RiskBands:
        """`valid` holds the validation rows (`encounter_id`, `prediction_ts`, `label`)."""
        high = evaluate.threshold_for_alert_budget(valid, valid_scores, budget)
        medium = evaluate.threshold_for_fraction(valid_scores, medium_fraction)
        return cls(medium=min(medium, high), high=high)

    def band(self, risk) -> np.ndarray:
        risk = np.asarray(risk, dtype=float)
        return np.select([risk >= self.high, risk >= self.medium], ["High", "Medium"], "Low")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def top_factors_table(
    explanation: explain.Explanation, df: pd.DataFrame, k: int = 3
) -> list[list[dict]]:
    """`explain.top_factors` for every row at once (vectorised over groups)."""
    contrib = explanation.contributions
    columns = np.array(contrib.columns)
    groups = np.array([explain.feature_group(c) for c in columns])
    names = sorted(set(groups))
    values = to_matrix(df, list(columns)).to_numpy(float)
    c = contrib.to_numpy(float)

    group_sum = np.zeros((len(df), len(names)))
    driver = np.zeros((len(df), len(names)), dtype=int)  # column index of each group's driver
    for j, name in enumerate(names):
        idx = np.flatnonzero(groups == name)
        part = c[:, idx]
        group_sum[:, j] = part.sum(axis=1)
        driver[:, j] = idx[np.argmax(np.abs(part), axis=1)]

    order = np.argsort(-np.abs(group_sum), axis=1, kind="stable")[:, :k]
    rows = np.arange(len(df))
    out = []
    for i in rows:
        factors = []
        for j in order[i]:
            col = driver[i, j]
            value = values[i, col]
            factors.append(
                {
                    "factor": names[j],
                    "direction": "raises risk" if group_sum[i, j] > 0 else "lowers risk",
                    "log_odds": round(float(group_sum[i, j]), 3),
                    "driver": explain.describe_feature(columns[col]),
                    "driver_value": None if np.isnan(value) else round(float(value), 2),
                }
            )
        out.append(factors)
    return out


def score_table(
    model: CalibratedModel,
    table: pd.DataFrame,
    bands: RiskBands,
    background: pd.DataFrame,
    model_version: str,
    k: int = 3,
) -> pd.DataFrame:
    """Gold `risk_scores`: one row per patient-hour with risk, band and top factors (as JSON)."""
    risk = model.predict_proba(table)[:, 1]
    expl = explain.explain(model, table, background=background)
    keys = ["encounter_id", "patient_id", "prediction_ts", "site_id", "unit_id"]
    out = table[keys].copy()
    out["risk"] = risk
    out["risk_band"] = bands.band(risk)
    out["news2_total"] = table["news2_total"]
    out["top_factors"] = [json.dumps(f) for f in top_factors_table(expl, table, k)]
    out["model_version"] = model_version
    return out.reset_index(drop=True)


# --- What-if ---------------------------------------------------------------------------------

WHAT_IF_VITALS = ("heart_rate", "resp_rate", "spo2", "temp_c", "sbp", "dbp")
ACVPU = ("A", "C", "V", "P", "U")


def apply_changes(row: pd.Series, changes: dict) -> pd.Series:
    """One patient-hour with the latest hour's vitals (and/or nurse observations) changed.

    A new vital value replaces the last hour's mean, median, min and max. The 3 h and 6 h
    averages move by the change divided by 3 and 6 (the last hour is one of 3 or 6 hours), and
    the 3 h trend moves by the full change. NEWS2 is recomputed from the new values.
    """
    unknown = set(changes) - {*WHAT_IF_VITALS, "on_oxygen", "acvpu"}
    if unknown:
        raise ValueError(f"cannot change {sorted(unknown)}")
    new = row.copy()
    for vital in WHAT_IF_VITALS:
        if changes.get(vital) is None:
            continue
        value = float(changes[vital])
        old = new[f"{vital}_mean_1h"]
        delta = 0.0 if pd.isna(old) else value - old
        for stat in ("mean_1h", "median_1h", "min_1h", "max_1h"):
            new[f"{vital}_{stat}"] = value
        for stat, hours in (("mean_3h", 3), ("mean_6h", 6)):
            current = new[f"{vital}_{stat}"]
            new[f"{vital}_{stat}"] = value if pd.isna(current) else current + delta / hours
        if not pd.isna(new[f"{vital}_trend_3h"]):
            new[f"{vital}_trend_3h"] += delta
    if changes.get("on_oxygen") is not None:
        new["on_oxygen"] = bool(changes["on_oxygen"])
        if not new["on_oxygen"]:
            new["o2_flow_lpm"] = 0.0
        elif not new.get("o2_flow_lpm"):
            new["o2_flow_lpm"] = 2.0  # the first titration step
    if changes.get("acvpu") is not None:
        if changes["acvpu"] not in ACVPU:
            raise ValueError(f"acvpu must be one of {ACVPU}")
        new["acvpu"] = changes["acvpu"]
        new["new_confusion"] = changes["acvpu"] != "A"

    points = news2.news2(new.to_frame().T, suffix="_median_1h")
    for column in points.columns:
        new[column] = points[column].iloc[0]
    return new


def what_if(
    model: CalibratedModel,
    row: pd.Series,
    changes: dict,
    bands: RiskBands,
    background: pd.DataFrame,
    k: int = 3,
) -> dict:
    """Risk before and after `changes`, with the new band, NEWS2 and top factors."""
    before = row.to_frame().T.infer_objects()
    after = apply_changes(row, changes).to_frame().T.infer_objects()
    p_before, p_after = model.predict_proba(pd.concat([before, after]))[:, 1]
    expl = explain.explain(model, after, background=background)
    return {
        "risk_before": float(p_before),
        "risk_after": float(p_after),
        "band_before": str(bands.band(p_before)),
        "band_after": str(bands.band(p_after)),
        "news2_before": _int_or_none(before["news2_total"].iloc[0]),
        "news2_after": _int_or_none(after["news2_total"].iloc[0]),
        "top_factors": top_factors_table(expl, after, k)[0],
    }


def _int_or_none(value) -> int | None:
    return None if pd.isna(value) else int(value)
