"""Per-patient explanations: SHAP contributions and plain-language top factors.

Both candidate models have **exact** SHAP values without the `shap` package:

* logistic regression is linear in its standardised inputs, so each feature's contribution is
  `coef * (z - mean z over a background sample)` (interventional linear SHAP);
* LightGBM computes TreeSHAP itself (`predict(..., pred_contrib=True)`).

Contributions are in **log-odds of the raw model**, before isotonic calibration. Calibration is
monotone, so it never changes which factors pushed a patient up or down. For each row,
`base_value + contributions.sum()` equals the raw model's log-odds.

Correlated inputs (e.g. the 1 h, 3 h and 6 h breathing averages) can split credit oddly between
them, so explanations are also summed into **groups** a nurse would recognise ("breathing",
"oxygen saturation", ...). `top_factors` is the payload behind the assistant's
`explain_patient_risk` tool (Plan.md §11).
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from heavden.ml.train import CalibratedModel, to_matrix

# feature prefix -> group, checked in order (first match wins)
GROUPS = (
    ("news2_total", "NEWS2 total"),
    ("resp_rate", "breathing rate"),
    ("news2_resp_rate", "breathing rate"),
    ("spo2", "oxygen saturation"),
    ("news2_spo2", "oxygen saturation"),
    ("heart_rate", "heart rate"),
    ("news2_heart_rate", "heart rate"),
    ("temp_c", "temperature"),
    ("news2_temp", "temperature"),
    ("sbp", "blood pressure"),
    ("dbp", "blood pressure"),
    ("news2_sbp", "blood pressure"),
    ("new_confusion", "alertness"),
    ("news2_consciousness", "alertness"),
    ("on_oxygen", "oxygen therapy"),
    ("o2_flow_lpm", "oxygen therapy"),
    ("news2_oxygen", "oxygen therapy"),
    ("readings_1h", "monitoring gaps"),
    ("missing_", "monitoring gaps"),
    ("hours_since_obs", "monitoring gaps"),
    ("hours_since_admission", "time since admission"),
    ("unit_type=", "ward type"),
)
BACKGROUND = "patient background"  # age, sex, long-term conditions, medications

_VITALS = {
    "resp_rate": "breathing rate",
    "spo2": "SpO2",
    "heart_rate": "heart rate",
    "temp_c": "temperature",
    "sbp": "systolic BP",
    "dbp": "diastolic BP",
}
_STATS = {
    "mean_1h": "average, last hour",
    "median_1h": "median, last hour",
    "min_1h": "lowest, last hour",
    "max_1h": "highest, last hour",
    "mean_3h": "average, last 3 h",
    "mean_6h": "average, last 6 h",
    "trend_3h": "change over 3 h",
}


def feature_group(column: str) -> str:
    for prefix, group in GROUPS:
        if column.startswith(prefix):
            return group
    return BACKGROUND


def describe_feature(column: str) -> str:
    """Readable name for a model column, e.g. `resp_rate_trend_3h` -> 'breathing rate (change
    over 3 h)'."""
    for vital, name in _VITALS.items():
        if column.startswith(vital + "_") and column[len(vital) + 1 :] in _STATS:
            return f"{name} ({_STATS[column[len(vital) + 1 :]]})"
    if column.startswith("news2_"):
        return "NEWS2 " + column.removeprefix("news2_").replace("_", " ") + " points"
    return column.replace("=", ": ").replace("_", " ")


@dataclass
class Explanation:
    contributions: pd.DataFrame  # rows x model columns, log-odds units
    base_value: pd.Series  # per row; base + contributions.sum(axis=1) = raw log-odds

    def by_group(self) -> pd.DataFrame:
        groups = [feature_group(c) for c in self.contributions.columns]
        return self.contributions.T.groupby(groups).sum().T


def explain(
    model: CalibratedModel, df: pd.DataFrame, background: pd.DataFrame | None = None
) -> Explanation:
    """Exact SHAP contributions of `model` for each row of `df`.

    `background` (logistic regression only) is the reference population the contributions are
    measured against; it defaults to `df` itself. Use the training data for stable explanations.
    """
    x = to_matrix(df, model.columns)
    base = model.base
    if isinstance(base, lgb.LGBMClassifier):
        raw = base.predict(x, pred_contrib=True)
        contrib = pd.DataFrame(raw[:, :-1], index=df.index, columns=model.columns)
        return Explanation(contrib, pd.Series(raw[:, -1], index=df.index))

    transform, linear = base[:-1], base[-1]
    z = transform.transform(x)
    z_ref = transform.transform(
        to_matrix(background if background is not None else df, model.columns)
    )
    centre = z_ref.mean(axis=0)
    coef = linear.coef_.ravel()
    contrib = pd.DataFrame((z - centre) * coef, index=df.index, columns=model.columns)
    base_value = float(linear.intercept_[0] + centre @ coef)
    return Explanation(contrib, pd.Series(base_value, index=df.index))


def top_factors(
    explanation: Explanation, df: pd.DataFrame, row, k: int = 3
) -> list[dict[str, object]]:
    """The `k` groups that moved this patient-hour's risk most, each with its strongest feature.

    `row` is an index label of `df`. Each factor says whether it raised or lowered the risk, by
    how much (log-odds), and the value of the feature that drove it.
    """
    contrib = explanation.contributions.loc[row]
    groups = explanation.by_group().loc[row]
    factors = []
    for group in groups.abs().sort_values(ascending=False).index[:k]:
        members = contrib[[feature_group(c) == group for c in contrib.index]]
        strongest = members.abs().idxmax()
        value = to_matrix(df.loc[[row]], [strongest]).iloc[0, 0]
        factors.append(
            {
                "factor": group,
                "direction": "raises risk" if groups[group] > 0 else "lowers risk",
                "log_odds": round(float(groups[group]), 3),
                "driver": describe_feature(strongest),
                "driver_value": None if pd.isna(value) else round(float(value), 2),
            }
        )
    return factors


def raw_log_odds(model: CalibratedModel, df: pd.DataFrame) -> np.ndarray:
    """Log-odds of the uncalibrated model (what the contributions add up to)."""
    p = np.clip(model.raw_scores(df), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))
