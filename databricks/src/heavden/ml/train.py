"""Training: time split, candidate models, calibration and champion selection (Plan.md §8).

Rules learned in Phase 0 (Plan.md §8, NEWS2 fairness check):
* split by **time**, with a 6-hour gap between blocks so look-ahead labels can't straddle them;
* keep class weighting **modest**, and stop LightGBM on **average precision**, not log-loss;
* pick the champion on **validation**, whichever candidate wins, simple or not;
* a candidate must beat NEWS2 on validation AUPRC **and** precision at the alert budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from heavden.ml import evaluate, features

GAP = pd.Timedelta(hours=6)
CATEGORIES = {"unit_type": ("general", "step_down", "respiratory"), "sex": ("F", "M")}


@dataclass(frozen=True)
class Split:
    train_days: float = 9
    valid_days: float = 2
    gap: pd.Timedelta = GAP


def time_split(table: pd.DataFrame, split: Split | None = None):
    """Labelled rows split into consecutive train / validation / test blocks by prediction time.

    Rows within `gap` before each boundary are dropped, so no label window crosses a boundary.
    """
    split = split or Split()
    labelled = table[table["label"].notna()]
    start = labelled["prediction_ts"].min().floor("D")
    t1 = start + pd.Timedelta(days=split.train_days)
    t2 = t1 + pd.Timedelta(days=split.valid_days)
    ts = labelled["prediction_ts"]
    train = labelled[ts < t1 - split.gap]
    valid = labelled[(ts >= t1) & (ts < t2 - split.gap)]
    test = labelled[ts >= t2]
    return train, valid, test


def model_columns(table: pd.DataFrame) -> list[str]:
    """Numeric model inputs: feature columns with categoricals one-hot encoded."""
    base = [c for c in features.feature_columns(table) if c not in CATEGORIES]
    dummies = [f"{c}={v}" for c, values in CATEGORIES.items() for v in values]
    return base + dummies


def to_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Float matrix in a fixed column order (missing values stay NaN)."""
    out = pd.DataFrame(index=df.index)
    for c in columns:
        if "=" in c:
            name, value = c.split("=", 1)
            out[c] = (df[name] == value).astype(float)
        else:
            out[c] = df[c].astype(float)
    return out


class CalibratedModel(BaseEstimator, ClassifierMixin):
    """A fitted model plus an isotonic map from its raw scores to honest probabilities."""

    def __init__(self, base=None, calibrator: IsotonicRegression | None = None, columns=None):
        self.base = base
        self.calibrator = calibrator
        self.columns = columns

    @property
    def classes_(self) -> np.ndarray:
        return np.array([0, 1])

    def raw_scores(self, df: pd.DataFrame) -> np.ndarray:
        return self.base.predict_proba(to_matrix(df, self.columns))[:, 1]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.raw_scores(df)
        # Isotonic output is a step function with many ties; a negligible share of the raw score
        # keeps the original ranking, so alert thresholds can hit the budget precisely.
        p = np.clip(self.calibrator.predict(raw) + 1e-6 * raw, 0, 1)
        return np.column_stack([1 - p, p])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df)[:, 1] >= 0.5).astype(int)


def fit_logistic(train: pd.DataFrame, columns: list[str], c: float = 0.1):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=c, max_iter=5000),
    )
    return model.fit(to_matrix(train, columns), train["label"].astype(int))


def fit_lightgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    columns: list[str],
    pos_weight: float = 1.0,
    seed: int = 0,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        random_state=seed,
        verbose=-1,
    )
    return model.fit(
        to_matrix(train, columns),
        train["label"].astype(int),
        eval_X=to_matrix(valid, columns),
        eval_y=valid["label"].astype(int),
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def calibrate(base, valid: pd.DataFrame, columns: list[str]) -> CalibratedModel:
    """Isotonic calibration fitted on validation predictions.

    The extreme steps of an isotonic fit rest on a handful of points (e.g. the 5 highest-scored
    hours all escalated), which would make the model claim 0% or 100%. Two anchor points, a
    negative at the highest raw score and a positive at the lowest, keep both ends honest
    (4 of 5 rather than 5 of 5) without changing the middle of the curve or the ranking.
    """
    raw = base.predict_proba(to_matrix(valid, columns))[:, 1]
    y = valid["label"].to_numpy(float)
    raw_anchored = np.concatenate([raw, [raw.max(), raw.min()]])
    y_anchored = np.concatenate([y, [0.0, 1.0]])
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw_anchored, y_anchored)
    return CalibratedModel(base, iso, columns)


@dataclass
class CandidateResult:
    name: str
    model: CalibratedModel
    valid_metrics: dict[str, float]
    params: dict = field(default_factory=dict)


def news2_scores(df: pd.DataFrame) -> np.ndarray:
    """NEWS2 as a ranking score; ties broken by lowest SpO2 so it can be thresholded finely."""
    total = df["news2_total"].fillna(0).to_numpy(float)
    return total - 1e-3 * df["spo2_min_1h"].fillna(100).to_numpy(float)


def train_candidates(
    train: pd.DataFrame, valid: pd.DataFrame, budget: evaluate.AlertBudget | None = None
) -> tuple[list[CandidateResult], dict[str, float]]:
    """Fit and calibrate the candidates; score them and NEWS2 on validation."""
    budget = budget or evaluate.AlertBudget()
    columns = model_columns(train)
    y = valid["label"].to_numpy(int)

    baseline_scores = news2_scores(valid)
    baseline = evaluate.evaluate(
        y, baseline_scores, evaluate.threshold_for_budget(baseline_scores, budget), False
    )

    specs = {
        "logistic_regression": (fit_logistic(train, columns), {"C": 0.1}),
        "lightgbm": (fit_lightgbm(train, valid, columns), {"pos_weight": 1.0}),
    }
    results = []
    for name, (base, params) in specs.items():
        model = calibrate(base, valid, columns)
        p = model.predict_proba(valid)[:, 1]
        metrics = evaluate.evaluate(y, p, evaluate.threshold_for_budget(p, budget))
        if name == "lightgbm":
            params = params | {"best_iteration": int(base.best_iteration_ or 0)}
        results.append(CandidateResult(name, model, metrics, params))
    return results, baseline


def passes_gates(metrics: dict[str, float], baseline: dict[str, float]) -> bool:
    """A model must beat NEWS2 on validation AUPRC and on precision at the alert budget."""
    return (
        metrics["auprc"] > baseline["auprc"]
        and metrics["precision_at_budget"] > baseline["precision_at_budget"]
    )


def select_champion(
    results: list[CandidateResult], baseline: dict[str, float]
) -> CandidateResult | None:
    """The candidate with the best validation AUPRC among those that pass the gates."""
    eligible = [r for r in results if passes_gates(r.valid_metrics, baseline)]
    return max(eligible, key=lambda r: r.valid_metrics["auprc"]) if eligible else None
