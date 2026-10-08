"""NEWS2 (National Early Warning Score 2), the clinical baseline the model must beat.

Royal College of Physicians, 2017. Seven parameters each earn points; the total triggers review:
breathing rate, SpO2, supplemental oxygen (+2), systolic BP, heart rate, consciousness (ACVPU:
anything other than Alert scores 3) and temperature. Patients with chronic lung disease (COPD)
use **SpO2 scale 2**, whose target range is 88-92% (and which penalises high SpO2 on oxygen).
Consciousness and oxygen come from nurse observations; if they are absent a patient is scored
as alert and on room air.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS = ("resp_rate", "spo2", "sbp", "heart_rate", "temp_c")


def _bands(values: np.ndarray, edges: list[float], points: list[int]) -> np.ndarray:
    """Points for each value: `edges` are upper bounds (inclusive) of each band but the last."""
    idx = np.searchsorted(np.asarray(edges, dtype=float), values, side="left")
    out = np.asarray(points, dtype=float)[np.clip(idx, 0, len(points) - 1)]
    return np.where(np.isnan(values), np.nan, out)


def resp_rate_points(rr: np.ndarray) -> np.ndarray:
    # <=8: 3 | 9-11: 1 | 12-20: 0 | 21-24: 2 | >=25: 3
    return _bands(np.round(rr), [8, 11, 20, 24], [3, 1, 0, 2, 3])


def spo2_points(
    spo2: np.ndarray, copd: np.ndarray | None = None, on_oxygen: np.ndarray | None = None
) -> np.ndarray:
    spo2 = np.round(spo2)
    # Scale 1: <=91: 3 | 92-93: 2 | 94-95: 1 | >=96: 0
    scale1 = _bands(spo2, [91, 93, 95], [3, 2, 1, 0])
    if copd is None:
        return scale1
    # Scale 2: <=83: 3 | 84-85: 2 | 86-87: 1 | 88-92 (or >=93 on air): 0
    #          on oxygen: 93-94: 1 | 95-96: 2 | >=97: 3 (over-oxygenation risks CO2 retention)
    scale2 = _bands(spo2, [83, 85, 87], [3, 2, 1, 0])
    if on_oxygen is not None:
        high = _bands(spo2, [92, 94, 96], [0, 1, 2, 3])
        scale2 = np.where(np.asarray(on_oxygen, dtype=bool) & (spo2 >= 93), high, scale2)
    return np.where(np.asarray(copd, dtype=bool), scale2, scale1)


def oxygen_points(on_oxygen: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(on_oxygen, dtype=bool), 2.0, 0.0)


def consciousness_points(acvpu: np.ndarray) -> np.ndarray:
    """ACVPU: Alert scores 0; new Confusion, Voice, Pain or Unresponsive score 3."""
    acvpu = pd.Series(np.asarray(acvpu, dtype=object))
    return np.where(acvpu.isna() | (acvpu == "A"), 0.0, 3.0)


def sbp_points(sbp: np.ndarray) -> np.ndarray:
    # <=90: 3 | 91-100: 2 | 101-110: 1 | 111-219: 0 | >=220: 3
    return _bands(np.round(sbp), [90, 100, 110, 219], [3, 2, 1, 0, 3])


def heart_rate_points(hr: np.ndarray) -> np.ndarray:
    # <=40: 3 | 41-50: 1 | 51-90: 0 | 91-110: 1 | 111-130: 2 | >=131: 3
    return _bands(np.round(hr), [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])


def temp_points(temp: np.ndarray) -> np.ndarray:
    # <=35.0: 3 | 35.1-36.0: 1 | 36.1-38.0: 0 | 38.1-39.0: 1 | >=39.1: 2
    return _bands(np.round(temp, 1), [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2])


def news2(
    frame: pd.DataFrame,
    suffix: str = "",
    copd_column: str = "copd",
    oxygen_column: str = "on_oxygen",
    acvpu_column: str = "acvpu",
) -> pd.DataFrame:
    """NEWS2 component points and total for each row.

    `frame` holds vitals in columns `<vital><suffix>` (e.g. `spo2_median_1h`), plus optional
    COPD, oxygen and ACVPU columns. Rows with a missing vital get NaN for that component and
    for the total.
    """
    n = len(frame)
    col = {v: frame[f"{v}{suffix}"].to_numpy(dtype=float) for v in COMPONENTS}
    copd = frame[copd_column].to_numpy() if copd_column in frame else None
    oxygen = (
        frame[oxygen_column].fillna(False).to_numpy(bool)
        if oxygen_column in frame
        else np.zeros(n, dtype=bool)
    )
    acvpu = frame[acvpu_column].to_numpy() if acvpu_column in frame else np.full(n, "A")
    points = pd.DataFrame(
        {
            "news2_resp_rate": resp_rate_points(col["resp_rate"]),
            "news2_spo2": spo2_points(col["spo2"], copd, oxygen),
            "news2_oxygen": oxygen_points(oxygen),
            "news2_sbp": sbp_points(col["sbp"]),
            "news2_heart_rate": heart_rate_points(col["heart_rate"]),
            "news2_consciousness": consciousness_points(acvpu),
            "news2_temp": temp_points(col["temp_c"]),
        },
        index=frame.index,
    )
    points["news2_total"] = points.sum(axis=1, skipna=False)
    return points
