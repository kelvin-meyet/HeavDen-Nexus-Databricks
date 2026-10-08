"use client";

import { useState } from "react";

import { postJson } from "@/lib/api";
import { num, riskPct } from "@/lib/format";
import type { SeriesRow, WhatIfChanges, WhatIfResult } from "@/lib/types";

import { BandTag, FactorList } from "./bits";
import styles from "./WhatIf.module.css";

type VitalKey = "resp_rate" | "spo2" | "heart_rate" | "sbp" | "temp_c";

const SLIDERS: { key: VitalKey; label: string; unit: string; min: number; max: number; step: number; from: keyof SeriesRow }[] = [
  { key: "resp_rate", label: "Breathing rate", unit: "/min", min: 6, max: 40, step: 1, from: "resp_rate_mean_1h" },
  { key: "spo2", label: "Oxygen saturation", unit: "%", min: 75, max: 100, step: 1, from: "spo2_min_1h" },
  { key: "heart_rate", label: "Heart rate", unit: "/min", min: 35, max: 160, step: 1, from: "heart_rate_mean_1h" },
  { key: "sbp", label: "Systolic blood pressure", unit: "mmHg", min: 70, max: 200, step: 1, from: "sbp_mean_1h" },
  { key: "temp_c", label: "Temperature", unit: "°C", min: 34, max: 41, step: 0.1, from: "temp_c_mean_1h" },
];

const ACVPU = [
  ["A", "Alert"],
  ["C", "New confusion"],
  ["V", "Responds to voice"],
  ["P", "Responds to pain"],
  ["U", "Unresponsive"],
] as const;

function initialValues(latest: SeriesRow) {
  const values = {} as Record<VitalKey, number>;
  for (const s of SLIDERS) {
    const v = latest[s.from] as number | null;
    values[s.key] = v == null ? (s.min + s.max) / 2 : Math.round(v / s.step) * s.step;
  }
  return values;
}

export function WhatIf({ encounterId, latest }: { encounterId: string; latest: SeriesRow }) {
  const start = initialValues(latest);
  const [values, setValues] = useState(start);
  const [oxygen, setOxygen] = useState(latest.on_oxygen);
  const [acvpu, setAcvpu] = useState<WhatIfChanges["acvpu"]>((latest.acvpu as WhatIfChanges["acvpu"]) ?? "A");
  const [result, setResult] = useState<WhatIfResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const changes: WhatIfChanges = {};
  for (const s of SLIDERS) if (values[s.key] !== start[s.key]) changes[s.key] = values[s.key];
  if (oxygen !== latest.on_oxygen) changes.on_oxygen = oxygen;
  if (acvpu !== latest.acvpu) changes.acvpu = acvpu;
  const changed = Object.keys(changes).length > 0;

  async function score() {
    setBusy(true);
    setError(null);
    try {
      setResult(await postJson<WhatIfResult>("/risk/score", { encounter_id: encounterId, changes }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setValues(start);
    setOxygen(latest.on_oxygen);
    setAcvpu((latest.acvpu as WhatIfChanges["acvpu"]) ?? "A");
    setResult(null);
  }

  return (
    <section className="section" aria-labelledby="whatif-title">
      <h2 id="whatif-title">What if?</h2>
      <p className="muted small">
        Change this hour&rsquo;s observations and see how the model would respond. Nothing is saved.
      </p>
      <form
        className={styles.form}
        onSubmit={(e) => {
          e.preventDefault();
          if (changed) score();
        }}
      >
        {SLIDERS.map((s) => (
          <label key={s.key} className={styles.slider}>
            <span className={styles.sliderHead}>
              <span>{s.label}</span>
              <output className={values[s.key] !== start[s.key] ? styles.moved : undefined}>
                {num(values[s.key], s.step < 1 ? 1 : 0)}
                {s.unit}
              </output>
            </span>
            <input
              type="range"
              min={s.min}
              max={s.max}
              step={s.step}
              value={values[s.key]}
              onChange={(e) => setValues({ ...values, [s.key]: Number(e.target.value) })}
            />
          </label>
        ))}
        <label className={styles.check}>
          <input type="checkbox" checked={oxygen} onChange={(e) => setOxygen(e.target.checked)} /> On supplemental
          oxygen
        </label>
        <label>
          Consciousness{" "}
          <select className="select" value={acvpu} onChange={(e) => setAcvpu(e.target.value as WhatIfChanges["acvpu"])}>
            {ACVPU.map(([code, label]) => (
              <option key={code} value={code}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <div className="filters">
          <button className="button" type="submit" disabled={!changed || busy}>
            {busy ? "Scoring…" : "Score these changes"}
          </button>
          <button className="buttonQuiet" type="button" onClick={reset}>
            Reset to measured
          </button>
        </div>
      </form>

      {error && (
        <div className="errorBox" role="alert">
          Couldn&rsquo;t score the changes: {error}
        </div>
      )}

      {result && (
        <div className={styles.result} aria-live="polite">
          <div className={styles.compare}>
            <div>
              <span className="muted small">Measured</span>
              <strong>{riskPct(result.risk_before)}</strong>
              <BandTag band={result.band_before} />
              <span className="small">NEWS2 {num(result.news2_before)}</span>
            </div>
            <div>
              <span className="muted small">With your changes</span>
              <strong className={styles.after}>{riskPct(result.risk_after)}</strong>
              <BandTag band={result.band_after} />
              <span className="small">NEWS2 {num(result.news2_after)}</span>
            </div>
          </div>
          <h3>What drives the new score</h3>
          <FactorList factors={result.top_factors} />
        </div>
      )}
    </section>
  );
}
