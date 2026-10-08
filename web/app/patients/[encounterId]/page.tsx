"use client";

import Link from "next/link";
import { useParams } from "next/navigation";

import { BandTag, ErrorBox, FactorList, Loading } from "@/components/bits";
import { ObsChart, ZoneKey } from "@/components/ObsChart";
import { WhatIf } from "@/components/WhatIf";
import { useApi } from "@/lib/api";
import { conditions, num, riskPct, unitName, when } from "@/lib/format";
import { VITALS } from "@/lib/news2";
import type { Health, PatientDetail } from "@/lib/types";

import styles from "./patient.module.css";

const CHARTED = ["resp_rate_mean_1h", "spo2_min_1h", "heart_rate_mean_1h", "sbp_mean_1h", "temp_c_mean_1h"] as const;

export default function PatientPage() {
  const { encounterId } = useParams<{ encounterId: string }>();
  const detail = useApi<PatientDetail>(`/risk/patients/${encounterId}?hours=24`);
  const health = useApi<Health>("/health");
  const d = detail.data;
  const p = d?.patient;
  const onWard = !!d && d.series.at(-1)?.prediction_ts === d.as_of;

  return (
    <div className="page">
      <p className="small">
        <Link href="/patients">Back to all patients</Link>
      </p>
      {detail.error && <ErrorBox what="this patient" message={detail.error} />}
      {!d && !detail.error && <Loading lines={6} />}
      {p && d && (
        <>
          <header className={styles.head}>
            <div>
              <h1>{p.patient_label}</h1>
              <p className="lede">
                {p.age}
                {p.sex}, {unitName(p.unit_id)}, bed {p.bed_id.split("-").at(-1)}.{" "}
                {p.conditions === "none" ? "No long-term conditions." : `Long-term conditions: ${conditions(p.conditions)}.`}
              </p>
              <p className="muted small">
                Admitted {when(p.admit_ts)}
                {p.discharge_ts ? `, leaves ${when(p.discharge_ts)}` : ""}.
              </p>
            </div>
            <div className={styles.riskNow}>
              {p.risk != null ? (
                <>
                  <span className={styles.riskValue}>{riskPct(p.risk)}</span>
                  <span>chance of escalation within 6 hours</span>
                  <BandTag band={p.risk_band} />
                  <span className="muted small">NEWS2 {num(p.news2_total)}</span>
                </>
              ) : (
                <span className="muted">No score at {when(d.as_of)}: the patient has left the ward.</span>
              )}
            </div>
          </header>

          <div className={styles.layout}>
            <section className="section" aria-labelledby="chart-title" style={{ marginTop: 0 }}>
              <h2 id="chart-title">Last 24 hours</h2>
              <ObsChart
                title="Risk of escalation within 6 hours"
                points={d.series.map((r) => ({ t: r.prediction_ts, value: r.risk }))}
                format={(v) => riskPct(v)}
                domain={[0, "auto"]}
                threshold={health.data ? { value: health.data.risk_bands.high, label: "alert threshold" } : undefined}
              />
              {CHARTED.map((key) => (
                <ObsChart
                  key={key}
                  title={VITALS[key].label}
                  unit={VITALS[key].unit}
                  zones={VITALS[key].zones}
                  domain={VITALS[key].domain}
                  height={130}
                  points={d.series.map((r) => ({ t: r.prediction_ts, value: r[key] }))}
                />
              ))}
              <ZoneKey />
            </section>

            <aside className={styles.side}>
              <section className="section" aria-labelledby="why-title" style={{ marginTop: 0 }}>
                <h2 id="why-title">Why this score</h2>
                <FactorList factors={p.top_factors} />
                <p className="muted small">
                  The biggest influences on the latest score, grouped the way a nurse would describe them.
                </p>
              </section>

              {onWard && <WhatIf encounterId={p.encounter_id} latest={d.series.at(-1)!} />}

              <section className="section" aria-labelledby="alerts-title">
                <h2 id="alerts-title">Alerts this stay</h2>
                {d.alerts.length === 0 ? (
                  <p className="muted">No alerts.</p>
                ) : (
                  <ul className={styles.alerts}>
                    {d.alerts.map((a) => (
                      <li key={a.alert_ts}>
                        {when(a.alert_ts)}, risk {riskPct(a.risk)}:{" "}
                        {a.escalated_within_6h == null
                          ? "too soon to tell"
                          : a.escalated_within_6h
                            ? `escalated ${num(a.hours_to_escalation, 1)} h later`
                            : "no escalation followed"}
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            </aside>
          </div>
        </>
      )}
    </div>
  );
}
