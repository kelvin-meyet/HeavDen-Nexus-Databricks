"use client";

import Link from "next/link";

import { ErrorBox, Loading, RiskWithTrend } from "@/components/bits";
import { ObsChart, ZoneKey } from "@/components/ObsChart";
import { useApi } from "@/lib/api";
import { num, pct, riskPct, unitName } from "@/lib/format";
import { VITALS } from "@/lib/news2";
import type { Board, PatientDetail, Summary } from "@/lib/types";

import styles from "./home.module.css";

const SECTIONS = [
  {
    href: "/analytics",
    name: "Analytics",
    tone: "ink",
    text: "Census, alerts, escalations and device health by hospital and day.",
  },
  {
    href: "/assistant",
    name: "Assistant",
    tone: "amber",
    text: "Ask about the numbers, the protocols or a patient. Every answer shows its working.",
  },
  {
    href: "/behind-the-scenes",
    name: "Behind the scenes",
    tone: "ok",
    text: "How it was built, how well the model does, and what testing changed.",
  },
];

export default function Home() {
  const summary = useApi<Summary>("/analytics/summary");
  const board = useApi<Board>("/risk/board?band=High&limit=6");
  const top = board.data?.patients[0];
  const detail = useApi<PatientDetail>(top ? `/risk/patients/${top.encounter_id}?hours=24` : null);
  const s = summary.data;

  return (
    <div className="page">
      <section className={styles.hero}>
        <div className={styles.heroText}>
          <h1>Spot the patients who are getting worse, hours before they need rescuing.</h1>
          <p className="lede">
            HeavDen Nexus watches every patient on three hospitals&rsquo; wards, hour by hour. It flags those likely
            to need an emergency team within six hours, explains why, and answers questions in plain English.
          </p>

          <dl className={styles.vitals}>
            <div className={styles.vital} data-tone="alert">
              <dt>in the High band now</dt>
              <dd>{s ? num(s.high_risk) : "–"}</dd>
            </div>
            <div className={styles.vital} data-tone="ok">
              <dt>of escalations flagged in advance this week</dt>
              <dd>{s ? pct(s.escalations_flagged_7d) : "–"}</dd>
            </div>
            <div className={styles.vital} data-tone="amber">
              <dt>alerts per nurse per shift (budget 2)</dt>
              <dd>{s ? num(s.alerts_per_nurse_shift_24h, 1) : "–"}</dd>
            </div>
          </dl>
          <p className="muted small">All patients and hospitals are invented, built to show how such a system is made, tested and monitored.</p>
        </div>

        <div className={styles.heroChart}>
          {board.error && <ErrorBox what="the ward" message={board.error} />}
          {!board.error && (!top || !detail.data) && <Loading lines={6} />}
          {top && detail.data && (
            <>
              <p className={styles.chartCaption}>
                Highest risk right now:{" "}
                <Link href={`/patients/${top.encounter_id}`}>
                  {top.patient_label}, {unitName(top.unit_id)}
                </Link>
                , {riskPct(top.risk)} chance of escalation within six hours.
              </p>
              <div className={styles.chartPair}>
                {(["resp_rate_mean_1h", "spo2_min_1h"] as const).map((key) => (
                  <ObsChart
                    key={key}
                    title={VITALS[key].label}
                    unit={VITALS[key].unit}
                    zones={VITALS[key].zones}
                    domain={VITALS[key].domain}
                    height={190}
                    animate
                    points={detail.data!.series.map((r) => ({ t: r.prediction_ts, value: r[key] }))}
                  />
                ))}
              </div>
              <ZoneKey />
            </>
          )}
        </div>
      </section>

      <section className={styles.lower}>
        <div className={styles.ward}>
          <div className={styles.wardHead}>
            <h2>On the wards now</h2>
            <Link href="/patients?band=High" className={styles.more}>
              All {s ? num(s.high_risk) : ""} High-band patients
            </Link>
          </div>
          {board.data && (
            <ul className={styles.wardList}>
              {board.data.patients.map((p) => (
                <li key={p.encounter_id}>
                  <Link href={`/patients/${p.encounter_id}`} className={styles.wardRow}>
                    <span className={styles.wardName}>{p.patient_label}</span>
                    <span className="muted">{unitName(p.unit_id)}</span>
                    <span className={styles.wardRisk}>
                      <RiskWithTrend risk={p.risk} before={p.risk_6h_ago} />
                    </span>
                    <span className={styles.wardWhy}>{p.top_factors[0]?.factor ?? ""}</span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
          {!board.data && !board.error && <Loading lines={5} />}
        </div>

        <nav className={styles.sections} aria-label="Explore">
          {SECTIONS.map((section) => (
            <Link key={section.href} href={section.href} className={styles.section} data-tone={section.tone}>
              <span className={styles.sectionName}>{section.name}</span>
              <span className="muted">{section.text}</span>
            </Link>
          ))}
        </nav>
      </section>

      {summary.error && <ErrorBox what="the summary" message={summary.error} />}
    </div>
  );
}
