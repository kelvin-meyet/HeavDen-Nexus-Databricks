"use client";

import Link from "next/link";

import { ErrorBox, Loading, RiskWithTrend } from "@/components/bits";
import { ObsChart, ZoneKey } from "@/components/ObsChart";
import { useApi } from "@/lib/api";
import { num, pct, riskPct, unitName } from "@/lib/format";
import { VITALS } from "@/lib/news2";
import { BUILDER, MAILTO } from "@/lib/site";
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
  {
    href: "/work-with-me",
    name: "Work with me",
    tone: "ink",
    text: "The engineer behind HeavDen: what I build for teams, in any industry.",
  },
];

// What powers the product. Each figure is measured (README results, the dev pipeline run of
// 9 Oct 2026, the assistant's red-team run); update them when those change.
const POWERS = [
  {
    name: "A predictive model that must beat the clinical standard",
    proof: "Flagged 25 of 28 escalations in advance, against 21 for NEWS2, with the same alert budget.",
    detail: "Calibrated risk every hour, with the reasons behind each score.",
    href: "/behind-the-scenes",
    link: "See the results",
  },
  {
    name: "A governed data platform on Azure Databricks",
    proof: "153,311 monitor readings checked and reconciled in the latest run: none lost.",
    detail: "Hospital records and device files flow through bronze, silver and gold, under Unity Catalog.",
    href: "/about-the-data",
    link: "How the data flows",
  },
  {
    name: "An AI assistant that shows its working",
    proof: "Passed all 19 adversarial tests: no leaks, no off-topic answers, typos understood.",
    detail: "Answers from the data, the hospital's documents and each patient's risk, with its queries and sources.",
    href: "/assistant",
    link: "Ask it something",
  },
  {
    name: "MLOps built to run for years, cheaply",
    proof: "Every change tested in CI; serverless, designed to run for under $25 a month.",
    detail: "Drift monitoring and human-approved retraining are being built next.",
    href: "/behind-the-scenes",
    link: "How it's built",
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
          <p className={styles.eyebrow}>HeavDen Nexus · AI early warning for hospital wards</p>
          <h1>Spot the patients who are getting worse, hours before they need rescuing.</h1>
          <p className="lede">
            A predictive model, a data platform on Azure Databricks and an AI assistant, working together. HeavDen
            Nexus watches every patient on three hospitals&rsquo; wards, hour by hour, flags those likely to need an
            emergency team within six hours, explains why, and answers questions in plain English.
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
          <p className="muted small">
            Figures at the simulated ward time shown in the sidebar. All patients and hospitals are invented, built to
            show how such a system is made, tested and monitored.
          </p>
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

      <section className={styles.powers} aria-labelledby="powers-title">
        <h2 id="powers-title">What powers it</h2>
        <ul className={styles.powerList}>
          {POWERS.map((p) => (
            <li key={p.name} className={styles.power}>
              <h3>{p.name}</h3>
              <p className={styles.powerProof}>{p.proof}</p>
              <p className="muted small">{p.detail}</p>
              <Link href={p.href} className={styles.more}>
                {p.link}
              </Link>
            </li>
          ))}
        </ul>
      </section>

      <section className={styles.builder} aria-labelledby="builder-title">
        <div>
          <h2 id="builder-title">Built end to end by one engineer</h2>
          <p>
            Designed and built by <strong>{BUILDER.name}</strong>, {BUILDER.title} and the developer behind{" "}
            {BUILDER.brand}: from the synthetic data and the cloud platform to the model, the assistant and this app.
            Available for roles, contracts and projects in any industry: data platforms on Databricks and beyond,
            predictive models and MLOps, and AI assistants.
          </p>
        </div>
        <div className={styles.builderActions}>
          <Link href="/work-with-me" className="button">
            Work with me
          </Link>
          <a href={BUILDER.linkedin} className="buttonQuiet" rel="noopener noreferrer" target="_blank">
            LinkedIn<span className="visually-hidden"> (opens in a new tab)</span>
          </a>
          <a href={MAILTO} className="buttonQuiet">
            Email
          </a>
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
