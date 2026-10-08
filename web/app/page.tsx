"use client";

import Link from "next/link";

import { ErrorBox, Loading } from "@/components/bits";
import { ObsChart, ZoneKey } from "@/components/ObsChart";
import { useApi } from "@/lib/api";
import { num, pct, riskPct, unitName } from "@/lib/format";
import { VITALS } from "@/lib/news2";
import type { Board, PatientDetail, Summary } from "@/lib/types";

import styles from "./home.module.css";

export default function Home() {
  const summary = useApi<Summary>("/analytics/summary");
  const board = useApi<Board>("/risk/board?limit=1");
  const top = board.data?.patients[0];
  const detail = useApi<PatientDetail>(top ? `/risk/patients/${top.encounter_id}?hours=24` : null);
  const chat = useApi<{ live: boolean; questions: string[] }>("/chat/examples");
  const s = summary.data;

  return (
    <div className="page">
      <section className={styles.hero}>
        <div className={styles.heroText}>
          <h1>Which patients are getting worse, before they need rescuing?</h1>
          <p className="lede">
            HeavDen Nexus watches every patient on three hospitals&rsquo; wards, hour by hour, and flags the ones
            likely to need an emergency team in the next six hours. It explains why, and it answers questions in
            plain English.
          </p>
          <p className="muted">
            Everything here runs on invented patients, built to show how such a system is made, tested and
            monitored.
          </p>
        </div>

        <div className={styles.heroChart}>
          {board.error && <ErrorBox what="the ward" message={board.error} />}
          {!board.error && (!top || !detail.data) && <Loading lines={6} />}
          {top && detail.data && (
            <>
              <p className={styles.chartCaption}>
                <Link href={`/patients/${top.encounter_id}`}>
                  {top.patient_label}, {unitName(top.unit_id)}
                </Link>
                : currently the highest risk on any ward, {riskPct(top.risk)} chance of escalation within six
                hours.
              </p>
              <div className={styles.chartPair}>
                {(["resp_rate_mean_1h", "spo2_min_1h"] as const).map((key) => (
                  <ObsChart
                    key={key}
                    title={VITALS[key].label}
                    unit={VITALS[key].unit}
                    zones={VITALS[key].zones}
                    domain={VITALS[key].domain}
                    height={150}
                    points={detail.data!.series.map((r) => ({ t: r.prediction_ts, value: r[key] }))}
                  />
                ))}
              </div>
              <ZoneKey />
            </>
          )}
        </div>
      </section>

      <section aria-labelledby="board-title" className={styles.board}>
        <h2 id="board-title" className="visually-hidden">
          What you can do here
        </h2>

        <Link href="/patients" className={styles.row}>
          <span className={styles.rowName}>Patients</span>
          <span className={styles.rowFigure}>
            {s ? (
              <>
                <strong>{num(s.high_risk)}</strong> of {num(s.census)} patients are in the High band
              </>
            ) : (
              <span className="skeleton" style={{ display: "inline-block", width: 220 }} />
            )}
          </span>
          <span className={styles.rowWhat}>
            See the ward list, each patient&rsquo;s chart and reasons, and try what-if changes to their vital
            signs.
          </span>
        </Link>

        <Link href="/analytics" className={styles.row}>
          <span className={styles.rowName}>Analytics</span>
          <span className={styles.rowFigure}>
            {s ? (
              <>
                <strong>{pct(s.escalations_flagged_7d)}</strong> of {num(s.escalations_7d)} escalations this week
                were flagged in advance
              </>
            ) : (
              <span className="skeleton" style={{ display: "inline-block", width: 260 }} />
            )}
          </span>
          <span className={styles.rowWhat}>
            Census, alerts, escalations and device health by hospital and day.
          </span>
        </Link>

        <Link href="/assistant" className={styles.row}>
          <span className={styles.rowName}>Assistant</span>
          <span className={styles.rowFigure}>
            {chat.data ? (
              <>&ldquo;{chat.data.questions[0]}&rdquo;</>
            ) : (
              <span className="skeleton" style={{ display: "inline-block", width: 240 }} />
            )}
          </span>
          <span className={styles.rowWhat}>
            Ask about the numbers, the hospital&rsquo;s protocols or a patient. Every answer shows its working.
          </span>
        </Link>

        <Link href="/behind-the-scenes" className={styles.row}>
          <span className={styles.rowName}>Behind the scenes</span>
          <span className={styles.rowFigure}>How it was built, and what testing changed</span>
          <span className={styles.rowWhat}>
            The data platform, the model, how it&rsquo;s monitored and the mistakes caught along the way.
          </span>
        </Link>
      </section>

      {summary.error && <ErrorBox what="the summary" message={summary.error} />}
    </div>
  );
}
