"use client";

import { useState } from "react";
import {
  Bar,
  BarChart,
  Legend,
  Line,
  LineChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ErrorBox, Loading } from "@/components/bits";
import { legendText, SizedChart } from "@/components/SizedChart";
import { useApi } from "@/lib/api";
import { dayLabel, hourLabel, num, pct, SITE_IDS, SITES, siteName, when } from "@/lib/format";
import type { AlertsDayRow, CensusRow, DevicesDayRow, SiteId, Summary } from "@/lib/types";

import styles from "./analytics.module.css";

const SITE_STROKE: Record<SiteId, string> = {
  SITE_A: "var(--ink)",
  SITE_B: "var(--amber)",
  SITE_C: "#0e9aa7",
};
const SITE_DASH: Record<SiteId, string | undefined> = { SITE_A: undefined, SITE_B: "6 3", SITE_C: "2 3" };

function siteQuery(site: SiteId | null, extra = "") {
  const parts = [site ? `site_id=${site}` : "", extra].filter(Boolean);
  return parts.length ? `?${parts.join("&")}` : "";
}

export default function AnalyticsPage() {
  const [site, setSite] = useState<SiteId | null>(null);
  const summary = useApi<Summary>(`/analytics/summary${siteQuery(site)}`);
  const census = useApi<CensusRow[]>(`/analytics/census${siteQuery(site, "hours=72")}`);
  const alerts = useApi<AlertsDayRow[]>(`/analytics/alerts${siteQuery(site, "days=7")}`);
  const devices = useApi<DevicesDayRow[]>(`/analytics/devices${siteQuery(site, "days=7")}`);
  const s = summary.data;

  return (
    <div className="page">
      <header className="pageHead">
        <h1>Analytics</h1>
        <p className="lede">How the wards and the early-warning system are doing{s ? `, as of ${when(s.as_of)}` : ""}.</p>
      </header>

      <div className="filters" role="group" aria-label="Hospital">
        <button className="buttonQuiet" aria-pressed={site === null} onClick={() => setSite(null)}>
          All hospitals
        </button>
        {SITE_IDS.map((id) => (
          <button key={id} className="buttonQuiet" aria-pressed={site === id} onClick={() => setSite(id)}>
            {SITES[id].short}
          </button>
        ))}
      </div>

      <section className="section" aria-labelledby="now-title">
        <h2 id="now-title">Now and this week</h2>
        {summary.error && <ErrorBox what="the summary" message={summary.error} />}
        {!s && !summary.error && <Loading lines={4} />}
        {s && (
          <dl className={styles.figures}>
            <Figure value={num(s.census)} label="patients on the ward" def={s.definitions.census} />
            <Figure value={num(s.high_risk)} label="in the High band (alerting)" def={s.definitions.high_risk} tone="alert" />
            <Figure
              value={num(s.alerts_per_nurse_shift_24h, 1)}
              label={`alerts per nurse per shift, last 24 h (budget ${s.alert_budget_per_nurse_shift})`}
              def={s.definitions.alerts_per_nurse_shift_24h}
              tone="amber"
            />
            <Figure
              value={pct(s.escalations_flagged_7d)}
              label={`of ${num(s.escalations_7d)} escalations flagged in advance, last 7 days`}
              def={s.definitions.escalations_flagged_7d}
              tone="ok"
            />
            <Figure
              value={s.median_hours_flagged_before_7d != null ? `${num(s.median_hours_flagged_before_7d, 1)} h` : "–"}
              label="typical warning before an escalation"
              def={s.definitions.median_hours_flagged_before_7d}
              tone="ok"
            />
            <Figure
              value={
                s.alert_precision_7d ? `1 in ${Math.round(1 / s.alert_precision_7d)}` : "–"
              }
              label="alerts followed by an escalation within 6 h"
              def={s.definitions.alert_precision_7d}
            />
          </dl>
        )}
        <p className="muted small">
          Most alerts are not followed by an escalation. That&rsquo;s expected for an early-warning system, and
          why the number of alerts is capped at a load nurses can handle.
        </p>
      </section>

      <div className={styles.charts}>
      <section className="section" aria-labelledby="high-title">
        <h2 id="high-title">Patients in the High band, last 72 hours</h2>
        {census.error && <ErrorBox what="the census" message={census.error} />}
        {!census.data && !census.error && <Loading />}
        {census.data && <HighBandChart rows={census.data} />}
      </section>

      <section className="section" aria-labelledby="alerts-title">
        <h2 id="alerts-title">Alerts per day, and what followed</h2>
        {alerts.error && <ErrorBox what="alerts" message={alerts.error} />}
        {!alerts.data && !alerts.error && <Loading />}
        {alerts.data && <AlertsChart rows={alerts.data} />}
        <p className="muted small">
          An alert fires when a patient enters the High band. The same patient doesn&rsquo;t alert again for 6
          hours. Outcomes stay pending until the 6-hour window has passed.
        </p>
      </section>

      </div>

      <section className="section" aria-labelledby="devices-title">
        <h2 id="devices-title">Wearable monitors</h2>
        {devices.error && <ErrorBox what="device health" message={devices.error} />}
        {!devices.data && !devices.error && <Loading />}
        {devices.data && <DevicesTable rows={devices.data} />}
      </section>
    </div>
  );
}

type Tone = "ink" | "alert" | "amber" | "ok";

function Figure({ value, label, def, tone = "ink" }: { value: string; label: string; def?: string; tone?: Tone }) {
  return (
    <div className={styles.figure} data-tone={tone}>
      <dt className={styles.figureLabel}>{label}</dt>
      <dd className={styles.figureValue}>{value}</dd>
      {def && (
        <dd className={styles.figureDef}>
          <details>
            <summary>What this counts</summary>
            <p>{def}</p>
          </details>
        </dd>
      )}
    </div>
  );
}

function bySite<T extends { site_id: SiteId }>(rows: T[], key: (r: T) => string, value: (r: T) => number) {
  const map = new Map<string, Record<string, number | string>>();
  for (const r of rows) {
    const k = key(r);
    const entry = map.get(k) ?? { t: k };
    entry[r.site_id] = value(r);
    map.set(k, entry);
  }
  return [...map.values()];
}

function HighBandChart({ rows }: { rows: CensusRow[] }) {
  const data = bySite(rows, (r) => r.hour_ts, (r) => r.n_high);
  const sites = SITE_IDS.filter((id) => rows.some((r) => r.site_id === id));
  return (
    <figure className="obsChart" style={{ margin: 0 }}>
      <SizedChart height={260}>
        {(w, h) => (
          <LineChart width={w} height={h} data={data} margin={{ top: 10, right: 12, bottom: 0, left: -16 }}>
            <XAxis
              dataKey="t"
              tickFormatter={(t: string) => (hourLabel(t) === "00:00" ? dayLabel(t) : hourLabel(t))}
              minTickGap={40}
              tick={{ fontSize: 12, fill: "var(--graphite-soft)" }}
              stroke="var(--rule)"
            />
            <YAxis allowDecimals={false} tick={{ fontSize: 12, fill: "var(--graphite-soft)" }} stroke="var(--rule)" />
            <Tooltip labelFormatter={(t) => when(String(t))} contentStyle={{ borderRadius: 6 }} />
            <Legend formatter={(v: string) => legendText(siteName(v))} />
            {sites.map((id) => (
              <Line
                key={id}
                dataKey={id}
                name={id}
                stroke={SITE_STROKE[id]}
                strokeDasharray={SITE_DASH[id]}
                strokeWidth={1.75}
                dot={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        )}
      </SizedChart>
    </figure>
  );
}

function AlertsChart({ rows }: { rows: AlertsDayRow[] }) {
  const days = new Map<string, { t: string; escalated: number; none: number; pending: number }>();
  for (const r of rows) {
    const d = days.get(r.day) ?? { t: r.day, escalated: 0, none: 0, pending: 0 };
    d.escalated += r.followed_by_escalation;
    d.none += r.no_escalation;
    d.pending += r.outcome_pending;
    days.set(r.day, d);
  }
  return (
    <figure className="obsChart" style={{ margin: 0 }}>
      <SizedChart height={260}>
        {(w, h) => (
          <BarChart width={w} height={h} data={[...days.values()]} margin={{ top: 10, right: 12, bottom: 0, left: -16 }}>
            <XAxis dataKey="t" tickFormatter={dayLabel} tick={{ fontSize: 12, fill: "var(--graphite-soft)" }} stroke="var(--rule)" />
            <YAxis allowDecimals={false} tick={{ fontSize: 12, fill: "var(--graphite-soft)" }} stroke="var(--rule)" />
            <Tooltip labelFormatter={(t) => dayLabel(String(t))} contentStyle={{ borderRadius: 6 }} />
            <Legend formatter={legendText} />
            <Bar dataKey="escalated" name="Escalation followed" stackId="a" fill="var(--alert)" isAnimationActive={false} />
            <Bar dataKey="none" name="No escalation" stackId="a" fill="#aebcf2" isAnimationActive={false} />
            <Bar dataKey="pending" name="Too soon to tell" stackId="a" fill="var(--zone-2)" radius={[6, 6, 0, 0]} isAnimationActive={false} />
          </BarChart>
        )}
      </SizedChart>
    </figure>
  );
}

function DevicesTable({ rows }: { rows: DevicesDayRow[] }) {
  const latestDay = rows.reduce((d, r) => (r.day > d ? r.day : d), "");
  const latest = rows.filter((r) => r.day === latestDay);
  return (
    <>
      <p className="muted small">On {dayLabel(latestDay)}:</p>
      <div className="tableWrap">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Hospital</th>
              <th scope="col" className="num">Monitors worn</th>
              <th scope="col" className="num">Readings received</th>
              <th scope="col" className="num">Battery outages</th>
              <th scope="col" className="num">Stuck-sensor minutes</th>
              <th scope="col" className="num">Average SpO2</th>
              <th scope="col">Firmware</th>
            </tr>
          </thead>
          <tbody>
            {latest.map((r) => (
              <tr key={r.site_id}>
                <th scope="row" style={{ fontWeight: 400 }}>{siteName(r.site_id)}</th>
                <td className="num">{num(r.devices_worn)}</td>
                <td className="num">{num(r.uptime_pct, 1)}%</td>
                <td className="num">{num(r.battery_outages)}</td>
                <td className="num">{num(r.stuck_minutes)}</td>
                <td className="num">{num(r.mean_spo2, 1)}%</td>
                <td>{r.latest_firmware}{r.firmware_versions > 1 ? " and older" : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small">
        A drop in one hospital&rsquo;s average SpO2 with nothing else changing points to a sensor fault, not
        sicker patients.
      </p>
    </>
  );
}
