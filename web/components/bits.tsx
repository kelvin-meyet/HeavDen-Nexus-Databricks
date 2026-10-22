"use client";

import { useEffect, useState } from "react";

import { num, riskPct } from "@/lib/format";
import { readsWarehouse, useHealth } from "@/lib/health";
import type { Band, Factor } from "@/lib/types";

// A warm load takes a second or two, so one still running after this is almost always the API,
// or in live mode Databricks' SQL warehouse, waking up.
const SLOW_MS = 4000;

/** `children`, shown only once `after` milliseconds have passed. */
export function SlowNote({
  after = SLOW_MS,
  className = "wakingNote",
  children,
}: {
  after?: number;
  className?: string;
  children: React.ReactNode;
}) {
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => setShown(true), after);
    return () => clearTimeout(timer);
  }, [after]);
  return shown ? (
    <p className={className} role="status">
      {children}
    </p>
  ) : null;
}

/** Whole seconds since the component appeared. */
function useSecondsWaiting(): number {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const start = Date.now();
    const timer = setInterval(() => setSeconds(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(timer);
  }, []);
  return seconds;
}

const LINK_S = 1.6; // seconds for the pulse to cross from one stop to the next
const BASE = 20; // the trace's flat line

/** Flat between stops, with one heartbeat (the site's logo) on each link. */
function tracePath(stops: number, pad: number, gap: number): string {
  let d = `M ${pad} ${BASE}`;
  for (let i = 0; i < stops - 1; i++) {
    const x = pad + i * gap;
    d += ` L ${x + gap * 0.36} ${BASE} l 5 -3 l 5 3 l 7 3 l 5 -18 l 5 26 l 6 -11 l 9 -4 l 8 4 L ${x + gap} ${BASE}`;
  }
  return d;
}

/** The request's path, website to what is waking up: a pulse runs along a heartbeat trace and
 *  each stop lights as it arrives. Decorative: the words around it say the same. */
function Trace({ stops, labels }: { stops: string[]; labels: boolean }) {
  const pad = 14;
  const gap = labels ? 110 : 64;
  const width = pad * 2 + gap * (stops.length - 1);
  const height = labels ? 54 : 36;
  const d = tracePath(stops.length, pad, gap);
  // the pulse travels for 80% of a cycle, then rests (globals.css @keyframes wake-pulse)
  const cycle = ((stops.length - 1) * LINK_S) / 0.8;
  return (
    <svg
      className="wakeTrace"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      aria-hidden="true"
      style={{ "--cycle": `${cycle}s` } as React.CSSProperties}
    >
      <path className="wakeLine" d={d} />
      <path className="wakePulse" d={d} pathLength={100} />
      {stops.map((stop, i) => (
        <circle
          key={stop}
          className="wakeStop"
          cx={pad + i * gap}
          cy={BASE}
          r={5}
          style={{ "--at": `${i * LINK_S}s` } as React.CSSProperties}
        />
      ))}
      {labels &&
        stops.map((stop, i) => {
          const last = i === stops.length - 1;
          const x = pad + i * gap + (i === 0 ? -6 : last ? 6 : 0);
          return (
            <text key={stop} x={x} y={48} textAnchor={i === 0 ? "start" : last ? "end" : "middle"}>
              {stop}
            </text>
          );
        })}
    </svg>
  );
}

/** Said when a load is still running after a few seconds: the API, and in live mode Databricks'
 *  SQL warehouse, are waking up. `compact`: the trace and one line, for the dark top bar. */
export function WakeNotice({ compact = false, className = "" }: { compact?: boolean; className?: string }) {
  const health = useHealth();
  const seconds = useSecondsWaiting();
  if (seconds * 1000 < SLOW_MS) return null;
  const warehouse = readsWarehouse(health);
  const stops = warehouse ? ["Website", "API", "SQL warehouse"] : ["Website", "API"];
  const title = warehouse ? "Waking up the API and Databricks’ SQL warehouse" : "Waking up the API";
  return (
    <div className={`wake ${compact ? "wakeCompact" : ""} ${className}`} role="status">
      <Trace stops={stops} labels={!compact} />
      <div className="wakeText">
        {!compact && <p className="wakeTitle">{title}</p>}
        <p className="wakeDetail">
          {compact && <strong>{title}. </strong>}
          Cold starts take 30–40 seconds, saving compute cost.
          <span className="wakeTime" aria-hidden="true">
            {seconds} s so far
          </span>
        </p>
      </div>
    </div>
  );
}

export function Loading({ lines = 3 }: { lines?: number }) {
  return (
    <div aria-busy="true" aria-label="Loading" style={{ display: "grid", gap: 8 }}>
      <WakeNotice />
      {Array.from({ length: lines }, (_, i) => (
        <div key={i} className="skeleton" style={{ width: `${90 - i * 15}%` }} />
      ))}
    </div>
  );
}

export function ErrorBox({ message, what }: { message: string; what: string }) {
  return (
    <div className="errorBox" role="alert">
      <strong>Couldn&rsquo;t load {what}.</strong> {message}. If the data service was asleep, reload in a few
      seconds.
    </div>
  );
}

export function BandTag({ band }: { band: Band }) {
  return band === "High" ? (
    <span className="bandHigh">High, alert</span>
  ) : (
    <span className="bandLow">Low</span>
  );
}

/** Risk now, with the change over the last 6 hours (shown instead of a Medium band). */
export function RiskWithTrend({ risk, before }: { risk: number; before: number | null }) {
  if (before == null) return <span>{riskPct(risk)}</span>;
  const ratio = risk / Math.max(before, 1e-6);
  const rising = ratio >= 1.5;
  const falling = ratio <= 0.67;
  return (
    <span>
      {riskPct(risk)}{" "}
      <span className={rising ? "trendUp" : falling ? "trendDown" : "muted"}>
        {rising ? "rising" : falling ? "falling" : "steady"}
        <span className="visually-hidden">, was {riskPct(before)} 6 hours ago</span>
      </span>
    </span>
  );
}

/** The model's top reasons, grouped into factors a nurse would recognise. */
export function FactorList({ factors }: { factors: Factor[] }) {
  if (!factors.length) return <p className="muted">No reasons recorded for this hour.</p>;
  return (
    <ul style={{ margin: 0, paddingLeft: 18, display: "grid", gap: 6 }}>
      {factors.map((f) => (
        <li key={f.factor}>
          <strong className={f.direction === "raises risk" ? "trendUp" : "trendDown"}>
            {f.direction === "raises risk" ? "Raises" : "Lowers"} risk:
          </strong>{" "}
          {f.factor}
          <span className="muted">
            , mainly {f.driver}
            {f.driver_value != null ? ` at ${num(f.driver_value, 1)}` : ""}
          </span>
        </li>
      ))}
    </ul>
  );
}
