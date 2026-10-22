"use client";

import { useEffect, useState } from "react";

import { num, riskPct } from "@/lib/format";
import { readsWarehouse, useHealth } from "@/lib/health";
import type { Band, Factor, Health } from "@/lib/types";

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

export function wakingMessage(health: Health | null): string {
  return readsWarehouse(health)
    ? "Waking up the API and Databricks’ SQL warehouse: they sleep when nobody is visiting, to keep costs near zero (up to about 40 seconds)."
    : "Waking up the data service: it sleeps when nobody is visiting, to keep costs near zero (up to about 40 seconds).";
}

export function Loading({ lines = 3 }: { lines?: number }) {
  const health = useHealth();
  return (
    <div aria-busy="true" aria-label="Loading" style={{ display: "grid", gap: 8 }}>
      <SlowNote>{wakingMessage(health)}</SlowNote>
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
