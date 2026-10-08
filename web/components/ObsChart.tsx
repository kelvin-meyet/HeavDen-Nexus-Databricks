"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { hourLabel } from "@/lib/format";
import { SizedChart } from "./SizedChart";
import { type VitalZones, ZONE_FILL } from "@/lib/news2";

interface Point {
  t: string; // ISO time
  value: number | null;
}

interface ObsChartProps {
  title: string;
  points: Point[];
  unit?: string;
  /** NEWS2 zones to shade, like a paper observation chart */
  zones?: VitalZones["zones"];
  domain?: [number | "auto", number | "auto"];
  /** a horizontal line, e.g. the alert threshold */
  threshold?: { value: number; label: string };
  format?: (v: number) => string;
  height?: number;
  /** draw the line in on first render (once, honouring reduced-motion settings) */
  animate?: boolean;
}

function prefersReducedMotion() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * The project's signature: values plotted as dots joined by ink lines on chart paper,
 * over NEWS2 scoring zones, the way nurses chart observations by hand.
 */
export function ObsChart({
  title,
  points,
  unit = "",
  zones,
  domain,
  threshold,
  format,
  height = 180,
  animate = false,
}: ObsChartProps) {
  const fmt = format ?? ((v: number) => `${Math.round(v * 10) / 10}${unit}`);
  const latest = [...points].reverse().find((p) => p.value != null)?.value;
  const [lo, hi] = domain ?? ["auto", "auto"];

  return (
    <figure className="obsChart" style={{ margin: 0 }}>
      <figcaption className="obsTitle">
        <span>{title}</span>
        {latest != null && <span className="obsLatest">{fmt(latest)}</span>}
      </figcaption>
      <SizedChart height={height}>
        {(w, h) => (
          <LineChart width={w} height={h} data={points} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
            <CartesianGrid stroke="transparent" />
            {zones?.map((z) => (
              <ReferenceArea
                key={`${z.from}-${z.to}`}
                y1={z.from}
                y2={z.to}
                fill={ZONE_FILL[z.zone]}
                fillOpacity={0.85}
                ifOverflow="hidden"
              />
            ))}
            {threshold && (
              <ReferenceLine
                y={threshold.value}
                stroke="var(--alert)"
                strokeDasharray="5 4"
                label={{ value: threshold.label, position: "insideTopLeft", fill: "var(--alert)", fontSize: 12 }}
              />
            )}
            <XAxis
              dataKey="t"
              tickFormatter={hourLabel}
              minTickGap={28}
              tick={{ fontSize: 12, fill: "var(--graphite-soft)" }}
              stroke="var(--rule)"
            />
            <YAxis
              domain={[lo, hi]}
              allowDataOverflow
              tickFormatter={(v: number) => fmt(v)}
              width={64}
              tick={{ fontSize: 12, fill: "var(--graphite-soft)" }}
              stroke="var(--rule)"
            />
            <Tooltip
              formatter={(v) => [fmt(Number(v)), title]}
              labelFormatter={(t) => `${hourLabel(String(t))} UTC`}
              contentStyle={{ borderRadius: 6, borderColor: "var(--rule)" }}
            />
            <Line
              type="linear"
              dataKey="value"
              stroke="var(--ink)"
              strokeWidth={2.25}
              dot={{ r: 3, fill: "var(--sheet)", stroke: "var(--ink)", strokeWidth: 2 }}
              activeDot={{ r: 4 }}
              connectNulls={false}
              isAnimationActive={animate && !prefersReducedMotion()}
              animationDuration={1400}
              animationEasing="ease-out"
            />
          </LineChart>
        )}
      </SizedChart>
    </figure>
  );
}

export function ZoneKey() {
  return (
    <p className="zoneKey" aria-label="NEWS2 shading key">
      <span style={{ ["--swatch" as string]: "var(--zone-1)" }}>scores 1 point</span>
      <span style={{ ["--swatch" as string]: "var(--zone-2)" }}>2 points</span>
      <span style={{ ["--swatch" as string]: "var(--zone-3)" }}>3 points (NEWS2)</span>
    </p>
  );
}
