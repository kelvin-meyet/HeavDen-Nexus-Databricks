import type { SiteId } from "./types";

export const SITES: Record<SiteId, { name: string; short: string; city: string }> = {
  SITE_A: { name: "HeavDen General Hospital", short: "General Hospital", city: "Boston" },
  SITE_B: { name: "HeavDen Northshore Medical Center", short: "Northshore", city: "Salem" },
  SITE_C: { name: "HeavDen Valley Community Hospital", short: "Valley", city: "Worcester" },
};

export const SITE_IDS = Object.keys(SITES) as SiteId[];

const CONDITION_NAMES: Record<string, string> = { ckd: "CKD", copd: "COPD" };

/** "diabetes, ckd" -> "diabetes, CKD" */
export function conditions(list: string): string {
  if (list === "none") return "no long-term conditions";
  return list
    .split(", ")
    .map((c) => CONDITION_NAMES[c] ?? c)
    .join(", ");
}

export function siteName(id: string): string {
  return SITES[id as SiteId]?.short ?? id;
}

export function unitName(unitId: string): string {
  const [, site, kind] = unitId.match(/^(SITE_[A-C])-(.+)$/) ?? [];
  if (!site) return unitId;
  const label = { GENERAL: "general ward", STEP_DOWN: "step-down unit", RESPIRATORY: "respiratory ward" }[kind];
  const unit = label ?? kind.toLowerCase();
  return `${unit[0].toUpperCase()}${unit.slice(1)} at ${siteName(site)}`;
}

/** Risk is a probability; show it as a percentage with sensible precision. */
export function riskPct(p: number | null | undefined): string {
  if (p == null || Number.isNaN(p)) return "–";
  const pct = p * 100;
  return `${pct < 10 ? pct.toFixed(1) : pct.toFixed(0)}%`;
}

export function pct(share: number | null | undefined, digits = 0): string {
  if (share == null || Number.isNaN(share)) return "–";
  return `${(share * 100).toFixed(digits)}%`;
}

export function num(value: number | null | undefined, digits = 0): string {
  if (value == null || Number.isNaN(value)) return "–";
  return value.toLocaleString("en-GB", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

const timeFormat = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
  timeZone: "UTC",
});
const hourFormat = new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: "UTC" });
const dayFormat = new Intl.DateTimeFormat("en-GB", { weekday: "short", day: "numeric", month: "short", timeZone: "UTC" });

const fullFormat = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  timeZone: "UTC",
});

/** With the year, for the simulated clock: it runs on its own calendar, not today's. */
export function whenFull(iso: string | null | undefined): string {
  return iso ? `${fullFormat.format(new Date(iso))} UTC` : "–";
}

/** Simulation timestamps are UTC; say so rather than silently converting. */
export function when(iso: string | null | undefined): string {
  return iso ? `${timeFormat.format(new Date(iso))} UTC` : "–";
}

export function hourLabel(iso: string): string {
  return hourFormat.format(new Date(iso));
}

export function dayLabel(iso: string): string {
  return dayFormat.format(new Date(iso));
}

/** Hours between an ISO time and "now" (the snapshot's as_of), e.g. for the x axis. */
export function hoursBefore(iso: string, asOf: string): number {
  return (new Date(iso).getTime() - new Date(asOf).getTime()) / 3_600_000;
}
