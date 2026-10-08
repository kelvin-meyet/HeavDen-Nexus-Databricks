// Shapes returned by the FastAPI backend (backend/heavden_api/routes).

export type Band = "Low" | "High";
export type SiteId = "SITE_A" | "SITE_B" | "SITE_C";

export interface Health {
  status: string;
  mode: "demo" | "live";
  synthetic: boolean;
  as_of: string;
  data_start: string;
  model: { name: string; version: string; type: string };
  risk_bands: { high: number };
  documents: boolean;
}

export interface Summary {
  as_of: string;
  site_id: SiteId | null;
  census: number;
  high_risk: number;
  alerts_24h: number;
  alerts_per_nurse_shift_24h: number | null;
  escalations_24h: number;
  alert_precision_7d: number | null;
  alerts_with_outcome_7d: number;
  escalations_7d: number;
  escalations_flagged_7d: number | null;
  median_hours_flagged_before_7d: number | null;
  alert_budget_per_nurse_shift: number;
  definitions: Record<string, string>;
}

export interface CensusRow {
  hour_ts: string;
  site_id: SiteId;
  census: number;
  n_low: number;
  n_high: number;
  alerts: number;
  escalations: number;
}

export interface AlertsDayRow {
  day: string;
  site_id: SiteId;
  alerts: number;
  followed_by_escalation: number;
  no_escalation: number;
  outcome_pending: number;
}

export interface DevicesDayRow {
  day: string;
  site_id: SiteId;
  devices_worn: number;
  uptime_pct: number;
  battery_outages: number;
  stuck_minutes: number;
  mean_spo2: number;
  firmware_versions: number;
  latest_firmware: string;
}

export interface Factor {
  factor: string;
  direction: "raises risk" | "lowers risk";
  log_odds: number;
  driver: string;
  driver_value: number | null;
}

export interface BoardPatient {
  encounter_id: string;
  patient_label: string;
  age: number;
  sex: string;
  conditions: string;
  site_id: SiteId;
  unit_id: string;
  bed_id: string;
  admit_ts: string;
  risk: number;
  risk_band: Band;
  news2_total: number | null;
  top_factors: Factor[];
  risk_6h_ago: number | null;
}

export interface Board {
  as_of: string;
  bands: { high: number };
  patients: BoardPatient[];
}

export interface SeriesRow {
  prediction_ts: string;
  risk: number;
  risk_band: Band;
  news2_total: number | null;
  heart_rate_mean_1h: number | null;
  resp_rate_mean_1h: number | null;
  spo2_min_1h: number | null;
  temp_c_mean_1h: number | null;
  sbp_mean_1h: number | null;
  on_oxygen: boolean;
  o2_flow_lpm: number;
  acvpu: string;
  unit_id: string;
}

export interface PatientDetail {
  as_of: string;
  patient: Omit<BoardPatient, "risk_6h_ago"> & {
    discharge_ts: string | null;
    model_version: string | null;
  };
  series: SeriesRow[];
  alerts: {
    alert_ts: string;
    risk: number;
    news2_total: number | null;
    escalated_within_6h: number | null;
    hours_to_escalation: number | null;
  }[];
}

export interface WhatIfChanges {
  heart_rate?: number;
  resp_rate?: number;
  spo2?: number;
  temp_c?: number;
  sbp?: number;
  on_oxygen?: boolean;
  acvpu?: "A" | "C" | "V" | "P" | "U";
}

export interface WhatIfResult {
  encounter_id: string;
  changes: WhatIfChanges;
  risk_before: number;
  risk_after: number;
  band_before: Band;
  band_after: Band;
  news2_before: number | null;
  news2_after: number | null;
  top_factors: Factor[];
}

export interface ChatStep {
  tool: "query_gold" | "search_documents" | "explain_patient" | string;
  purpose?: string;
  sql?: string;
  columns?: string[];
  rows?: unknown[][];
  row_count?: number;
  error?: string | boolean;
  query?: string;
  as_of?: string | null;
  refs?: number[];
  patient?: string;
}

export interface Citation {
  ref: number;
  citation: string;
  doc_id: string;
  version: string;
  text: string;
}

export interface ChatReply {
  mode: "live" | "recorded" | "guardrail";
  model: string | null;
  answer: string;
  steps: ChatStep[];
  citations: Citation[];
  suggestions?: string[];
  notice?: string;
}
