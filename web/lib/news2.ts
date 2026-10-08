// NEWS2 scoring zones (Royal College of Physicians, 2017; scale 1), as drawn on paper
// observation charts. Each zone is the value range that scores 1, 2 or 3 points.
// Used to shade time-series charts the way a ward chart is shaded.

export type Zone = 1 | 2 | 3;

export interface VitalZones {
  key: string;
  label: string;
  unit: string;
  domain: [number, number];
  zones: { from: number; to: number; zone: Zone }[];
}

export const VITALS: Record<string, VitalZones> = {
  resp_rate_mean_1h: {
    key: "resp_rate_mean_1h",
    label: "Breathing rate",
    unit: "/min",
    domain: [6, 32],
    zones: [
      { from: 6, to: 8.5, zone: 3 },
      { from: 8.5, to: 11.5, zone: 1 },
      { from: 20.5, to: 24.5, zone: 2 },
      { from: 24.5, to: 32, zone: 3 },
    ],
  },
  spo2_min_1h: {
    key: "spo2_min_1h",
    label: "Oxygen saturation (lowest)",
    unit: "%",
    domain: [84, 100],
    zones: [
      { from: 84, to: 91.5, zone: 3 },
      { from: 91.5, to: 93.5, zone: 2 },
      { from: 93.5, to: 95.5, zone: 1 },
    ],
  },
  heart_rate_mean_1h: {
    key: "heart_rate_mean_1h",
    label: "Heart rate",
    unit: "/min",
    domain: [35, 145],
    zones: [
      { from: 35, to: 40.5, zone: 3 },
      { from: 40.5, to: 50.5, zone: 1 },
      { from: 90.5, to: 110.5, zone: 1 },
      { from: 110.5, to: 130.5, zone: 2 },
      { from: 130.5, to: 145, zone: 3 },
    ],
  },
  sbp_mean_1h: {
    key: "sbp_mean_1h",
    label: "Systolic blood pressure",
    unit: "mmHg",
    domain: [80, 180],
    zones: [
      { from: 80, to: 90.5, zone: 3 },
      { from: 90.5, to: 100.5, zone: 2 },
      { from: 100.5, to: 110.5, zone: 1 },
    ],
  },
  temp_c_mean_1h: {
    key: "temp_c_mean_1h",
    label: "Temperature",
    unit: "°C",
    domain: [34.5, 40],
    zones: [
      { from: 34.5, to: 35.05, zone: 3 },
      { from: 35.05, to: 36.05, zone: 1 },
      { from: 38.05, to: 39.05, zone: 1 },
      { from: 39.05, to: 40, zone: 2 },
    ],
  },
};

export const ZONE_FILL: Record<Zone, string> = {
  1: "var(--zone-1)",
  2: "var(--zone-2)",
  3: "var(--zone-3)",
};
