"use client";

import { createContext, useContext } from "react";

import type { Health } from "./types";

// What /health last said (null until it answers), shared by the Shell with every page, so a
// slow section can say what is waking up.
export const HealthContext = createContext<Health | null>(null);

export function useHealth(): Health | null {
  return useContext(HealthContext);
}

/** True when the pages read Databricks live right now (live mode, today's allowance left). */
export function readsWarehouse(health: Health | null): boolean {
  return health?.mode === "live" && health.live_data !== false;
}
