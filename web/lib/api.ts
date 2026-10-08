"use client";

import { useEffect, useState } from "react";

// All calls go to /api/*, which Next rewrites to the FastAPI backend (next.config.ts).

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    /* not JSON */
  }
  return response.statusText || "Request failed";
}

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`/api${path}`, { cache: "no-store" });
  if (!response.ok) throw new ApiError(await readError(response), response.status);
  return response.json();
}

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new ApiError(await readError(response), response.status);
  return response.json();
}

export interface ApiState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/** Fetch `path` (or nothing while it's null) and refetch whenever it changes. */
export function useApi<T>(path: string | null): ApiState<T> {
  const [state, setState] = useState<ApiState<T>>({ data: null, error: null, loading: !!path });

  useEffect(() => {
    if (!path) return;
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: null }));
    getJson<T>(path)
      .then((data) => !cancelled && setState({ data, error: null, loading: false }))
      .catch((e: Error) => !cancelled && setState({ data: null, error: e.message, loading: false }));
    return () => {
      cancelled = true;
    };
  }, [path]);

  return state;
}
