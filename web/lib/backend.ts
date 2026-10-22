// How the website's own routes (app/api/) reach the FastAPI backend. Server side only: the
// browser only ever calls /api/*, and BACKEND_URL and ORIGIN_SECRET never reach it.

// Locally uvicorn on :8000; on Vercel, the API on Azure Container Apps.
export const backend = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

// The API's own URL is public, so on Container Apps it answers only requests carrying this
// secret (heavden_api.config.ORIGIN_HEADER). Unset locally: the local API doesn't check.
const ORIGIN_HEADER = "x-origin-secret";

export function backendHeaders(request: Request, extra: Record<string, string> = {}): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  // Keep the visitor's address for the backend's per-visitor limit (see chat/limits.py).
  // Vercel sets it to the visitor's IP, replacing anything the visitor sent.
  const forwarded = request.headers.get("x-forwarded-for");
  if (forwarded) headers["X-Forwarded-For"] = forwarded;
  const secret = process.env.ORIGIN_SECRET;
  if (secret) headers[ORIGIN_HEADER] = secret;
  return headers;
}
