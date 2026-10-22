// Every /api/* call except the streamed chat (chat/stream/route.ts): forwarded to the backend
// with the website's secret (lib/backend.ts).
//
// A GET answer is the same for every visitor and changes only when new hours are scored, so
// Vercel's CDN keeps it for 5 minutes. After that it serves the stored copy and fetches a fresh
// one in the background (for up to a day), so a visitor rarely waits for the API to wake from
// zero, and repeat visits never reach the API or the SQL warehouse behind it.

import { backend, backendHeaders } from "@/lib/backend";

export const dynamic = "force-dynamic";

const CDN_CACHE = "max-age=300, stale-while-revalidate=86400";

async function forward(request: Request, init: RequestInit): Promise<Response> {
  const url = new URL(request.url);
  const path = url.pathname.replace(/^\/api/, "");
  let upstream: Response;
  try {
    upstream = await fetch(`${backend}${path}${url.search}`, { ...init, cache: "no-store" });
  } catch {
    return Response.json({ detail: "The API isn't answering yet; it may be waking up." }, { status: 503 });
  }
  const headers = new Headers({ "Content-Type": upstream.headers.get("Content-Type") ?? "application/json" });
  // Vercel's CDN reads this header and doesn't pass it on; errors are never stored
  if (init.method === "GET" && upstream.ok) headers.set("Vercel-CDN-Cache-Control", CDN_CACHE);
  return new Response(upstream.body, { status: upstream.status, headers });
}

export function GET(request: Request): Promise<Response> {
  return forward(request, { method: "GET", headers: backendHeaders(request) });
}

export async function POST(request: Request): Promise<Response> {
  return forward(request, {
    method: "POST",
    headers: backendHeaders(request, { "Content-Type": "application/json" }),
    body: await request.text(),
  });
}
