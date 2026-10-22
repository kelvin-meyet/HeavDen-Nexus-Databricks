// Streams the assistant's answer from the backend to the browser as it is produced.
//
// Every other /api/* call goes through app/api/[...path]/route.ts, which waits for the whole
// answer; this more specific route takes precedence and passes the backend's server-sent
// events straight through.

import { backend, backendHeaders } from "@/lib/backend";

export const dynamic = "force-dynamic";

export async function POST(request: Request): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(`${backend}/chat/stream`, {
      method: "POST",
      headers: backendHeaders(request, { "Content-Type": "application/json", Accept: "text/event-stream" }),
      body: await request.text(),
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return Response.json({ detail: "The assistant service isn't answering yet; it may be waking up." }, { status: 503 });
  }

  if (!upstream.ok || !upstream.body) {
    return new Response(upstream.body, {
      status: upstream.status,
      headers: { "Content-Type": upstream.headers.get("Content-Type") ?? "application/json" },
    });
  }
  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
    },
  });
}
