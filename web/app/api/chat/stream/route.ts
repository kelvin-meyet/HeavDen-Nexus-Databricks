// Streams the assistant's answer from the backend to the browser as it is produced.
//
// Every other /api/* call goes through the rewrite in next.config.ts, but that proxy buffers
// the whole response, which would defeat streaming. This route takes precedence over the
// rewrite and passes the backend's server-sent events straight through.

export const dynamic = "force-dynamic";

const backend = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request): Promise<Response> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "text/event-stream",
  };
  // Keep the visitor's address for the backend's per-visitor limit (see chat/limits.py).
  const forwarded = request.headers.get("x-forwarded-for");
  if (forwarded) headers["X-Forwarded-For"] = forwarded;

  let upstream: Response;
  try {
    upstream = await fetch(`${backend}/chat/stream`, {
      method: "POST",
      headers,
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
