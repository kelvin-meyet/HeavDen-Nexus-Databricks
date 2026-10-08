import { ApiError } from "./api";
import type { ChatStep, Citation } from "./types";

// Events sent by POST /chat/stream (backend/heavden_api/chat/agent.py and routes/chat.py).
export type ChatEvent =
  | {
      type: "meta";
      mode: "live" | "recorded" | "guardrail";
      model: string | null;
      notice?: string;
      guardrail?: string;
      conversation_id?: string;
    }
  | { type: "step_start"; index: number; tool: string }
  | { type: "step"; index: number; step: ChatStep }
  | { type: "delta"; text: string }
  | { type: "reset" }
  | { type: "redact"; answer: string }
  | {
      type: "done";
      answer: string;
      steps: ChatStep[];
      citations: Citation[];
      suggestions?: string[];
      guardrail?: string;
    }
  | { type: "error"; message: string };

/** POST and yield each server-sent event. (EventSource can't POST, so read the stream.) */
export async function* streamChat(body: unknown, signal?: AbortSignal): AsyncGenerator<ChatEvent> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    let message = response.statusText || "Request failed";
    try {
      const detail = (await response.json()).detail;
      if (typeof detail === "string") message = detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(message, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("\n");
      if (data) yield JSON.parse(data) as ChatEvent;
    }
  }
}
