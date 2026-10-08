"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { ErrorBox } from "@/components/bits";
import { useApi } from "@/lib/api";
import { type ChatEvent, streamChat } from "@/lib/stream";
import type { ChatReply, ChatStep } from "@/lib/types";

import styles from "./assistant.module.css";

/** A reply as it builds up from streamed events. `pending` names a tool that is running. */
interface LiveReply extends ChatReply {
  streaming: boolean;
  pending: string | null;
  guardrail?: string;
}

interface Turn {
  question: string;
  reply?: LiveReply;
  error?: string;
}

const PENDING: Record<string, string> = {
  query_gold: "Querying the data",
  search_documents: "Searching the documents",
  explain_patient: "Looking up the patient",
};

function applyEvent(reply: LiveReply | undefined, event: ChatEvent): LiveReply {
  const r: LiveReply = reply ?? {
    mode: "live",
    model: null,
    answer: "",
    steps: [],
    citations: [],
    streaming: true,
    pending: null,
  };
  switch (event.type) {
    case "meta":
      return { ...r, mode: event.mode as ChatReply["mode"], model: event.model, notice: event.notice, guardrail: event.guardrail };
    case "step_start":
      return { ...r, pending: event.tool };
    case "step": {
      const steps = [...r.steps];
      steps[event.index] = event.step;
      return { ...r, steps, pending: null };
    }
    case "delta":
      return { ...r, answer: r.answer + event.text, pending: null };
    case "reset":
      return { ...r, answer: "" };
    case "redact":
      return { ...r, answer: event.answer, guardrail: "leak" };
    case "done":
      return {
        ...r,
        answer: event.answer,
        steps: event.steps,
        citations: event.citations,
        suggestions: event.suggestions,
        guardrail: event.guardrail ?? r.guardrail,
        streaming: false,
        pending: null,
      };
    case "error":
      return { ...r, streaming: false, pending: null, notice: event.message };
  }
}

export default function AssistantPage() {
  const examples = useApi<{ live: boolean; questions: string[] }>("/chat/examples");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  // The server remembers the conversation; we only hold its id (see backend chat/memory.py).
  const [conversationId, setConversationId] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length]);

  async function ask(question: string) {
    const text = question.trim();
    if (text.length < 2 || busy) return;
    setDraft("");
    setBusy(true);
    setTurns((all) => [...all, { question: text }]);
    const update = (fn: (t: Turn) => Turn) =>
      setTurns((all) => all.map((t, i) => (i === all.length - 1 ? fn(t) : t)));
    try {
      for await (const event of streamChat({ message: text, conversation_id: conversationId })) {
        if (event.type === "meta" && event.conversation_id) setConversationId(event.conversation_id);
        update((t) => ({ ...t, reply: applyEvent(t.reply, event) }));
      }
      update((t) => (t.reply ? { ...t, reply: { ...t.reply, streaming: false, pending: null } } : t));
    } catch (e) {
      update((t) => ({ ...t, error: (e as Error).message }));
    } finally {
      setBusy(false);
    }
  }

  const live = examples.data?.live;

  return (
    <div className="page">
      <header className="pageHead">
        <div className={styles.titleRow}>
          <h1>Assistant</h1>
          {turns.length > 0 && (
            <button
              className="buttonQuiet"
              type="button"
              disabled={busy}
              onClick={() => {
                setTurns([]);
                setConversationId(null);
              }}
            >
              New conversation
            </button>
          )}
        </div>
        <p className="lede">
          Ask about the wards&rsquo; numbers, the hospital&rsquo;s protocols and documents, or why a patient is
          flagged. Each answer shows the queries and sources behind it.
        </p>
        {examples.data && (
          <p className="muted small">
            {live
              ? "Answers are written live by a language model, using the tools shown with each answer."
              : "Live answers are switched off in this demo, so the questions below replay answers recorded earlier."}{" "}
            The assistant can be wrong; check its working, and never use it for real clinical decisions.
          </p>
        )}
      </header>

      <div className={styles.thread}>
        {turns.map((turn, i) => (
          <article key={i} className={styles.turn}>
            <p className={styles.question}>{turn.question}</p>
            {!turn.reply && !turn.error && (
              <p className="muted" aria-live="polite">
                Looking that up&hellip;
              </p>
            )}
            {turn.error && <ErrorBox what="an answer" message={turn.error} />}
            {turn.reply && <Answer reply={turn.reply} onAsk={ask} />}
          </article>
        ))}
        <div ref={endRef} />
      </div>

      {(turns.length === 0 || !live) && examples.data && (
        <section className="section" aria-labelledby="try-title">
          <h2 id="try-title" className={styles.tryTitle}>
            {turns.length === 0 ? "Try one of these" : "Other questions to try"}
          </h2>
          <ul className={styles.examples}>
            {examples.data.questions.map((q) => (
              <li key={q}>
                <button className={styles.example} onClick={() => ask(q)} disabled={busy}>
                  {q}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <form
        className={styles.ask}
        onSubmit={(e) => {
          e.preventDefault();
          ask(draft);
        }}
      >
        <label htmlFor="question" className="visually-hidden">
          Your question
        </label>
        <textarea
          id="question"
          className={styles.input}
          rows={2}
          maxLength={1000}
          placeholder={live ? "Ask a question about the wards, documents or a patient" : "Choose a question above"}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              ask(draft);
            }
          }}
        />
        <button className="button" type="submit" disabled={busy || draft.trim().length < 2}>
          {busy ? "Answering…" : "Ask"}
        </button>
      </form>
    </div>
  );
}

function Answer({ reply, onAsk }: { reply: LiveReply; onAsk: (q: string) => void }) {
  const source =
    reply.mode === "guardrail" || reply.guardrail
      ? "Not answered: this is outside what the assistant can share"
      : reply.mode === "live"
        ? `Live answer${reply.model ? ` from ${reply.model}` : ""}`
        : "Recorded answer";
  return (
    <div className={styles.answer} aria-busy={reply.streaming}>
      <p className={styles.source}>
        {source}
        {reply.notice ? `. ${reply.notice}` : ""}
      </p>
      {reply.pending && (
        <p className={styles.pending} aria-live="polite">
          {PENDING[reply.pending] ?? "Working"}&hellip;
        </p>
      )}
      {reply.answer ? (
        <div className={styles.markdown}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{reply.answer}</ReactMarkdown>
        </div>
      ) : (
        !reply.pending && reply.streaming && <p className="muted">Thinking&hellip;</p>
      )}

      {reply.suggestions && reply.suggestions.length > 0 && (
        <ul className={styles.examples}>
          {reply.suggestions.slice(0, 4).map((q) => (
            <li key={q}>
              <button className={styles.example} onClick={() => onAsk(q)}>
                {q}
              </button>
            </li>
          ))}
        </ul>
      )}

      {!reply.streaming && (reply.steps.length > 0 || reply.citations.length > 0) && (
        <details className={styles.working}>
          <summary>
            How this answer was found ({reply.steps.length} {reply.steps.length === 1 ? "step" : "steps"}
            {reply.citations.length ? `, ${reply.citations.length} sources` : ""})
          </summary>
          <ol className={styles.steps}>
            {reply.steps.filter(Boolean).map((step, i) => (
              <li key={i}>
                <Step step={step} />
              </li>
            ))}
          </ol>
          {reply.citations.length > 0 && (
            <>
              <h3 className={styles.sourcesTitle}>Sources</h3>
              <ol className={styles.sources}>
                {reply.citations.map((c) => (
                  <li key={c.ref} value={c.ref}>
                    <details>
                      <summary>{c.citation}</summary>
                      <div className={styles.markdown}>
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{c.text}</ReactMarkdown>
                      </div>
                    </details>
                  </li>
                ))}
              </ol>
            </>
          )}
        </details>
      )}
    </div>
  );
}

function Step({ step }: { step: ChatStep }) {
  if (step.tool === "query_gold") {
    return (
      <div className={styles.step}>
        <p>
          <strong>Queried the data.</strong> {step.purpose && <span className="muted">{step.purpose}</span>}
        </p>
        {step.sql && <pre className={styles.sql}>{step.sql.trim()}</pre>}
        {step.error ? (
          <p className="small trendUp">The query failed, so the assistant tried again: {String(step.error)}</p>
        ) : (
          step.columns && step.rows && <ResultTable columns={step.columns} rows={step.rows} total={step.row_count} />
        )}
      </div>
    );
  }
  if (step.tool === "search_documents") {
    return (
      <div className={styles.step}>
        <p>
          <strong>Searched the documents</strong> for &ldquo;{step.query}&rdquo;
          {step.as_of ? ` (versions in force on ${step.as_of})` : ""}
          {step.refs?.length ? <span className="muted"> and found sources {step.refs.join(", ")}</span> : null}
        </p>
      </div>
    );
  }
  if (step.tool === "explain_patient") {
    return (
      <div className={styles.step}>
        <p>
          <strong>Looked up patient {step.patient}.</strong>{" "}
          {step.error && typeof step.error === "string" && <span className="muted">{step.error}</span>}
        </p>
      </div>
    );
  }
  return (
    <div className={styles.step}>
      <strong>{step.tool}</strong>
    </div>
  );
}

function ResultTable({ columns, rows, total }: { columns: string[]; rows: unknown[][]; total?: number }) {
  if (rows.length === 0) return <p className="small muted">No rows returned.</p>;
  return (
    <div className="tableWrap">
      <table className="table">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c} scope="col">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 12).map((row, i) => (
            <tr key={i}>
              {row.map((v, j) => (
                <td key={j} className={typeof v === "number" ? "num" : undefined}>
                  {v == null ? "–" : typeof v === "number" ? Number(v.toFixed(4)).toString() : String(v)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {total != null && total > 12 && <p className="small muted" style={{ padding: "6px 10px" }}>Showing 12 of {total} rows.</p>}
    </div>
  );
}
