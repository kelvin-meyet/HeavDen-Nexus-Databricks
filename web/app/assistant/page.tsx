"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { ErrorBox } from "@/components/bits";
import { postJson, useApi } from "@/lib/api";
import type { ChatReply, ChatStep } from "@/lib/types";

import styles from "./assistant.module.css";

interface Turn {
  question: string;
  reply?: ChatReply;
  error?: string;
}

export default function AssistantPage() {
  const examples = useApi<{ live: boolean; questions: string[] }>("/chat/examples");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  async function ask(question: string) {
    const text = question.trim();
    if (text.length < 2 || busy) return;
    setDraft("");
    setBusy(true);
    const history = turns
      .filter((t) => t.reply)
      .slice(-5)
      .flatMap((t) => [
        { role: "user", content: t.question },
        { role: "assistant", content: t.reply!.answer },
      ]);
    setTurns((all) => [...all, { question: text }]);
    try {
      const reply = await postJson<ChatReply>("/chat", { message: text, history });
      setTurns((all) => all.map((t, i) => (i === all.length - 1 ? { ...t, reply } : t)));
    } catch (e) {
      const error = (e as Error).message;
      setTurns((all) => all.map((t, i) => (i === all.length - 1 ? { ...t, error } : t)));
    } finally {
      setBusy(false);
    }
  }

  const live = examples.data?.live;

  return (
    <div className="page">
      <header className="pageHead">
        <h1>Assistant</h1>
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

function Answer({ reply, onAsk }: { reply: ChatReply; onAsk: (q: string) => void }) {
  return (
    <div className={styles.answer}>
      <p className={styles.source}>
        {reply.mode === "live" ? `Live answer${reply.model ? ` from ${reply.model}` : ""}` : "Recorded answer"}
        {reply.notice ? `. ${reply.notice}` : ""}
      </p>
      <div className={styles.markdown}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{reply.answer}</ReactMarkdown>
      </div>

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

      {(reply.steps.length > 0 || reply.citations.length > 0) && (
        <details className={styles.working}>
          <summary>
            How this answer was found ({reply.steps.length} {reply.steps.length === 1 ? "step" : "steps"}
            {reply.citations.length ? `, ${reply.citations.length} sources` : ""})
          </summary>
          <ol className={styles.steps}>
            {reply.steps.map((step, i) => (
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
