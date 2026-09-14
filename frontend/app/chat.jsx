"use client";

import { useEffect, useRef, useState } from "react";

import { recommendedPlan } from "./preprocess";

import { API } from "./apiclient";

// Tiny, safe markdown -> HTML for chat bubbles: escape first (LLM output is
// untrusted), then bold / inline-code / bullets / line breaks. ponytail: not
// full markdown (no headings/tables/links) -- add a real parser if needed.
function mdToHtml(t) {
  const esc = t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/^\s*[-*]\s+(.+)$/gm, "• $1")
    .replace(/\n/g, "<br/>");
}

const MAX_FACT_COLUMNS = 100; // our starting point: keeps very wide tables inside the AI's context window

// Measured facts (profile + audit) for EVERY column -- no data rows -- so the AI can
// judge the chosen target against the alternatives instead of defending the choice.
function columnFacts(data) {
  const checks = data.report.checks;
  return data.profile.column_names.slice(0, MAX_FACT_COLUMNS).map((name) => {
    const info = data.profile.columns_info[name];
    return {
      name,
      detected_type: checks.column_types.columns.find((c) => c.column === name)?.detected_type,
      unique_values: info.unique_values,
      missing_percent: info.missing_percent,
      is_constant: info.is_constant,
      looks_like_id: checks.id_columns.columns.some((c) => c.column === name),
      top_values: info.top_values?.slice(0, 5),
      statistics: info.statistics,
    };
  });
}

// Docked AI chat about preprocessing. Asks for a suggestion every time the target
// changes (a target picked by mistake, or a second look at another column); the
// conversation history is kept. Each turn sends the history + the current plan.
export default function ChatPanel({ data, target, task, plan, results, asked }) {
  const [messages, setMessages] = useState([]); // {role:"user"|"assistant", content}
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const bodyRef = useRef(null);
  const inFlight = useRef(null);  // AbortController of the reply being streamed

  useEffect(() => {
    if (!data.id || !target) return;
    setMessages((m) => [...m, { role: "user", content: `Target: ${target}` }]);
    ask(
      `I selected "${target}" as the column to predict. First, is that a sensible choice for this dataset, ` +
      `or is another column a more natural target? Be honest, even if it disagrees with my choice. ` +
      `Then, which columns need cleaning or preprocessing, and why? ` +
      `Answer in a few short, simple sentences.`,
      true
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.id, target]);

  // a question sent from a button elsewhere on the page (e.g. "Ask AI" on a result row)
  useEffect(() => {
    if (asked?.text) ask(asked.text);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [asked?.n]);

  useEffect(() => {
    bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight });
  }, [messages, loading]);

  async function ask(text, seed = false) {
    const outgoing = seed ? [{ role: "user", content: text }] : [...messages, { role: "user", content: text }];
    if (!seed) setMessages((m) => [...m, { role: "user", content: text }]);
    inFlight.current?.abort();      // a newer question replaces a reply still streaming
    const ctrl = new AbortController();
    inFlight.current = ctrl;
    setLoading(true);
    setErr("");
    try {
      const res = await fetch(`${API}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: ctrl.signal,
        body: JSON.stringify({
          messages: outgoing.map(({ role, content }) => ({ role, content })),
          // live edited plan when it belongs to this target (so the AI remembers the
          // user's decisions), else the rule-based recommendation for this target.
          context: {
            ...(plan?.target === target ? plan : recommendedPlan(data, target, task)),
            rows: data.profile.rows,
            dataset_facts: columnFacts(data),
            results, // the training run on the page; the backend keeps only what the question needs
          },
        }),
      });
      if (!res.ok) {
        let d = `Error ${res.status}`;
        try {
          const j = await res.json();
          if (j.detail) d = j.detail;
        } catch {}
        throw new Error(d);
      }
      // Stream the reply as it arrives, growing the last assistant bubble.
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let acc = "";
      let started = false;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        acc += dec.decode(value, { stream: true });
        if (!started) {
          started = true;
          setLoading(false);
          setMessages((m) => [...m, { role: "assistant", content: acc }]);
        } else {
          setMessages((m) => {
            const c = m.slice();
            c[c.length - 1] = { role: "assistant", content: acc };
            return c;
          });
        }
      }
      if (!started) setMessages((m) => [...m, { role: "assistant", content: "(no response)" }]);
    } catch (e) {
      if (e.name === "AbortError") return;   // replaced by a newer question
      setErr(e instanceof TypeError ? `Cannot reach the backend at ${API}.` : e.message);
    } finally {
      if (inFlight.current === ctrl) setLoading(false);
    }
  }

  function submit() {
    const t = input.trim();
    if (t && !loading) {
      ask(t);
      setInput("");
    }
  }

  return (
    <div className="chat">
      <div className="chat-head">AI assistant — why &amp; which preprocessing</div>

      <div className="chat-body" ref={bodyRef}>
        {messages.length === 0 && !loading && !err && (
          <div className="chat-empty">Pick a target column above and I’ll suggest what to clean &amp; preprocess.</div>
        )}
        {messages.map((m, i) =>
          m.role === "assistant" ? (
            <div key={i} className="bubble assistant" dangerouslySetInnerHTML={{ __html: mdToHtml(m.content) }} />
          ) : (
            <div key={i} className="bubble user">{m.content}</div>
          )
        )}
        {loading && <div className="bubble assistant loading">…</div>}
        {err && <div className="chat-err">{err}</div>}
      </div>

      <div className="chat-input">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder="Ask why a step is chosen…  (Enter to send, Shift+Enter for a new line)"
          rows={2}
        />
        <button className="primary" onClick={submit} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}
