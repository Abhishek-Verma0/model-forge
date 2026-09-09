"use client";

import { useEffect, useRef, useState } from "react";

import { recommendedPlan } from "./preprocess";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

// Docked AI chat about preprocessing. Seeds one suggestion from the target on
// first load, then keeps the whole conversation -- history is NOT cleared when
// the target or tab changes. Each turn sends the history + the current plan.
export default function ChatPanel({ data, target, task, plan }) {
  const [messages, setMessages] = useState([]); // {role:"user"|"assistant", content}
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const bodyRef = useRef(null);
  const seeded = useRef(false);

  // Seed once, when the user FIRST picks a target (not auto). History is kept
  // after -- changing the target later never wipes the conversation.
  useEffect(() => {
    if (!data.id || !target || seeded.current) return;
    seeded.current = true;
    ask(
      `I want to predict "${target}". Which columns need cleaning or preprocessing, and why? Answer in a few short, simple sentences.`,
      true
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.id, target]);

  useEffect(() => {
    bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight });
  }, [messages, loading]);

  async function ask(text, seed = false) {
    const outgoing = seed ? [{ role: "user", content: text }] : [...messages, { role: "user", content: text }];
    if (!seed) setMessages((m) => [...m, { role: "user", content: text }]);
    setLoading(true);
    setErr("");
    try {
      const res = await fetch(`${API}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: outgoing.map(({ role, content }) => ({ role, content })),
          // live edited plan when available (so the AI remembers the user's
          // decisions), else the rule-based recommendation.
          context: plan || recommendedPlan(data, target, task),
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
      setErr(e instanceof TypeError ? `Cannot reach the backend at ${API}.` : e.message);
    } finally {
      setLoading(false);
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
