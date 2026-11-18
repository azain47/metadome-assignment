import { useState } from "react";
import type { PanelProps } from "./types";

/** Message box with send, or stop while a run is streaming (input disabled meanwhile).
 * While an approval is pending, the text is sent as a revise note to that pause, not as a new message. */
export function Composer({ view, streaming, actions }: PanelProps) {
  const [text, setText] = useState("");
  const pending = view.pending;
  const send = () => {
    const t = text.trim();
    if (!t || streaming) return;
    if (pending) actions.answer("revise", t);
    else actions.send(t);
    setText("");
  };
  const placeholder = streaming
    ? "running…"
    : pending
      ? `revision note for ${pending.payload.job_id || "this shot"} (sends revise)`
      : "describe the shot · enter to send · shift+enter for newline";
  return (
    <div className="composer">
      <span className="prompt" aria-hidden="true">
        {">"}
      </span>
      <textarea
        value={text}
        disabled={streaming}
        rows={2}
        placeholder={placeholder}
        aria-label="message"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            send();
          }
        }}
      />
      {streaming ? (
        <button className="stop" onClick={actions.stop}>stop</button>
      ) : (
        <button className="primary" onClick={send}>{pending ? "revise" : "send"}</button>
      )}
    </div>
  );
}
