import { useEffect, useRef, useState } from "react";
import { sessionRows } from "../sessions";
import type { PanelProps } from "./types";

/** Header `[ sessions ]` toggle: a dropdown of the user's sessions, newest first; a row opens it. */
export function SessionsMenu({ sessionId, sessions, sessionsError, actions }: PanelProps) {
  const [shown, setShown] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!shown) return;
    const onDown = (e: MouseEvent) => ref.current?.contains(e.target as Node) || setShown(false);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setShown(false);
    addEventListener("mousedown", onDown);
    addEventListener("keydown", onKey);
    return () => {
      removeEventListener("mousedown", onDown);
      removeEventListener("keydown", onKey);
    };
  }, [shown]);
  const rows = sessions ? sessionRows(sessions, sessionId, new Date()) : [];
  return (
    <div className="sessions-menu" ref={ref}>
      <button
        aria-expanded={shown}
        onClick={() => {
          if (!shown) actions.refreshSessions();
          setShown(!shown);
        }}
      >
        sessions
      </button>
      {shown && (
        <div className="sessions">
          <div className="rule">
            sessions{sessions ? ` ${sessions.length}` : ""}
            <button onClick={actions.refreshSessions}>refresh</button>
          </div>
          {sessionsError && <div className="sessions-error">!! list failed: {sessionsError}</div>}
          {!sessions && !sessionsError && <div className="dim">loading…</div>}
          {sessions && !rows.length && <div className="dim">no sessions yet · the first message starts one</div>}
          <div className="pane">
            {rows.map((r) => (
              <button
                key={r.id}
                className={`srow ${r.empty ? "never-ran" : r.status.replace(/ /g, "_")}${r.current ? " current" : ""}`}
                title={r.id}
                onClick={() => {
                  setShown(false);
                  if (!r.current) actions.open(r.id);
                }}
              >
                <span className="cur">{r.current ? ">" : " "}</span>
                <span className="t">{r.time}</span>
                <span className="id">{r.short}</span>
                <span className="rst">{r.status}</span>
                <span className="locs">{r.locations.join(", ") || "–"}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
