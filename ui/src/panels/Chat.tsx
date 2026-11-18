import type { Json } from "../api";
import type { PanelProps } from "./types";
import { usePinToBottom } from "./usePinToBottom";

const CAR = String.raw`
                  ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
             ▄▄▀▀░░░░░░░░▐░░░░░░░░░░▀▀▄▄
     ▄▄▄▄▄▀▀▀░░░░░░░░░░░░▐░░░░░░░░░░░░░░▀▀▀▀▀▄▄▄▄▄
   ▐▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▌
   ▐▓▒▒▒▒▄▄▄▄▄▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▄▄▄▄▄▒▒▒▒▒▒▒▓▓▌
    ▀▀▀▀▐█▓▒▓█▌▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▐█▓▒▓█▌▀▀▀▀▀▀▀▀
         ▀▀▀▀▀                          ▀▀▀▀▀
  ─ ── ─── ──────────────────────────────────── ─── ── ─
`.slice(1, -1);

function JsonCard({ data }: { data: Json }) {
  const label = Array.isArray(data)
    ? `[${data.length} items]`
    : Object.entries(data as object).map(([k, v]) => (Array.isArray(v) ? `${k}[${v.length}]` : k)).join(", ");
  return (
    <details className="jsoncard">
      <summary>{"{ }"} {label}</summary>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </details>
  );
}

/** Chat messages; structured agent output renders as a collapsible JSON card. */
export function Chat({ view, sessionId }: PanelProps) {
  const ref = usePinToBottom<HTMLDivElement>();
  const empty = !sessionId && !view.msgs.length;
  return (
    <section className="chat">
      <div className="rule">
        <span>chat</span>
      </div>
      <div className="pane" ref={ref}>
        <div>
          {empty && (
            <div className="empty">
              <pre aria-hidden="true">{CAR}</pre>
              <p className="dim">describe a shot below, press [ new ], or open a session id to start.</p>
            </div>
          )}
          {view.msgs.map((m) => (
            <div key={m.id} className={`msg ${m.role}`}>
              <span className="author">{m.role === "user" ? ">" : m.role === "error" ? "!!" : `${m.author}:`}</span>
              {m.json !== undefined ? <JsonCard data={m.json} /> : <div className="text">{m.text}</div>}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
