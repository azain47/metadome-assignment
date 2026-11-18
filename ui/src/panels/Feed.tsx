import { feedGroups, type FeedItem } from "../events";
import type { PanelProps } from "./types";
import { usePinToBottom } from "./usePinToBottom";

function duration(it: FeedItem): string {
  if (it.status === "running") return "…";
  return it.durationMs != null ? `${(it.durationMs / 1000).toFixed(1)}s` : "";
}

const GLYPH = { ok: "+", error: "x", running: "~" } as const;

/** Pipeline steps (drawer tab): LLM tool calls and ts_step pipeline steps, grouped by graph node run. */
export function Feed({ view }: PanelProps) {
  const ref = usePinToBottom<HTMLDivElement>();
  return (
    <div className="pane feed" ref={ref}>
      <div>
        {!view.feed.length && <p className="dim">tool calls and pipeline steps stream here.</p>}
        {feedGroups(view.feed).map((g) => (
          <section key={g.key} className="group">
            <h4>
              :: {g.node}
              {/* @N counts runs of the node in this invocation: parallel workers and retries alike */}
              {g.attempt > 1 && <span className="dim"> · run {g.attempt}</span>}
            </h4>
            {g.items.map((it) => (
              <details key={it.id} className={`item ${it.status} ${it.kind}`}>
                <summary title={`${it.kind === "llm" ? "tool" : it.kind === "step" ? "step" : "human"} · click for args/result`}>
                  <span className="st">{it.kind === "approval" && it.status !== "error" ? "?" : GLYPH[it.status]}</span>
                  <b className="name">{it.name}</b>
                  <span className="args">{it.args ? JSON.stringify(it.args) : ""}</span>
                  <span className="dur">{duration(it)}</span>
                  {it.summary && <div className="sum">└ {it.summary}</div>}
                </summary>
                <pre>{JSON.stringify({ args: it.args, result: it.result ?? it.summary }, null, 2)}</pre>
              </details>
            ))}
          </section>
        ))}
      </div>
    </div>
  );
}
