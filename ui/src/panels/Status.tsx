import { useEffect, useState } from "react";
import { canResume } from "../events";
import type { PanelProps } from "./types";

const money = (n?: number) => (typeof n === "number" ? `$${n.toFixed(2)}` : "–");
const SPIN = "|/-\\";
const BAR = 20;

/** Run status from state.run, spend from state.spend, Resume after an interrupted run, notices. */
export function Status({ view, streaming, watching, notice, actions }: PanelProps) {
  const { run, spend } = view.state;
  const status = run?.status || "idle";
  const [tick, setTick] = useState(0);
  const live = streaming || watching;
  useEffect(() => {
    if (!live) return;
    const t = setInterval(() => setTick((n) => n + 1), 120);
    return () => clearInterval(t);
  }, [live]);
  const bits = [
    run?.node && `node ${run.node}`,
    run?.job_id && `job ${run.job_id}`,
    run?.location,
    run?.attempt ? `attempt ${run.attempt}` : "",
  ].filter(Boolean);
  const used = spend?.cap_usd ? Math.min(BAR, Math.round(((spend.total_usd || 0) / spend.cap_usd) * BAR)) : 0;
  return (
    <>
      <div className={`status ${status}${live ? " live" : ""}`}>
        <span className="state" title={streaming ? "stream connected" : watching ? "watching a run started elsewhere" : "no stream"}>
          {">> "}
          {status.replace(/_/g, " ")}
          {live && <span className="spin"> {SPIN[tick % SPIN.length]}</span>}
          {watching && <span className="dim"> watching</span>}
        </span>
        <span className="grow dim">{bits.length ? ` · ${bits.join(" · ")}` : ""}</span>
        {canResume(view, live) && (
          <>
            <span className="resume-hint" title={run?.error || ""}>
              {status === "failed" ? `last run failed${run?.error ? `: ${run.error}` : ""}` : "last run interrupted"}
              {" · resume re-runs it from where it stopped"}
            </span>
            <button className="stop" onClick={actions.resume}>
              resume
            </button>
          </>
        )}
        {spend && (
          <span className="spend" title={`llm ${money(spend.llm_usd)} · image ${money(spend.image_usd)}`}>
            {money(spend.total_usd)} / {money(spend.cap_usd)}{" "}
            <span className="bar">
              [<b>{"█".repeat(used)}</b>
              {"░".repeat(BAR - used)}]
            </span>
          </span>
        )}
      </div>
      {notice && <div className="notice">!! {notice}</div>}
    </>
  );
}
