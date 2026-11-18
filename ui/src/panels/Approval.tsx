import { useState } from "react";
import { imgUrl } from "../api";
import type { PanelProps } from "./types";

/** Human-approval card for a pending RequestInput pause: the Blender guide (before any image is
 * made) or the final image, with scores, approve / revise (note) / reject. */
export function Approval({ view, streaming, actions }: PanelProps) {
  const [note, setNote] = useState("");
  const p = view.pending;
  if (!p) return null;
  const { payload } = p;
  const guide = payload.stage === "guide";
  const answer = (decision: "approve" | "revise" | "reject") => {
    actions.answer(decision, note.trim());
    setNote("");
  };
  return (
    <section className="approval">
      <div className="rule">
        <span>{guide ? "!! guide approval needed (no image made yet)" : "!! approval needed"}</span>
      </div>
      <div className="card-body">
        {payload.final_key && (
          <a className="frame" href={imgUrl(payload.final_key)} target="_blank" rel="noreferrer" title="open full image">
            <img src={imgUrl(payload.final_key)} alt={guide ? "Blender guide awaiting approval" : "final shot awaiting approval"} loading="eager" />
          </a>
        )}
        <div className="body">
          <div className="title">
            <b>{payload.job_id || "shot"}</b>
            {payload.verdict && (
              <span className={`verdict ${payload.verdict}`}>[{payload.verdict === "best_below_bar" ? "below bar" : payload.verdict}]</span>
            )}
            {payload.below_threshold && <span className="verdict fail">[below threshold]</span>}
          </div>
          <div className="dim">{[payload.location, payload.attempt ? `attempt ${payload.attempt}` : ""].filter(Boolean).join(" · ")}</div>
          <div className="msgtext">{p.message}</div>
          {payload.reason && <div className="reason">!! {payload.reason}</div>}
          <div className="scores">
            {Object.entries(payload.scores || {}).map(([k, s]) => (
              <span key={k}>
                {k} <b>{typeof s === "number" ? +s.toFixed(3) : s ? "+" : "x"}</b>
              </span>
            ))}
          </div>
          <label className="note">
            <span className="dim">note&gt;</span>
            <input value={note} onChange={(e) => setNote(e.target.value)} placeholder={guide ? "required for revise: camera, road, set…" : "required for revise"} disabled={streaming} />
          </label>
          <div className="row actions">
            <button className="primary" disabled={streaming} onClick={() => answer("approve")}>approve</button>
            <button disabled={streaming || !note.trim()} onClick={() => answer("revise")}>revise</button>
            <button className="danger" disabled={streaming} onClick={() => answer("reject")}>reject</button>
          </div>
        </div>
      </div>
    </section>
  );
}
