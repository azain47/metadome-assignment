import { useEffect, useState } from "react";
import { imgUrl } from "../api";
import type { PanelProps } from "./types";

/** Everything the agent made (stills, composites, plates, guides, car renders), newest first, as
 * a grid that fills the stage and scrolls with the wheel. Thumbnails are a 1-bit ordered dither
 * (colour on hover/focus); a click opens the full image in a viewer (arrows browse, Esc closes).
 * Approval lives in the Approval panel. */
export function Gallery({ view }: PanelProps) {
  const [open, setOpen] = useState<number | null>(null);
  const items = [...(view.state.gallery || [])].reverse();
  const shown = open !== null ? items[open] : undefined;
  useEffect(() => {
    if (open === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(null);
      else if (e.key === "ArrowRight" || e.key === "ArrowDown") setOpen((i) => Math.min((i ?? 0) + 1, items.length - 1));
      else if (e.key === "ArrowLeft" || e.key === "ArrowUp") setOpen((i) => Math.max((i ?? 0) - 1, 0));
      else return;
      e.preventDefault();
      e.stopPropagation();
    };
    addEventListener("keydown", onKey, true);
    return () => removeEventListener("keydown", onKey, true);
  }, [open, items.length]);
  return (
    <section className="gallery">
      <div className="rule">
        <span>gallery{items.length ? ` (${items.length})` : ""}</span>
      </div>
      <div className="pane grid">
        {!items.length && <p className="dim">stills, plates, guides and car renders appear here as the agent makes them.</p>}
        {items.map((g, i) => (
          <figure key={g.id} className={`card ${g.kind}`}>
            <a
              className="dither"
              href={imgUrl(g.key)}
              title={`${g.kind}: view`}
              onClick={(e) => {
                e.preventDefault();
                setOpen(i);
              }}
            >
              <img src={imgUrl(g.key)} alt={g.kind} loading="lazy" />
            </a>
            <figcaption>
              <div>
                <span className="kind">{g.kind}</span>
                {g.verdict && (
                  <span className={`verdict ${g.verdict}`} title={g.reason || ""}>
                    [{g.verdict === "best_below_bar" ? "below bar" : g.verdict}]
                  </span>
                )}
              </div>
              <div className="dim">{[g.location, g.attempt ? `a${g.attempt}` : ""].filter(Boolean).join(" · ")}</div>
              <div className="scores">
                {Object.entries(g.scores || {}).slice(0, 3).map(([k, s]) => (
                  <span key={k} title={k}>
                    {k.slice(0, 8)} <b>{typeof s === "number" ? +s.toFixed(2) : s ? "+" : "x"}</b>
                  </span>
                ))}
              </div>
            </figcaption>
          </figure>
        ))}
      </div>
      {shown && (
        <div className="lightbox" onClick={() => setOpen(null)} role="dialog" aria-label={`${shown.kind} image`}>
          <img src={imgUrl(shown.key)} alt={shown.kind} onClick={(e) => e.stopPropagation()} />
          <div className="row lb-bar" onClick={(e) => e.stopPropagation()}>
            <span className="kind">{shown.kind}</span>
            {shown.verdict && <span className={`verdict ${shown.verdict}`}>[{shown.verdict}]</span>}
            <span className="grow dim">
              {[shown.location, shown.attempt ? `attempt ${shown.attempt}` : "", shown.reason].filter(Boolean).join(" · ")}
            </span>
            <span className="dim">
              {open! + 1}/{items.length} · arrows browse · esc closes
            </span>
            <a href={imgUrl(shown.key)} target="_blank" rel="noreferrer">
              open original
            </a>
            <button onClick={() => setOpen(null)}>close</button>
          </div>
        </div>
      )}
    </section>
  );
}
