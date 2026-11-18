import { memoryState } from "./Memory";
import type { PanelProps } from "./types";

/** Top-left of the stage: opens the drawer, which pops out right here over the gallery (its own
 * tab row lands on top of this one). */
export function StageBar({ view, drawer, actions }: PanelProps) {
  return (
    <div className="row stagebar">
      <button className={drawer === "steps" ? "on" : ""} onClick={() => actions.toggleDrawer("steps")}
              title="pipeline steps and tool calls (Esc closes)">
        steps {view.feed.length}
        {view.feed.some((it) => it.status === "running") ? " ~" : ""}
      </button>
      <button className={drawer === "memory" ? "on" : ""} onClick={() => actions.toggleDrawer("memory")}
              title="long-term memory (Esc closes)">
        memory {Object.keys(memoryState(view.state)).length}
      </button>
    </div>
  );
}
