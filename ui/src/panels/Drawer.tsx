import { useEffect } from "react";
import { Feed } from "./Feed";
import { Memory } from "./Memory";
import type { DrawerTab, PanelProps } from "./types";

const TABS: DrawerTab[] = ["steps", "memory"];

/** Pop-out over the stage: the pipeline steps (every tool call, grouped by graph node) or
 * long-term memory. The chat stays visible beside it; Esc or [ close ] hides it. */
export function Drawer(props: PanelProps) {
  const { drawer, actions } = props;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") actions.toggleDrawer(null);
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [actions]);
  return (
    <aside className="drawer" aria-label={drawer ?? undefined}>
      <div className="row tabs">
        {TABS.map((t) => (
          <button key={t} className={drawer === t ? "on" : ""} onClick={() => actions.toggleDrawer(t)}>
            {t}
          </button>
        ))}
        <span className="grow" />
        <button onClick={() => actions.toggleDrawer(null)} title="Esc">
          close
        </button>
      </div>
      {drawer === "steps" ? <Feed {...props} /> : <Memory {...props} />}
    </aside>
  );
}
