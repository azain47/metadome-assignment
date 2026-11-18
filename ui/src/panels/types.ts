import type { Session } from "../api";
import type { Decision, ViewState } from "../events";
import type { RunSettings } from "../settings";

/** The pop-out drawer over the gallery: the pipeline steps or long-term memory (null = closed). */
export type DrawerTab = "steps" | "memory";

/** What App gives every panel. A panel reads `view` and calls `actions`; it never parses events or does HTTP. */
export interface PanelProps {
  view: ViewState;
  sessionId: string;
  streaming: boolean;
  notice: string;
  /** The user's sessions from the last list fetch (null before the first one lands). */
  sessions: Session[] | null;
  /** Last list fetch error, "" when it succeeded. */
  sessionsError: string;
  /** Header choices, sent with every run (apply from the next message). */
  settings: RunSettings;
  drawer: DrawerTab | null;
  /** A run started elsewhere (another tab, or before a reload) is progressing; this window polls it. */
  watching: boolean;
  actions: {
    send(text: string): void;
    /** Answer the pending approval (RequestInput pause). */
    answer(decision: Decision, note?: string): void;
    stop(): void;
    resume(): void;
    /** Blank, sessionless view; the next message creates the server session. */
    newSession(): void;
    open(id: string): void;
    setSettings(settings: RunSettings): void;
    refreshSessions(): void;
    /** Open a drawer tab, or close it (null, or the open tab again). */
    toggleDrawer(tab: DrawerTab | null): void;
  };
}
