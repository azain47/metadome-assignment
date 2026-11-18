import { useCallback, useEffect, useRef, useState, type ComponentType } from "react";
import { createSession, getSession, listSessions, refusal, runSse, type Content, type Session } from "./api";
import { addLocalMessage, answerMessage, emptyView, fromSession, reduce, textMessage, type ViewState } from "./events";
import { Approval } from "./panels/Approval";
import { Chat } from "./panels/Chat";
import { Composer } from "./panels/Composer";
import { Drawer } from "./panels/Drawer";
import { Gallery } from "./panels/Gallery";
import { Header } from "./panels/Header";
import { StageBar } from "./panels/StageBar";
import { Status } from "./panels/Status";
import type { DrawerTab, PanelProps } from "./panels/types";
import { DEFAULT_SETTINGS, SETTINGS_KEY, settingsFrom, type RunSettings } from "./settings";

// Regions: top bars, the stage (left: what needs you, then the images) and the side (right: the
// conversation). The drawer pops out over the stage. Adding or removing a panel is one line here.
const TOP: ComponentType<PanelProps>[] = [Header, Status];
const STAGE: ComponentType<PanelProps>[] = [StageBar, Approval, Gallery];
const SIDE: ComponentType<PanelProps>[] = [Chat, Composer];

const urlSession = () => new URLSearchParams(location.search).get("s") || "";

export default function App() {
  const [sessionId, setSessionId] = useState(urlSession);
  const [view, setView] = useState<ViewState>(emptyView);
  const [streaming, setStreaming] = useState(false);
  const [notice, setNotice] = useState("");
  const [settings, setSettings] = useState<RunSettings>(DEFAULT_SETTINGS);
  const [sessions, setSessions] = useState<Session[] | null>(null);
  const [sessionsError, setSessionsError] = useState("");
  const [drawer, setDrawer] = useState<DrawerTab | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // A failed fetch keeps the last good list so the dropdown still works offline.
  const refreshSessions = useCallback(() => {
    listSessions().then(
      (list) => {
        setSessions(list);
        setSessionsError("");
      },
      (e) => setSessionsError((e as Error).message),
    );
  }, []);
  useEffect(refreshSessions, [refreshSessions]);

  // `adopt`: take the session's stored settings (opening one); false keeps choices made before it existed.
  // Latest call wins: a slow fetch must not overwrite a later open (or a later `[ new ]`).
  const openSeq = useRef(0);
  const open = useCallback(async (id: string, push = true, adopt = true) => {
    const seq = ++openSeq.current;
    setNotice("");
    if (!id) {
      setView(emptyView);
      setSessionId("");
      return;
    }
    try {
      const s = await getSession(id);
      if (seq !== openSeq.current) return;
      setView(fromSession(s));
      if (adopt) setSettings(settingsFrom(s.state));
      setSessionId(s.id);
      if (push && urlSession() !== s.id) history.pushState(null, "", `?s=${encodeURIComponent(s.id)}`);
    } catch (e) {
      if (seq === openSeq.current) setNotice(`open ${id}: ${(e as Error).message}`);
    }
  }, []);

  useEffect(() => {
    open(urlSession(), false);
    const onPop = () => open(urlSession(), false);
    addEventListener("popstate", onPop);
    return () => removeEventListener("popstate", onPop);
  }, [open]);

  // Watching a session this window is not streaming (a run started in another tab, or this tab before
  // a reload, or an approval that another window may answer): poll it so progress shows and a card
  // answered elsewhere goes away. A run counts as live while new events keep arriving; one that stays
  // silent for a minute is interrupted, and only then is it offered for resume.
  const runStatus = view.state.run?.status;
  const [lastChange, setLastChange] = useState(0);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!sessionId || streaming || (runStatus !== "running" && runStatus !== "awaiting_approval")) return;
    let seen = -1;
    let alive = true;
    const t = setInterval(() => {
      setNow(Date.now());
      getSession(sessionId).then((s) => {
        if (!alive || s.events.length === seen) return;
        if (seen >= 0) setLastChange(Date.now());
        seen = s.events.length;
        setView(fromSession(s));
      }, () => {});
    }, 4000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [sessionId, streaming, runStatus]);
  const watching = !streaming && runStatus === "running" && now - lastChange < 60_000;

  // The only place a server session is created: on the first run from a blank view. Keeps the
  // settings chosen before it existed, and lists it at once.
  const createForRun = async (): Promise<string | undefined> => {
    try {
      const s = await createSession();
      await open(s.id, true, false);
      refreshSessions();
      return s.id;
    } catch (e) {
      setNotice((e as Error).message);
    }
  };

  // Never reload mid-run: ADK aborts the run when the stream disconnects (that is also how Stop works).
  // `echo` shows the user's input at once (applied after a session exists, so opening one can't wipe it).
  const run = async (input: { newMessage: Content } | { invocationId: string }, echo?: (v: ViewState) => ViewState) => {
    if (streaming) return;
    const id = sessionId || (await createForRun());
    if (!id) return;
    setNotice("");
    if (echo) setView(echo);
    const ac = new AbortController();
    abortRef.current = ac;
    setStreaming(true);
    try {
      await runSse(id, input, { [SETTINGS_KEY]: settings }, (ev) => setView((v) => reduce(v, ev)), ac.signal);
    } catch (e) {
      const msg = (e as Error).message;
      if (msg.startsWith("409 ")) {
        // The server refused it (another window is running this session, or the approval was
        // already answered): reload the session, since the echo already resolved the card here.
        open(id, false, false);
        setNotice(refusal(msg));
      } else {
        setNotice(ac.signal.aborted ? "Stopped: the server aborts the run on disconnect." : `run failed: ${msg}`);
      }
    } finally {
      abortRef.current = null;
      setStreaming(false);
      // Sync state (user:/app: keys, final run status) without dropping what was streamed.
      getSession(id).then((s) => setView((v) => ({ ...v, state: s.state })), () => {});
      refreshSessions();
    }
  };

  const props: PanelProps = {
    view,
    sessionId,
    streaming,
    notice,
    settings,
    sessions,
    sessionsError,
    drawer,
    watching,
    actions: {
      send: (text) => run({ newMessage: textMessage(text) }, (v) => addLocalMessage(v, text)),
      answer: (decision, note) => {
        if (!view.pending) return;
        const newMessage = answerMessage(view.pending.id, decision, note);
        // Resolves the card at once; the reducer dedupes a server echo by interrupt id.
        run({ newMessage }, (v) => reduce(v, { author: "user", content: newMessage }));
      },
      stop: () => abortRef.current?.abort(),
      resume: () => view.lastInvocationId && run({ invocationId: view.lastInvocationId }),
      // Blank, sessionless view; no server call (the first message creates the session). Settings stay.
      newSession: () => {
        if (streaming) return;
        open("");
        if (urlSession()) history.pushState(null, "", location.pathname);
      },
      setSettings,
      open: (id) => open(id),
      refreshSessions,
      toggleDrawer: (tab) => setDrawer((d) => (tab === null || d === tab ? null : tab)),
    },
  };
  const region = (panels: ComponentType<PanelProps>[]) => panels.map((P, i) => <P key={i} {...props} />);
  return (
    <div className="app">
      {region(TOP)}
      <main className="main">
        <div className="stage">
          {region(STAGE)}
          {drawer && <Drawer {...props} />}
        </div>
        <div className="side">{region(SIDE)}</div>
      </main>
    </div>
  );
}
