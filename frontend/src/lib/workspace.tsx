/**
 * Workspace state shared by the pages: which ticker is being analysed and with
 * what horizon and history window. Keeping it in one place means the selection
 * survives navigation between the dashboard, analysis and prediction pages.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

const STORAGE_KEY = "spps.workspace";

export interface WorkspaceState {
  ticker: string;
  horizon: number;
  historyDays: number;
}

const DEFAULTS: WorkspaceState = { ticker: "AAPL", horizon: 5, historyDays: 1095 };

interface WorkspaceContextValue extends WorkspaceState {
  setTicker: (ticker: string) => void;
  setHorizon: (horizon: number) => void;
  setHistoryDays: (days: number) => void;
}

const WorkspaceContext = createContext<WorkspaceContextValue | null>(null);

function readStored(): WorkspaceState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULTS;
    const parsed = JSON.parse(raw) as Partial<WorkspaceState>;
    return {
      ticker: typeof parsed.ticker === "string" && parsed.ticker ? parsed.ticker : DEFAULTS.ticker,
      horizon: Number(parsed.horizon) || DEFAULTS.horizon,
      historyDays: Number(parsed.historyDays) || DEFAULTS.historyDays,
    };
  } catch {
    return DEFAULTS;
  }
}

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<WorkspaceState>(readStored);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      // The selection simply will not be remembered next visit.
    }
  }, [state]);

  const setTicker = useCallback(
    (ticker: string) => setState((prev) => ({ ...prev, ticker: ticker.trim().toUpperCase() })),
    [],
  );
  const setHorizon = useCallback((horizon: number) => setState((prev) => ({ ...prev, horizon })), []);
  const setHistoryDays = useCallback((historyDays: number) => setState((prev) => ({ ...prev, historyDays })), []);

  const value = useMemo<WorkspaceContextValue>(
    () => ({ ...state, setTicker, setHorizon, setHistoryDays }),
    [state, setTicker, setHorizon, setHistoryDays],
  );

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>;
}

export function useWorkspace(): WorkspaceContextValue {
  const context = useContext(WorkspaceContext);
  if (!context) {
    throw new Error("useWorkspace must be used inside WorkspaceProvider.");
  }
  return context;
}

export const HORIZON_OPTIONS = [1, 5, 10, 30];
export const HISTORY_OPTIONS = [
  { label: "1 year", value: 365 },
  { label: "2 years", value: 730 },
  { label: "3 years", value: 1095 },
  { label: "5 years", value: 1825 },
];
