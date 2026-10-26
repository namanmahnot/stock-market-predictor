/**
 * Typed HTTP client for the FastAPI backend.
 *
 * Requests go to a relative path so the Vite dev server proxies them to
 * http://127.0.0.1:8000; set VITE_API_BASE to point a build at another host.
 * No provider API key or secret is ever handled here.
 */
import axios, { AxiosError } from "axios";

import type {
  Backtest,
  BulkDeals,
  BulkDealSymbol,
  Comparison,
  History,
  Indicators,
  ModelInfo,
  Prediction,
  PredictionRun,
  StoredLeaderboard,
  Symbol,
  TokenResponse,
  WatchlistQuote,
} from "./types";

const TOKEN_KEY = "spps.token";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE ?? "/api/v1",
  timeout: 300_000, // model training can legitimately take minutes
});

export function readToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function storeToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Without storage the session simply does not survive a reload.
  }
}

api.interceptors.request.use((config) => {
  const token = readToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

/** Turn an Axios failure into a sentence a user can act on (SRS 14). */
export function describeError(error: unknown): string {
  if (!axios.isAxiosError(error)) {
    return error instanceof Error ? error.message : "Something went wrong.";
  }
  const axiosError = error as AxiosError<{ detail?: unknown }>;
  if (axiosError.code === "ECONNABORTED") {
    return "The request timed out. Training several models can take a few minutes - try a single model or a shorter history.";
  }
  if (!axiosError.response) {
    return "Could not reach the API. Start the backend with 'uvicorn app.main:app --reload' in the backend directory.";
  }
  const { status, data } = axiosError.response;
  const detail = data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors arrive as a list of field problems.
    return detail
      .map((item: { loc?: unknown[]; msg?: string }) => {
        const field = Array.isArray(item.loc) ? item.loc[item.loc.length - 1] : undefined;
        return field ? `${field}: ${item.msg}` : item.msg;
      })
      .filter(Boolean)
      .join("; ");
  }
  if (status === 429) return "Too many training requests. Wait a minute and try again.";
  if (status === 502) return "The market data provider did not return usable data for that symbol.";
  return `Request failed with status ${status}.`;
}

export const StocksApi = {
  search: (query: string) =>
    api.get<Symbol[]>("/stocks/search", { params: { q: query, limit: 10 } }).then((r) => r.data),
  history: (ticker: string, days: number) =>
    api.get<History>(`/stocks/${encodeURIComponent(ticker)}/history`, { params: { days } }).then((r) => r.data),
  indicators: (ticker: string, days: number) =>
    api
      .get<Indicators>(`/stocks/${encodeURIComponent(ticker)}/indicators`, { params: { days } })
      .then((r) => r.data),
};

export interface PredictionPayload {
  ticker: string;
  model: string;
  horizon: number;
  history_days: number;
  save?: boolean;
  include_explanation?: boolean;
  /** null lets the backend decide from this ticker's disclosure coverage. */
  use_bulk_deals?: boolean | null;
}

export const PredictionsApi = {
  create: (payload: PredictionPayload) =>
    api.post<Prediction>("/predictions", payload).then((r) => r.data),
  list: (limit = 25) =>
    api.get<PredictionRun[]>("/predictions", { params: { limit } }).then((r) => r.data),
  get: (id: number) => api.get<PredictionRun>(`/predictions/${id}`).then((r) => r.data),
};

export const BulkDealsApi = {
  forTicker: (ticker: string, limit = 100) =>
    api
      .get<BulkDeals>(`/bulk-deals/${encodeURIComponent(ticker)}`, { params: { limit } })
      .then((r) => r.data),
  symbols: (onlySufficient = true, limit = 25) =>
    api
      .get<BulkDealSymbol[]>("/bulk-deals/symbols", {
        params: { only_sufficient: onlySufficient, limit },
      })
      .then((r) => r.data),
};

export const ModelsApi = {
  catalogue: () => api.get<ModelInfo[]>("/models").then((r) => r.data),
  compare: (payload: {
    ticker: string;
    horizon: number;
    history_days: number;
    models: string[];
    use_bulk_deals?: boolean | null;
  }) =>
    api.post<Comparison>("/models/compare", payload).then((r) => r.data),
  leaderboard: (ticker?: string, segment: "validation" | "test" = "test") =>
    api
      .get<StoredLeaderboard>("/models/leaderboard", { params: { ticker, segment } })
      .then((r) => r.data),
};

export const WatchlistApi = {
  list: () => api.get<WatchlistQuote[]>("/watchlist").then((r) => r.data),
  add: (ticker: string, note?: string) =>
    api.post("/watchlist", { ticker, note: note || null }).then((r) => r.data),
  remove: (id: number) => api.delete(`/watchlist/${id}`).then(() => undefined),
};

export const BacktestApi = {
  run: (payload: {
    ticker: string;
    model: string;
    history_days: number;
    threshold_percent: number;
    transaction_cost_bps: number;
    use_bulk_deals?: boolean | null;
  }) => api.post<Backtest>("/backtest", payload).then((r) => r.data),
};

export const AuthApi = {
  register: (email: string, password: string) =>
    api.post<TokenResponse>("/auth/register", { email, password }).then((r) => r.data),
  login: (email: string, password: string) =>
    api.post<TokenResponse>("/auth/login", { email, password }).then((r) => r.data),
  me: () => api.get<TokenResponse["user"]>("/auth/me").then((r) => r.data),
};

/** Download a report through the browser, keeping the Authorization header. */
export async function downloadReport(runId: number, format: "csv" | "pdf"): Promise<void> {
  const response = await api.get(`/reports/${runId}`, { params: { format }, responseType: "blob" });
  triggerDownload(response.data as Blob, `run${runId}_report.${format}`);
}

export async function downloadComparison(payload: {
  ticker: string;
  horizon: number;
  history_days: number;
  models: string[];
}): Promise<void> {
  const response = await api.post("/reports/comparison", payload, { responseType: "blob" });
  triggerDownload(response.data as Blob, `${payload.ticker}_model_comparison.csv`);
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
