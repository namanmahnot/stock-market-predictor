/** Page 1: overview of the selected stock, the watchlist and recent runs. */
import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { PriceChart } from "../components/charts";
import { TickerSearch } from "../components/TickerSearch";
import {
  Badge,
  EmptyState,
  ErrorBanner,
  LoadingBlock,
  Panel,
  ProvenanceLine,
  StatCard,
  secondaryButtonClass,
} from "../components/ui";
import { PredictionsApi, StocksApi, WatchlistApi, describeError } from "../lib/api";
import { money, percent, shortDate } from "../lib/format";
import { useSession } from "../lib/session";
import type { History, PredictionRun, WatchlistQuote } from "../lib/types";
import { useWorkspace } from "../lib/workspace";

export function Dashboard() {
  const { ticker, setTicker } = useWorkspace();
  const { user, ready } = useSession();
  const navigate = useNavigate();

  const [history, setHistory] = useState<History | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [watchlist, setWatchlist] = useState<WatchlistQuote[]>([]);
  const [watchlistError, setWatchlistError] = useState<string | null>(null);
  const [runs, setRuns] = useState<PredictionRun[]>([]);

  const loadHistory = useCallback(() => {
    setLoading(true);
    setError(null);
    StocksApi.history(ticker, 365)
      .then(setHistory)
      .catch((cause) => setError(describeError(cause)))
      .finally(() => setLoading(false));
  }, [ticker]);

  useEffect(loadHistory, [loadHistory]);

  const loadWatchlist = useCallback(() => {
    if (!user) return;
    WatchlistApi.list()
      .then((rows) => {
        setWatchlist(rows);
        setWatchlistError(null);
      })
      .catch((cause) => setWatchlistError(describeError(cause)));
  }, [user]);

  useEffect(() => {
    if (!ready || !user) return;
    loadWatchlist();
    PredictionsApi.list(5).then(setRuns).catch(() => setRuns([]));
  }, [ready, user, loadWatchlist]);

  async function addCurrent() {
    try {
      await WatchlistApi.add(ticker);
      loadWatchlist();
    } catch (cause) {
      setWatchlistError(describeError(cause));
    }
  }

  async function remove(id: number) {
    try {
      await WatchlistApi.remove(id);
      loadWatchlist();
    } catch (cause) {
      setWatchlistError(describeError(cause));
    }
  }

  const up = (history?.change ?? 0) >= 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-ink">Dashboard</h1>
          <p className="text-sm text-ink-secondary">
            Latest prices for the stock you are analysing, your watchlist and your most recent forecasts.
          </p>
        </div>
        <div className="flex gap-2">
          <button type="button" className={secondaryButtonClass} onClick={addCurrent} disabled={!user}>
            + Add {ticker} to watchlist
          </button>
          <Link to="/predict" className={secondaryButtonClass}>
            Run a forecast →
          </Link>
        </div>
      </div>

      {error && <ErrorBanner message={error} onRetry={loadHistory} />}
      {loading && !history && <LoadingBlock label={`Loading ${ticker}`} />}

      {history && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              label="Last close"
              value={money(history.last_close, history.currency)}
              hint={`${history.ticker}${history.company_name ? ` · ${history.company_name}` : ""}`}
            />
            <StatCard
              label="Change (1 session)"
              value={percent(history.change_percent)}
              tone={up ? "up" : "down"}
              hint={
                <span>
                  <span aria-hidden="true">{up ? "▲" : "▼"}</span>{" "}
                  {money(history.change, history.currency)} on {shortDate(history.last_date)}
                </span>
              }
            />
            <StatCard
              label="Sessions available"
              value={history.n_rows}
              hint={`${shortDate(history.first_date)} → ${shortDate(history.last_date)}`}
            />
            <StatCard
              label="Data freshness"
              value={history.provenance.from_cache ? "Cached" : "Live fetch"}
              hint={<ProvenanceLine provenance={history.provenance} />}
            />
          </div>

          <Panel
            title={`${history.ticker} — last 12 months`}
            subtitle="Adjusted closing price. Open the analysis page for indicators and volume."
            actions={
              <Link to="/analysis" className={secondaryButtonClass}>
                Full analysis
              </Link>
            }
          >
            <PriceChart bars={history.bars} currency={history.currency} height={280} />
          </Panel>
        </>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel
          title="Watchlist"
          subtitle="Saved stocks with their latest available close."
          actions={<TickerSearch onSelect={(symbol) => void WatchlistApi.add(symbol).then(loadWatchlist)} placeholder="Add a stock…" />}
        >
          {watchlistError && <ErrorBanner message={watchlistError} />}
          {watchlist.length === 0 ? (
            <EmptyState title="Your watchlist is empty">
              Search for a stock above, or add the one you are currently analysing.
            </EmptyState>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase tracking-wide text-ink-muted">
                  <tr>
                    <th className="py-2 font-medium">Ticker</th>
                    <th className="py-2 font-medium">Last close</th>
                    <th className="py-2 font-medium">Change</th>
                    <th className="py-2 font-medium">As of</th>
                    <th className="py-2" />
                  </tr>
                </thead>
                <tbody>
                  {watchlist.map((row) => (
                    <tr key={row.id} className="border-t border-line">
                      <td className="py-2">
                        <button
                          type="button"
                          className="font-semibold text-accent hover:underline"
                          onClick={() => {
                            setTicker(row.ticker);
                            navigate("/analysis");
                          }}
                        >
                          {row.ticker}
                        </button>
                        {row.company_name && (
                          <span className="block text-xs text-ink-muted">{row.company_name}</span>
                        )}
                      </td>
                      <td className="py-2">{money(row.last_close)}</td>
                      <td
                        className={`py-2 ${
                          (row.change_percent ?? 0) >= 0
                            ? "text-[var(--color-up)]"
                            : "text-[var(--color-down)]"
                        }`}
                      >
                        {row.change_percent === null ? "-" : percent(row.change_percent)}
                      </td>
                      <td className="py-2 text-xs text-ink-secondary">
                        {row.error ? <span className="text-[var(--color-critical)]">{row.error}</span> : shortDate(row.last_date)}
                      </td>
                      <td className="py-2 text-right">
                        <button
                          type="button"
                          onClick={() => remove(row.id)}
                          aria-label={`Remove ${row.ticker} from the watchlist`}
                          className="rounded-md border border-line px-2 py-1 text-xs text-ink-secondary hover:border-[var(--color-critical)] hover:text-[var(--color-critical)]"
                        >
                          Remove
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>

        <Panel title="Recent forecasts" subtitle="The last runs saved to your prediction history.">
          {runs.length === 0 ? (
            <EmptyState title="No saved forecasts yet">
              Run one from the prediction lab; every run stores its model, horizon and measured metrics.
            </EmptyState>
          ) : (
            <ul className="space-y-2">
              {runs.map((run) => (
                <li
                  key={run.id}
                  className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-line p-3"
                >
                  <div>
                    <p className="text-sm font-semibold text-ink">
                      {run.ticker}{" "}
                      <span className="font-normal text-ink-secondary">· {run.model_name}</span>
                    </p>
                    <p className="text-xs text-ink-muted">
                      {run.forecast_horizon}-session horizon from {shortDate(run.reference_date)}
                      {run.auto_selected && " · auto-selected"}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    {run.forecasts[0] && (
                      <Badge tone={run.forecasts[0].change_percent >= 0 ? "good" : "critical"}>
                        {percent(run.forecasts[0].change_percent)} next session
                      </Badge>
                    )}
                    <Link to="/reports" className="text-xs font-medium text-accent hover:underline">
                      Open
                    </Link>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </div>
  );
}
