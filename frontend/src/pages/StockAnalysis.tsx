/** Page 2: historical price, volume and technical indicators (FR-02, FR-03, FR-04). */
import { useCallback, useEffect, useMemo, useState } from "react";

import { BulkDealsPanel } from "../components/BulkDealsPanel";
import { MacdChart, PriceChart, RsiChart, VolumeChart } from "../components/charts";
import {
  ErrorBanner,
  Field,
  InfoTip,
  LoadingBlock,
  Panel,
  ProvenanceLine,
  StatCard,
  selectClass,
} from "../components/ui";
import { StocksApi, describeError } from "../lib/api";
import { money, number, percent, shortDate } from "../lib/format";
import type { History, Indicators } from "../lib/types";
import { HISTORY_OPTIONS, useWorkspace } from "../lib/workspace";

const OVERLAYS = [
  { key: "sma_20", label: "SMA 20" },
  { key: "sma_50", label: "SMA 50" },
  { key: "ema_12", label: "EMA 12" },
  { key: "ema_26", label: "EMA 26" },
];

export function StockAnalysis() {
  const { ticker, historyDays, setHistoryDays, setTicker } = useWorkspace();
  const [history, setHistory] = useState<History | null>(null);
  const [indicators, setIndicators] = useState<Indicators | null>(null);
  const [active, setActive] = useState<string[]>(["sma_20", "sma_50"]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    Promise.all([StocksApi.history(ticker, historyDays), StocksApi.indicators(ticker, historyDays)])
      .then(([historyResult, indicatorResult]) => {
        setHistory(historyResult);
        setIndicators(indicatorResult);
      })
      .catch((cause) => setError(describeError(cause)))
      .finally(() => setLoading(false));
  }, [ticker, historyDays]);

  useEffect(load, [load]);

  const overlayData = useMemo(() => {
    if (!indicators) return [];
    return active.map((key) => ({
      key,
      label: OVERLAYS.find((item) => item.key === key)?.label ?? key,
      data: indicators.points.map((point) => ({ [key]: (point[key] as number | null) ?? null })),
    }));
  }, [indicators, active]);

  const latest = indicators?.points.at(-1);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-ink">
            Stock analysis — {ticker}
            {history?.company_name ? <span className="text-ink-secondary"> · {history.company_name}</span> : null}
          </h1>
          <p className="text-sm text-ink-secondary">
            Every indicator on a row uses only sessions up to that date, which is the same rule the forecasting
            models follow.
          </p>
        </div>
        <Field label="History window">
          <select
            className={selectClass}
            value={historyDays}
            onChange={(event) => setHistoryDays(Number(event.target.value))}
          >
            {HISTORY_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </Field>
      </div>

      {error && <ErrorBanner message={error} onRetry={load} />}
      {loading && <LoadingBlock label={`Loading ${ticker} history and indicators`} />}

      {history && indicators && !loading && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard label="Last close" value={money(history.last_close, history.currency)} hint={shortDate(history.last_date)} />
            <StatCard
              label="Session change"
              value={percent(history.change_percent)}
              tone={(history.change ?? 0) >= 0 ? "up" : "down"}
            />
            <StatCard
              label={"RSI (14)"}
              value={number(latest?.rsi_14 as number, 1)}
              hint={<span>{indicators.definitions.rsi_14}</span>}
            />
            <StatCard
              label="20-session volatility"
              value={number((latest?.volatility_20 as number) * 100, 2) + "%"}
              hint="Standard deviation of daily log returns, in percent."
            />
          </div>

          <Panel
            title="Price and moving averages"
            subtitle="Adjusted closing price with the selected overlays."
            actions={
              <div className="flex flex-wrap gap-2">
                {OVERLAYS.map((overlay) => {
                  const on = active.includes(overlay.key);
                  return (
                    <button
                      key={overlay.key}
                      type="button"
                      aria-pressed={on}
                      onClick={() =>
                        setActive((current) =>
                          on ? current.filter((key) => key !== overlay.key) : [...current, overlay.key],
                        )
                      }
                      className={`rounded-lg border px-2.5 py-1 text-xs font-medium ${
                        on ? "border-accent bg-accent-soft text-accent" : "border-line text-ink-secondary"
                      }`}
                    >
                      {overlay.label}
                    </button>
                  );
                })}
              </div>
            }
          >
            <PriceChart bars={history.bars} overlays={overlayData} currency={history.currency} />
            <div className="mt-3">
              <ProvenanceLine provenance={history.provenance} />
            </div>
          </Panel>

          <Panel title="Traded volume" subtitle="Shares traded per session, on its own axis.">
            <VolumeChart bars={history.bars} />
          </Panel>

          <div className="grid gap-6 lg:grid-cols-2">
            <Panel
              title="Relative Strength Index"
              subtitle={
                <span className="flex items-center">
                  Momentum over 14 sessions
                  <InfoTip text={indicators.definitions.rsi_14} />
                </span>
              }
            >
              <RsiChart points={indicators.points} />
            </Panel>

            <Panel
              title="MACD"
              subtitle={
                <span className="flex items-center">
                  Moving-average convergence and divergence
                  <InfoTip text={indicators.definitions.macd} />
                </span>
              }
            >
              <MacdChart points={indicators.points} />
            </Panel>
          </div>

          <BulkDealsPanel ticker={ticker} onSelectTicker={setTicker} />

          <Panel title="Indicator definitions" subtitle="What each series on this page measures.">
            <dl className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
              {Object.entries(indicators.definitions).map(([key, definition]) => (
                <div key={key}>
                  <dt className="text-xs font-semibold uppercase tracking-wide text-ink">{key}</dt>
                  <dd className="text-xs text-ink-secondary">{definition}</dd>
                </div>
              ))}
            </dl>
          </Panel>
        </>
      )}
    </div>
  );
}
