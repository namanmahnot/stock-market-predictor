/** Page 5: saved runs, report export, stored rankings and strategy backtesting. */
import { useCallback, useEffect, useState } from "react";

import { EquityCurveChart } from "../components/charts";
import {
  Badge,
  Disclaimer,
  EmptyState,
  ErrorBanner,
  Field,
  LoadingBlock,
  Panel,
  StatCard,
  buttonClass,
  secondaryButtonClass,
  selectClass,
} from "../components/ui";
import {
  BacktestApi,
  ModelsApi,
  PredictionsApi,
  describeError,
  downloadReport,
} from "../lib/api";
import { dateTime, number, percent, shortDate } from "../lib/format";
import { useSession } from "../lib/session";
import type { Backtest, ModelInfo, PredictionRun, StoredLeaderboard } from "../lib/types";
import { useWorkspace } from "../lib/workspace";

export function ReportsHistory() {
  const { ticker, historyDays } = useWorkspace();
  const { user } = useSession();

  const [runs, setRuns] = useState<PredictionRun[]>([]);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [stored, setStored] = useState<StoredLeaderboard | null>(null);

  const [models, setModels] = useState<ModelInfo[]>([]);
  const [backtestModel, setBacktestModel] = useState("xgboost");
  const [threshold, setThreshold] = useState(0);
  const [cost, setCost] = useState(10);
  const [backtest, setBacktest] = useState<Backtest | null>(null);
  const [backtesting, setBacktesting] = useState(false);
  const [backtestError, setBacktestError] = useState<string | null>(null);

  const loadRuns = useCallback(() => {
    if (!user) return;
    PredictionsApi.list(50)
      .then((rows) => {
        setRuns(rows);
        setRunsError(null);
      })
      .catch((cause) => setRunsError(describeError(cause)));
  }, [user]);

  useEffect(loadRuns, [loadRuns]);

  useEffect(() => {
    ModelsApi.catalogue().then(setModels).catch(() => setModels([]));
  }, []);

  useEffect(() => {
    ModelsApi.leaderboard(ticker).then(setStored).catch(() => setStored(null));
  }, [ticker]);

  const runBacktest = useCallback(async () => {
    setBacktesting(true);
    setBacktestError(null);
    try {
      const result = await BacktestApi.run({
        ticker,
        model: backtestModel,
        history_days: historyDays,
        threshold_percent: threshold,
        transaction_cost_bps: cost,
      });
      setBacktest(result);
    } catch (cause) {
      setBacktestError(describeError(cause));
    } finally {
      setBacktesting(false);
    }
  }, [ticker, backtestModel, historyDays, threshold, cost]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-ink">Reports and history</h1>
        <p className="text-sm text-ink-secondary">
          Saved prediction runs with their metadata, downloadable reports, stored model rankings and a
          historical strategy simulation.
        </p>
      </div>

      <Panel
        title="Prediction history"
        subtitle="Runs saved under your account, newest first."
        actions={
          <button type="button" className={secondaryButtonClass} onClick={loadRuns}>
            Refresh
          </button>
        }
      >
        {runsError && <ErrorBanner message={runsError} />}
        {runs.length === 0 ? (
          <EmptyState title="No saved runs yet">
            Forecasts you run in the prediction lab are stored here with their model, horizon, reference
            session and measured metrics.
          </EmptyState>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[48rem] text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-ink-muted">
                <tr>
                  <th className="py-2 font-medium">Run</th>
                  <th className="py-2 font-medium">Ticker</th>
                  <th className="py-2 font-medium">Model</th>
                  <th className="py-2 text-right font-medium">Horizon</th>
                  <th className="py-2 text-right font-medium">Next-session forecast</th>
                  <th className="py-2 text-right font-medium">Test MAE</th>
                  <th className="py-2 font-medium">Created</th>
                  <th className="py-2 font-medium">Report</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.id} className="border-t border-line">
                    <td className="py-2 text-ink-secondary">#{run.id}</td>
                    <td className="py-2 font-medium text-ink">{run.ticker}</td>
                    <td className="py-2">
                      {run.model_name}
                      {run.auto_selected && <Badge tone="accent">auto</Badge>}
                    </td>
                    <td className="py-2 text-right tabular-nums">{run.forecast_horizon}</td>
                    <td className="py-2 text-right tabular-nums">
                      {run.forecasts[0] ? percent(run.forecasts[0].change_percent) : "-"}
                    </td>
                    <td className="py-2 text-right tabular-nums">{number(run.metrics?.mae, 3)}</td>
                    <td className="py-2 text-xs text-ink-secondary">{dateTime(run.created_at)}</td>
                    <td className="py-2">
                      <div className="flex gap-2">
                        <button
                          type="button"
                          className="text-xs font-medium text-accent hover:underline"
                          onClick={() => downloadReport(run.id, "csv")}
                        >
                          CSV
                        </button>
                        <button
                          type="button"
                          className="text-xs font-medium text-accent hover:underline"
                          onClick={() => downloadReport(run.id, "pdf")}
                        >
                          PDF
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <Panel
        title={`Stored evaluations — ${ticker}`}
        subtitle={stored?.note ?? "Rankings measured in earlier comparison runs and saved to the database."}
      >
        {!stored || stored.rows.length === 0 ? (
          <EmptyState title="Nothing stored for this ticker yet">
            Run a comparison on the model comparison page; its evaluations are saved and ranked here.
          </EmptyState>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[40rem] text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-ink-muted">
                <tr>
                  <th className="py-2 font-medium">Model</th>
                  <th className="py-2 text-right font-medium">MAE</th>
                  <th className="py-2 text-right font-medium">RMSE</th>
                  <th className="py-2 text-right font-medium">MAE / naive</th>
                  <th className="py-2 font-medium">Evaluation period</th>
                  <th className="py-2 text-right font-medium">Sessions</th>
                </tr>
              </thead>
              <tbody>
                {stored.rows.map((row) => (
                  <tr key={`${row.model}-${row.evaluated_at}`} className="border-t border-line">
                    <td className="py-2 font-medium text-ink">{row.model}</td>
                    <td className="py-2 text-right tabular-nums">{number(row.mae, 3)}</td>
                    <td className="py-2 text-right tabular-nums">{number(row.rmse, 3)}</td>
                    <td className="py-2 text-right tabular-nums">{number(row.mae_vs_naive, 3)}</td>
                    <td className="py-2 text-xs text-ink-secondary">
                      {shortDate(row.period_start)} → {shortDate(row.period_end)}
                    </td>
                    <td className="py-2 text-right tabular-nums">{row.n_observations}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <Panel
        title={`Strategy backtest — ${ticker}`}
        subtitle="Go long the next session when the forecast exceeds the threshold, otherwise hold cash."
      >
        <div className="flex flex-wrap items-end gap-4">
          <Field label="Model">
            <select
              className={selectClass}
              value={backtestModel}
              onChange={(event) => setBacktestModel(event.target.value)}
            >
              {models.map((model) => (
                <option key={model.name} value={model.name} disabled={!model.available}>
                  {model.display_name}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Entry threshold (%)" hint="Predicted move needed to take the position.">
            <input
              type="number"
              step="0.05"
              min="-5"
              max="5"
              value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))}
              className={selectClass}
            />
          </Field>
          <Field label="Cost per trade (bps)" hint="One-way cost charged on every change of position.">
            <input
              type="number"
              step="1"
              min="0"
              max="200"
              value={cost}
              onChange={(event) => setCost(Number(event.target.value))}
              className={selectClass}
            />
          </Field>
          <button type="button" className={buttonClass} onClick={runBacktest} disabled={backtesting}>
            {backtesting ? "Simulating…" : "Run backtest"}
          </button>
        </div>

        {backtestError && <div className="mt-4"><ErrorBanner message={backtestError} /></div>}
        {backtesting && (
          <div className="mt-4">
            <LoadingBlock
              label="Walking the test window one session at a time"
              note="The model is refitted periodically on the expanding window, so this is slower than a single forecast."
            />
          </div>
        )}

        {backtest && !backtesting && (
          <div className="mt-4 space-y-4">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <StatCard
                label="Strategy return"
                value={percent(backtest.strategy.total_return_percent)}
                tone={backtest.strategy.total_return_percent >= 0 ? "up" : "down"}
                hint={`${backtest.n_sessions} sessions, ${backtest.n_trades} position changes`}
              />
              <StatCard
                label="Buy and hold"
                value={percent(backtest.benchmark.total_return_percent)}
                tone={backtest.benchmark.total_return_percent >= 0 ? "up" : "down"}
                hint="Passive benchmark over the same window"
              />
              <StatCard
                label="Max drawdown"
                value={percent(backtest.strategy.max_drawdown_percent)}
                hint={`Benchmark ${percent(backtest.benchmark.max_drawdown_percent)}`}
              />
              <StatCard
                label="Time in market"
                value={`${backtest.time_in_market_percent.toFixed(1)}%`}
                hint={`Sharpe ${number(backtest.strategy.sharpe_ratio, 2)}`}
              />
            </div>

            <EquityCurveChart points={backtest.equity_curve} />

            <ul className="list-inside list-disc space-y-1 text-xs text-ink-secondary">
              {backtest.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
            <Disclaimer text="A backtest measures one decision rule on one historical window. It is not a measure of forecast accuracy, and it is not a trading recommendation." />
          </div>
        )}
      </Panel>
    </div>
  );
}
