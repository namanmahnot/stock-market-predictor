/** Page 3: run a forecast, see its interval, its measured error and its explanation. */
import { useCallback, useEffect, useState } from "react";

import { ContributionChart, ForecastChart } from "../components/charts";
import {
  Badge,
  BaselineVerdict,
  Disclaimer,
  ErrorBanner,
  Field,
  InfoTip,
  LoadingBlock,
  Panel,
  ProvenanceLine,
  StatCard,
  buttonClass,
  secondaryButtonClass,
  selectClass,
} from "../components/ui";
import { ModelsApi, PredictionsApi, StocksApi, describeError, downloadReport } from "../lib/api";
import { money, number, percent, share, shortDate } from "../lib/format";
import type { History, ModelInfo, Prediction } from "../lib/types";
import { HISTORY_OPTIONS, HORIZON_OPTIONS, useWorkspace } from "../lib/workspace";

export const METRIC_HELP: Record<string, string> = {
  mae: "Mean absolute error: the average size of the forecast error, in price units. Lower is better.",
  rmse: "Root mean squared error: like MAE but large misses count for more. Lower is better.",
  mape: "Mean absolute percentage error. Easy to compare across stocks, but unstable when prices are near zero.",
  directional_accuracy:
    "Share of forecasts that predicted the right direction of the move. Around 0.5 is what an uninformative forecast produces on daily data.",
  r2: "Fraction of the variance of the test window explained by the forecast, relative to that window's mean. Can be negative.",
  skill:
    "Model MAE divided by the naive forecast's MAE on the same window. Below 1.0 means the model added predictive value.",
};

export function PredictionLab() {
  const { ticker, horizon, historyDays, setHorizon, setHistoryDays } = useWorkspace();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selected, setSelected] = useState("auto");
  const [explain, setExplain] = useState(true);
  const [bulkDeals, setBulkDeals] = useState<"auto" | "on" | "off">("auto");
  const [history, setHistory] = useState<History | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    ModelsApi.catalogue().then(setModels).catch(() => setModels([]));
  }, []);

  useEffect(() => {
    StocksApi.history(ticker, Math.max(historyDays, 365))
      .then(setHistory)
      .catch(() => setHistory(null));
  }, [ticker, historyDays]);

  const run = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const result = await PredictionsApi.create({
        ticker,
        model: selected,
        horizon,
        history_days: historyDays,
        save: true,
        include_explanation: explain,
        use_bulk_deals: bulkDeals === "auto" ? null : bulkDeals === "on",
      });
      setPrediction(result);
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setRunning(false);
    }
  }, [ticker, selected, horizon, historyDays, explain, bulkDeals]);

  const chosenModel = models.find((model) => model.name === (prediction?.model ?? selected));
  const first = prediction?.forecasts[0];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-ink">AI prediction lab — {ticker}</h1>
        <p className="text-sm text-ink-secondary">
          Pick an algorithm and a horizon. Every forecast is reported with the error it achieved on data it was
          never trained on, next to the naive baseline.
        </p>
      </div>

      <Panel title="Forecast settings">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-6">
          <Field label="Model">
            <select className={selectClass} value={selected} onChange={(event) => setSelected(event.target.value)}>
              <option value="auto">Auto-select (lowest validation MAE)</option>
              {models.map((model) => (
                <option key={model.name} value={model.name} disabled={!model.available}>
                  {model.display_name}
                  {model.in_default_set ? "" : " (optional)"}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Horizon (trading sessions)">
            <select className={selectClass} value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}>
              {HORIZON_OPTIONS.map((value) => (
                <option key={value} value={value}>
                  {value} session{value > 1 ? "s" : ""}
                </option>
              ))}
            </select>
          </Field>
          <Field label="History used">
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
          <Field
            label="NSE bulk deals"
            hint="Disclosed large trades as extra features."
          >
            <select
              className={selectClass}
              value={bulkDeals}
              onChange={(event) => setBulkDeals(event.target.value as "auto" | "on" | "off")}
            >
              <option value="auto">Auto (use when disclosed)</option>
              <option value="on">Always include</option>
              <option value="off">Never include</option>
            </select>
          </Field>
          <Field label="Explainability">
            <label className="flex items-center gap-2 rounded-lg border border-line bg-surface px-3 py-2 text-sm font-normal text-ink">
              <input type="checkbox" checked={explain} onChange={(event) => setExplain(event.target.checked)} />
              SHAP contributions
            </label>
          </Field>
          <div className="flex items-end">
            <button type="button" className={buttonClass} onClick={run} disabled={running}>
              {running ? "Training…" : "Run forecast"}
            </button>
          </div>
        </div>
        {chosenModel && (
          <p className="mt-3 text-xs text-ink-secondary">
            <span className="font-semibold text-ink">{chosenModel.display_name}:</span> {chosenModel.description}
          </p>
        )}
      </Panel>

      {error && <ErrorBanner message={error} onRetry={run} />}
      {running && (
        <LoadingBlock
          label="Fitting models and walking the evaluation windows"
          note="Auto-select trains the whole default model set, including the LSTM, so the first run on a new stock can take a few minutes."
        />
      )}

      {prediction && !running && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              label="Reference close"
              value={money(prediction.reference_price, prediction.currency)}
              hint={`Session of ${shortDate(prediction.reference_date)}`}
            />
            <StatCard
              label="Next session forecast"
              value={money(first?.predicted_price, prediction.currency)}
              tone={(first?.change ?? 0) >= 0 ? "up" : "down"}
              hint={
                first
                  ? `${percent(first.change_percent)} · target ${shortDate(first.target_date)}`
                  : undefined
              }
            />
            <StatCard
              label="95% interval"
              value={
                first?.lower_bound == null
                  ? "not available"
                  : `${money(first.lower_bound, prediction.currency)} – ${money(first.upper_bound, prediction.currency)}`
              }
              hint="Empirical quantiles of the validation residuals."
            />
            <StatCard
              label="Model"
              value={prediction.model_display_name}
              hint={
                prediction.auto_selected
                  ? `Auto-selected. ${prediction.selection_rule ?? ""}`
                  : "Chosen manually."
              }
            />
          </div>

          <Panel
            title={`${prediction.forecast_horizon}-session forecast`}
            subtitle="Everything right of the dashed line is predicted, not observed."
            actions={
              prediction.run_id ? (
                <div className="flex gap-2">
                  <button
                    type="button"
                    className={secondaryButtonClass}
                    onClick={() => downloadReport(prediction.run_id!, "csv")}
                  >
                    Export CSV
                  </button>
                  <button
                    type="button"
                    className={secondaryButtonClass}
                    onClick={() => downloadReport(prediction.run_id!, "pdf")}
                  >
                    Export PDF
                  </button>
                </div>
              ) : null
            }
          >
            {history && (
              <ForecastChart
                bars={history.bars}
                forecasts={prediction.forecasts}
                referenceDate={prediction.reference_date}
                referencePrice={prediction.reference_price}
                currency={prediction.currency}
              />
            )}
            <div className="mt-3 space-y-3">
              <ProvenanceLine provenance={prediction.provenance} />
              <Disclaimer text={prediction.disclaimer} />
            </div>
          </Panel>

          {prediction.feature_set && (
            <Panel
              title="Feature set"
              subtitle="Which inputs produced this forecast."
              actions={
                <Badge tone={prediction.feature_set === "price_only" ? "neutral" : "accent"}>
                  {prediction.feature_set === "price_only"
                    ? "price features only"
                    : "price + NSE bulk deals"}
                </Badge>
              }
            >
              <p className="text-sm text-ink-secondary">
                {prediction.dataset.n_features ?? 24} features over{" "}
                {prediction.dataset.feature_rows} sessions.
              </p>
              {prediction.bulk_deals && (
                <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-ink-secondary sm:grid-cols-4">
                  <Detail
                    label="Sessions with deals"
                    value={`${prediction.bulk_deals.coverage.sessions_with_deals}`}
                  />
                  <Detail
                    label="Disclosed rows"
                    value={`${prediction.bulk_deals.coverage.deal_rows}`}
                  />
                  <Detail
                    label="Window kept"
                    value={`${prediction.bulk_deals.rows_after_clipping} of ${prediction.bulk_deals.rows_before_clipping}`}
                  />
                  <Detail
                    label="Publication lag"
                    value={`${prediction.bulk_deals.publication_lag_sessions} session(s)`}
                  />
                </dl>
              )}
              {(prediction.dataset.notes?.length ?? 0) > 0 && (
                <ul className="mt-3 list-inside list-disc space-y-1 text-xs text-ink-secondary">
                  {prediction.dataset.notes?.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              )}
            </Panel>
          )}

          <div className="grid gap-6 lg:grid-cols-2">
            <Panel
              title="Measured performance"
              subtitle={prediction.metrics_source ?? undefined}
            >
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase tracking-wide text-ink-muted">
                  <tr>
                    <th className="py-2 font-medium">Metric</th>
                    <th className="py-2 text-right font-medium">This model</th>
                    <th className="py-2 text-right font-medium">Naive baseline</th>
                  </tr>
                </thead>
                <tbody className="text-ink">
                  <MetricRow
                    label="MAE"
                    help={METRIC_HELP.mae}
                    model={number(prediction.metrics?.mae, 3)}
                    baseline={number(prediction.baseline_metrics?.mae, 3)}
                  />
                  <MetricRow
                    label="RMSE"
                    help={METRIC_HELP.rmse}
                    model={number(prediction.metrics?.rmse, 3)}
                    baseline={number(prediction.baseline_metrics?.rmse, 3)}
                  />
                  <MetricRow
                    label="MAPE"
                    help={METRIC_HELP.mape}
                    model={prediction.metrics?.mape == null ? "-" : `${prediction.metrics.mape.toFixed(2)}%`}
                    baseline={
                      prediction.baseline_metrics?.mape == null
                        ? "-"
                        : `${prediction.baseline_metrics.mape.toFixed(2)}%`
                    }
                  />
                  <MetricRow
                    label="Directional accuracy"
                    help={METRIC_HELP.directional_accuracy}
                    model={share(prediction.metrics?.directional_accuracy)}
                    baseline={share(prediction.baseline_metrics?.directional_accuracy)}
                  />
                  <MetricRow
                    label="R²"
                    help={METRIC_HELP.r2}
                    model={number(prediction.metrics?.r2, 3)}
                    baseline={number(prediction.baseline_metrics?.r2, 3)}
                  />
                </tbody>
              </table>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <span className="text-xs text-ink-secondary">
                  Skill against the baseline
                  <InfoTip text={METRIC_HELP.skill} />
                </span>
                <BaselineVerdict
                  beats={prediction.beats_naive_on_test}
                  ratio={prediction.mae_vs_naive_test}
                />
              </div>
              <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-ink-secondary sm:grid-cols-3">
                <Detail label="Train" value={`${prediction.dataset.n_train} sessions`} />
                <Detail label="Validation" value={`${prediction.dataset.n_validation} sessions`} />
                <Detail label="Test" value={`${prediction.dataset.n_test} sessions`} />
                <Detail
                  label="Test window"
                  value={`${shortDate(prediction.dataset.test_start)} → ${shortDate(prediction.dataset.test_end)}`}
                />
                <Detail label="Runtime" value={`${prediction.runtime_seconds?.toFixed(1) ?? "-"} s`} />
                <Detail label="Run id" value={prediction.run_id ? `#${prediction.run_id}` : "not saved"} />
              </dl>
            </Panel>

            <Panel
              title="Why this prediction"
              subtitle="SHAP contributions to the next-session log return."
            >
              {prediction.explanation?.contributions?.length ? (
                <>
                  <ContributionChart contributions={prediction.explanation.contributions} />
                  <p className="mt-2 text-xs text-ink-secondary">{prediction.explanation.note}</p>
                </>
              ) : (
                <p className="text-sm text-ink-secondary">
                  {prediction.explanation?.unavailable ??
                    "Explanations are available for the tree-based models (Random Forest and XGBoost). Enable the checkbox and pick one of those models."}
                </p>
              )}
            </Panel>
          </div>

          <Panel title="Limitations" subtitle="What this forecast does not and cannot do.">
            <ul className="list-inside list-disc space-y-1 text-sm text-ink-secondary">
              {prediction.limitations.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </Panel>
        </>
      )}
    </div>
  );
}

function MetricRow({
  label,
  help,
  model,
  baseline,
}: {
  label: string;
  help: string;
  model: string;
  baseline: string;
}) {
  return (
    <tr className="border-t border-line">
      <td className="py-2 text-ink-secondary">
        {label}
        <InfoTip text={help} />
      </td>
      <td className="py-2 text-right font-medium tabular-nums">{model}</td>
      <td className="py-2 text-right tabular-nums text-ink-secondary">{baseline}</td>
    </tr>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="font-medium text-ink-muted">{label}</dt>
      <dd className="text-ink-secondary">{value}</dd>
    </div>
  );
}
