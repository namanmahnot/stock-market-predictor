/** Page 4: leaderboard and actual-versus-predicted comparison (FR-06, SRS 10.2). */
import { useCallback, useEffect, useState } from "react";

import { ActualVsPredictedChart, SkillBarChart } from "../components/charts";
import {
  Badge,
  BaselineVerdict,
  Disclaimer,
  EmptyState,
  ErrorBanner,
  Field,
  InfoTip,
  LoadingBlock,
  Panel,
  ProvenanceLine,
  buttonClass,
  secondaryButtonClass,
  selectClass,
} from "../components/ui";
import { ModelsApi, describeError, downloadComparison } from "../lib/api";
import { number, share, shortDate } from "../lib/format";
import type { Comparison, ModelInfo } from "../lib/types";
import { HORIZON_OPTIONS, useWorkspace } from "../lib/workspace";
import { METRIC_HELP } from "./PredictionLab";

export function ModelComparison() {
  const { ticker, horizon, historyDays, setHorizon } = useWorkspace();
  const [catalogue, setCatalogue] = useState<ModelInfo[]>([]);
  const [chosen, setChosen] = useState<string[]>([]);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [bulkDeals, setBulkDeals] = useState<"auto" | "on" | "off">("auto");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    ModelsApi.catalogue()
      .then((models) => {
        setCatalogue(models);
        setChosen(models.filter((model) => model.in_default_set).map((model) => model.name));
      })
      .catch((cause) => setError(describeError(cause)));
  }, []);

  const run = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const result = await ModelsApi.compare({
        ticker,
        horizon,
        history_days: historyDays,
        models: chosen,
        use_bulk_deals: bulkDeals === "auto" ? null : bulkDeals === "on",
      });
      setComparison(result);
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setRunning(false);
    }
  }, [ticker, horizon, historyDays, chosen, bulkDeals]);

  const curves =
    comparison?.models
      .filter((model) => model.test?.curve?.length)
      .map((model) => ({
        model: model.model,
        label: model.display_name,
        curve: model.test!.curve,
      })) ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-ink">Model comparison — {ticker}</h1>
        <p className="text-sm text-ink-secondary">
          Every selected model is evaluated on identical splits, identical forecast origins and the same
          prediction target, so the numbers are comparable.
        </p>
      </div>

      <Panel title="Comparison settings">
        <div className="flex flex-wrap items-end gap-4">
          <Field label="Horizon">
            <select className={selectClass} value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}>
              {HORIZON_OPTIONS.map((value) => (
                <option key={value} value={value}>
                  {value} session{value > 1 ? "s" : ""}
                </option>
              ))}
            </select>
          </Field>
          <Field label="NSE bulk deals" hint="Disclosed large trades as extra features.">
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
          <div>
            <p className="mb-1 text-xs font-medium text-ink-secondary">Models</p>
            <div className="flex flex-wrap gap-2">
              {catalogue.map((model) => {
                const on = chosen.includes(model.name);
                return (
                  <button
                    key={model.name}
                    type="button"
                    aria-pressed={on}
                    disabled={!model.available}
                    title={model.description}
                    onClick={() =>
                      setChosen((current) =>
                        on ? current.filter((name) => name !== model.name) : [...current, model.name],
                      )
                    }
                    className={`rounded-lg border px-2.5 py-1 text-xs font-medium disabled:opacity-40 ${
                      on ? "border-accent bg-accent-soft text-accent" : "border-line text-ink-secondary"
                    }`}
                  >
                    {model.display_name}
                  </button>
                );
              })}
            </div>
          </div>
          <button type="button" className={buttonClass} onClick={run} disabled={running || chosen.length === 0}>
            {running ? "Evaluating…" : "Run comparison"}
          </button>
          {comparison && (
            <button
              type="button"
              className={secondaryButtonClass}
              onClick={() => downloadComparison({ ticker, horizon, history_days: historyDays, models: chosen })}
            >
              Export leaderboard CSV
            </button>
          )}
        </div>
      </Panel>

      {error && <ErrorBanner message={error} onRetry={run} />}
      {running && (
        <LoadingBlock
          label="Training and walking every selected model"
          note="Each model is refitted at its forecast origins, so a full six-model comparison takes a few minutes the first time."
        />
      )}

      {!comparison && !running && (
        <EmptyState title="No comparison yet">
          Choose the models you want to compare and run the evaluation. The naive baseline is always included,
          because a model only demonstrates value by beating it.
        </EmptyState>
      )}

      {comparison && !running && (
        <>
          <Panel
            title="Leaderboard"
            subtitle={comparison.selection_rule}
            actions={
              <div className="flex flex-wrap items-center gap-2">
                {comparison.feature_set && (
                  <Badge tone={comparison.feature_set === "price_only" ? "neutral" : "accent"}>
                    {comparison.feature_set === "price_only"
                      ? "price features only"
                      : "price + NSE bulk deals"}
                  </Badge>
                )}
                {comparison.best_model && (
                  <Badge tone="accent">Selected: {comparison.best_model}</Badge>
                )}
              </div>
            }
          >
            <div className="overflow-x-auto">
              <table className="w-full min-w-[56rem] text-left text-sm">
                <thead className="text-xs uppercase tracking-wide text-ink-muted">
                  <tr>
                    <th className="py-2 font-medium">Model</th>
                    <th className="py-2 font-medium">Family</th>
                    <th className="py-2 text-right font-medium">
                      Val MAE<InfoTip text="Error on the validation window, which is what selects the model." />
                    </th>
                    <th className="py-2 text-right font-medium">
                      Test MAE<InfoTip text={METRIC_HELP.mae} />
                    </th>
                    <th className="py-2 text-right font-medium">
                      Test RMSE<InfoTip text={METRIC_HELP.rmse} />
                    </th>
                    <th className="py-2 text-right font-medium">
                      MAPE<InfoTip text={METRIC_HELP.mape} />
                    </th>
                    <th className="py-2 text-right font-medium">
                      Dir. acc.<InfoTip text={METRIC_HELP.directional_accuracy} />
                    </th>
                    <th className="py-2 text-right font-medium">
                      R²<InfoTip text={METRIC_HELP.r2} />
                    </th>
                    <th className="py-2 text-right font-medium">Fit (s)</th>
                    <th className="py-2 font-medium">
                      Versus baseline<InfoTip text={METRIC_HELP.skill} />
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.leaderboard.map((row) => (
                    <tr key={row.model} className={`border-t border-line ${row.is_best ? "bg-accent-soft" : ""}`}>
                      <td className="py-2 font-medium text-ink">
                        {row.display_name}
                        {row.is_best && <span className="ml-2 text-xs text-accent">selected</span>}
                      </td>
                      <td className="py-2 text-xs text-ink-secondary">{row.family?.replace("_", " ")}</td>
                      {row.error ? (
                        <td colSpan={8} className="py-2 text-xs text-[var(--color-critical)]">
                          {row.error}
                        </td>
                      ) : (
                        <>
                          <td className="py-2 text-right tabular-nums">{number(row.validation_mae, 3)}</td>
                          <td className="py-2 text-right tabular-nums">{number(row.test_mae, 3)}</td>
                          <td className="py-2 text-right tabular-nums">{number(row.test_rmse, 3)}</td>
                          <td className="py-2 text-right tabular-nums">
                            {row.test_mape == null ? "-" : `${row.test_mape.toFixed(2)}%`}
                          </td>
                          <td className="py-2 text-right tabular-nums">{share(row.test_directional_accuracy)}</td>
                          <td className="py-2 text-right tabular-nums">{number(row.test_r2, 3)}</td>
                          <td className="py-2 text-right tabular-nums">{number(row.training_seconds, 2)}</td>
                          <td className="py-2">
                            <BaselineVerdict beats={row.beats_naive_on_test} ratio={row.mae_vs_naive_test} />
                          </td>
                        </>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {(comparison.dataset.notes?.length ?? 0) > 0 && (
              <ul className="mt-3 list-inside list-disc space-y-1 text-xs text-ink-secondary">
                {comparison.dataset.notes?.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            )}
            <p className="mt-3 text-xs text-ink-secondary">
              Evaluated on {comparison.dataset.n_test} test sessions from{" "}
              {shortDate(comparison.dataset.test_start)} to {shortDate(comparison.dataset.test_end)}; the model
              was selected on {comparison.dataset.n_validation} earlier validation sessions. Total runtime{" "}
              {comparison.runtime_seconds.toFixed(1)}s.
            </p>
          </Panel>

          <Panel
            title="Skill against the naive baseline"
            subtitle="Test MAE divided by the naive forecast's MAE. Bars left of the dashed line added value."
          >
            <SkillBarChart rows={comparison.leaderboard} />
          </Panel>

          <Panel
            title="Actual versus predicted"
            subtitle="One-session-ahead forecasts at every evaluated origin in the held-out test window."
          >
            {curves.length > 0 ? (
              <ActualVsPredictedChart series={curves} />
            ) : (
              <EmptyState title="No evaluation curve available" />
            )}
            <div className="mt-3 space-y-3">
              <ProvenanceLine provenance={comparison.provenance} />
              <Disclaimer text={comparison.disclaimer} />
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}
