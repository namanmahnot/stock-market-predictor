/**
 * NSE bulk-deal disclosures for the selected stock (FR-15).
 *
 * A bulk deal is one client trading more than 0.5% of a company's equity in a
 * single session, disclosed by the exchange after the close. The panel is as
 * much about what is *not* there: large caps essentially never appear, and the
 * coverage line says so rather than showing an empty table.
 */
import { useCallback, useEffect, useState } from "react";
import {
  Bar,
  CartesianGrid,
  Cell,
  ComposedChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { BulkDealsApi, describeError } from "../lib/api";
import { axisDate, compactNumber, money, shortDate } from "../lib/format";
import { useChartTheme } from "../lib/theme";
import type { BulkDeals, BulkDealSymbol } from "../lib/types";
import { Badge, EmptyState, ErrorBanner, LoadingBlock, Panel, Spinner } from "./ui";

export function BulkDealsPanel({
  ticker,
  onSelectTicker,
}: {
  ticker: string;
  onSelectTicker?: (ticker: string) => void;
}) {
  const theme = useChartTheme();
  const [data, setData] = useState<BulkDeals | null>(null);
  const [suggestions, setSuggestions] = useState<BulkDealSymbol[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    BulkDealsApi.forTicker(ticker, 60)
      .then(setData)
      .catch((cause) => setError(describeError(cause)))
      .finally(() => setLoading(false));
  }, [ticker]);

  useEffect(load, [load]);

  useEffect(() => {
    BulkDealsApi.symbols(true, 12).then(setSuggestions).catch(() => setSuggestions([]));
  }, []);

  if (loading) {
    return (
      <Panel title="NSE bulk deals">
        <LoadingBlock label="Loading disclosures" />
      </Panel>
    );
  }

  if (error) {
    return (
      <Panel title="NSE bulk deals">
        <ErrorBanner message={error} onRetry={load} />
      </Panel>
    );
  }

  if (!data) return null;

  const { coverage } = data;
  // Oldest first for the chart; the table stays newest first.
  const chartData = [...data.daily].reverse().map((row) => ({
    date: row.date,
    net: row.net_quantity,
    turnover: row.gross_turnover,
  }));

  return (
    <Panel
      title="NSE bulk deals"
      subtitle={
        <>
          Single clients trading more than 0.5% of the company's equity in one session, as
          disclosed by the exchange after the close. Dataset covers{" "}
          {shortDate(coverage.dataset_start)} to {shortDate(coverage.dataset_end)}.
        </>
      }
      actions={
        coverage.in_dataset ? (
          <Badge tone={coverage.sufficient ? "good" : "warning"}>
            {coverage.sessions_with_deals} session{coverage.sessions_with_deals === 1 ? "" : "s"} ·{" "}
            {coverage.deal_rows} deals
          </Badge>
        ) : (
          <Badge>no disclosures</Badge>
        )
      }
    >
      {data.notes.length > 0 && (
        <ul className="mb-4 list-inside list-disc space-y-1 text-xs text-ink-secondary">
          {data.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}

      {!coverage.in_dataset ? (
        <EmptyState title={`No bulk deals disclosed for ${ticker}`}>
          <p>
            This is the normal case for a large stock. Symbols with enough disclosures to analyse
            include:
          </p>
          <div className="mt-2 flex flex-wrap justify-center gap-1.5">
            {suggestions.slice(0, 8).map((entry) => (
              <button
                key={entry.ticker}
                type="button"
                onClick={() => onSelectTicker?.(entry.ticker)}
                disabled={!onSelectTicker}
                title={`${entry.security_name} — ${entry.sessions_with_deals} sessions`}
                className="rounded-full border border-line px-2 py-0.5 text-[11px] font-medium text-ink-secondary hover:border-accent hover:text-accent disabled:cursor-default"
              >
                {entry.ticker}
              </button>
            ))}
            {suggestions.length === 0 && <Spinner label="Loading symbols" />}
          </div>
        </EmptyState>
      ) : (
        <>
          <div style={{ width: "100%", height: 220 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
                <CartesianGrid stroke={theme.grid} strokeDasharray="2 4" vertical={false} />
                <XAxis
                  dataKey="date"
                  tickFormatter={axisDate}
                  tick={{ fill: theme.inkMuted, fontSize: 11 }}
                  stroke={theme.axis}
                  minTickGap={40}
                />
                <YAxis
                  tick={{ fill: theme.inkMuted, fontSize: 11 }}
                  stroke={theme.axis}
                  tickFormatter={compactNumber}
                  width={62}
                />
                <ReferenceLine y={0} stroke={theme.axis} />
                <Tooltip
                  cursor={{ fill: theme.grid, fillOpacity: 0.35 }}
                  content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null;
                    const value = Number(payload[0].value);
                    return (
                      <div className="rounded-lg border border-line bg-surface-raised px-3 py-2 text-xs shadow-lg">
                        <p className="font-semibold text-ink">{shortDate(String(label))}</p>
                        <p className="text-ink-secondary">
                          Net {value >= 0 ? "bought" : "sold"}:{" "}
                          <span className="font-medium text-ink">
                            {compactNumber(Math.abs(value))} shares
                          </span>
                        </p>
                      </div>
                    );
                  }}
                />
                {/* Sign is the message, so the diverging pair carries it and the
                    zero line is the neutral midpoint. */}
                <Bar dataKey="net" name="Net disclosed quantity" radius={[2, 2, 0, 0]}>
                  {chartData.map((row) => (
                    <Cell
                      key={row.date}
                      fill={row.net >= 0 ? theme.series[0] : theme.series[7]}
                    />
                  ))}
                </Bar>
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 text-xs text-ink-muted">
            Bars above the line are sessions where disclosed buying exceeded disclosed selling.
          </p>

          <div className="mt-4 max-h-80 overflow-auto rounded-lg border border-line">
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-surface text-ink-secondary">
                <tr>
                  <th className="px-2 py-1.5 font-medium">Date</th>
                  <th className="px-2 py-1.5 font-medium">Client</th>
                  <th className="px-2 py-1.5 font-medium">Side</th>
                  <th className="px-2 py-1.5 text-right font-medium">Quantity</th>
                  <th className="px-2 py-1.5 text-right font-medium">Price</th>
                  <th className="px-2 py-1.5 text-right font-medium">Value</th>
                </tr>
              </thead>
              <tbody>
                {data.deals.map((deal, index) => (
                  <tr key={`${deal.deal_date}-${deal.client_name}-${index}`} className="border-t border-line">
                    <td className="px-2 py-1.5 whitespace-nowrap">{shortDate(deal.deal_date)}</td>
                    <td className="px-2 py-1.5">{deal.client_name}</td>
                    <td className="px-2 py-1.5">
                      <span
                        className={
                          deal.side === "BUY"
                            ? "font-medium text-[var(--color-up)]"
                            : "font-medium text-[var(--color-down)]"
                        }
                      >
                        {/* Word plus arrow: never colour alone. */}
                        <span aria-hidden="true">{deal.side === "BUY" ? "▲ " : "▼ "}</span>
                        {deal.side}
                      </span>
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums">
                      {deal.quantity.toLocaleString()}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums">{money(deal.price, "INR")}</td>
                    <td className="px-2 py-1.5 text-right tabular-nums">
                      {money(deal.turnover, "INR", 0)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-ink-muted">
            Source: {data.sources.join(", ")}. Disclosures are published after the close of the
            session they describe.
          </p>
        </>
      )}
    </Panel>
  );
}
