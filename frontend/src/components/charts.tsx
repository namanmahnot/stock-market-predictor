/**
 * Recharts wrappers.
 *
 * Conventions that apply to every chart here: one measure per vertical axis
 * (volume, RSI and MACD get their own panel rather than a second y-axis),
 * recessive grid and axes, 2px lines with no per-point dots, a crosshair
 * tooltip, a legend whenever two or more series are drawn, and a data-table
 * view so identity is never carried by colour alone.
 */
import { useId, useState, type ReactNode } from "react";
import {
  Area,
  Bar,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { axisDate, compactNumber, money, number, shortDate } from "../lib/format";
import { useChartTheme } from "../lib/theme";
import type { CurvePoint, ForecastStep, LeaderboardRow, PriceBar } from "../lib/types";

interface TooltipEntry {
  name?: string;
  value?: number | string;
  color?: string;
  dataKey?: string | number;
}

function ChartTooltip({
  active,
  label,
  payload,
  formatter,
}: {
  active?: boolean;
  label?: string | number;
  payload?: TooltipEntry[];
  formatter?: (value: number, key: string) => string;
}) {
  const theme = useChartTheme();
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div
      className="rounded-lg border border-line bg-surface-raised px-3 py-2 text-xs shadow-lg"
      style={{ color: theme.ink }}
    >
      <p className="mb-1 font-semibold">{shortDate(String(label))}</p>
      <ul className="space-y-0.5">
        {payload
          .filter((entry) => entry.value !== null && entry.value !== undefined)
          .map((entry) => (
            <li key={String(entry.dataKey)} className="flex items-center gap-2">
              <span
                aria-hidden="true"
                className="inline-block h-2 w-2 rounded-full"
                style={{ background: entry.color }}
              />
              <span className="text-ink-secondary">{entry.name}</span>
              <span className="ml-auto font-medium tabular-nums">
                {typeof entry.value === "number"
                  ? (formatter?.(entry.value, String(entry.dataKey)) ?? number(entry.value, 2))
                  : entry.value}
              </span>
            </li>
          ))}
      </ul>
    </div>
  );
}

/** Wrapper that adds the optional table view required when colour contrast is low. */
function ChartFrame({
  height = 300,
  table,
  children,
}: {
  height?: number;
  table?: ReactNode;
  children: ReactNode;
}) {
  const [showTable, setShowTable] = useState(false);
  const id = useId();
  return (
    <div>
      <div style={{ width: "100%", height }}>{children}</div>
      {table && (
        <div className="mt-2">
          <button
            type="button"
            aria-expanded={showTable}
            aria-controls={id}
            onClick={() => setShowTable((value) => !value)}
            className="text-xs font-medium text-accent underline-offset-2 hover:underline"
          >
            {showTable ? "Hide data table" : "Show data table"}
          </button>
          {showTable && (
            <div id={id} className="mt-2 max-h-64 overflow-auto rounded-lg border border-line">
              {table}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const AXIS_FONT = 11;

export function PriceChart({
  bars,
  overlays = [],
  currency,
  height = 320,
}: {
  bars: PriceBar[];
  overlays?: { key: string; label: string; data: Record<string, number | null>[] }[];
  currency?: string | null;
  height?: number;
}) {
  const theme = useChartTheme();
  const merged = bars.map((bar, index) => {
    const row: Record<string, number | string | null> = { date: bar.date, close: bar.adj_close };
    overlays.forEach((overlay) => {
      row[overlay.key] = (overlay.data[index]?.[overlay.key] as number | null) ?? null;
    });
    return row;
  });

  const hasOverlays = overlays.length > 0;

  return (
    <ChartFrame height={height}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={merged} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
          <CartesianGrid stroke={theme.grid} strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={axisDate}
            tick={{ fill: theme.inkMuted, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            minTickGap={40}
          />
          <YAxis
            domain={["auto", "auto"]}
            tick={{ fill: theme.inkMuted, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            tickFormatter={(value: number) => number(value, 0)}
            width={60}
          />
          <Tooltip
            content={<ChartTooltip formatter={(value) => money(value, currency)} />}
            cursor={{ stroke: theme.axis, strokeDasharray: "3 3" }}
          />
          {hasOverlays && <Legend wrapperStyle={{ fontSize: 11, color: theme.ink }} />}
          <Line
            type="monotone"
            dataKey="close"
            name="Adjusted close"
            stroke={theme.series[0]}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 2, stroke: theme.surface }}
          />
          {overlays.map((overlay, index) => (
            <Line
              key={overlay.key}
              type="monotone"
              dataKey={overlay.key}
              name={overlay.label}
              stroke={theme.series[(index + 1) % theme.series.length]}
              strokeWidth={2}
              dot={false}
              connectNulls
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function VolumeChart({ bars, height = 140 }: { bars: PriceBar[]; height?: number }) {
  const theme = useChartTheme();
  const data = bars.map((bar) => ({ date: bar.date, volume: bar.volume ?? 0 }));
  return (
    <ChartFrame height={height}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 4, right: 12, bottom: 4, left: 4 }}>
          <CartesianGrid stroke={theme.grid} strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={axisDate}
            tick={{ fill: theme.inkMuted, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            minTickGap={40}
          />
          <YAxis
            tick={{ fill: theme.inkMuted, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            tickFormatter={compactNumber}
            width={60}
          />
          <Tooltip
            content={<ChartTooltip formatter={(value) => compactNumber(value)} />}
            cursor={{ fill: theme.grid, fillOpacity: 0.4 }}
          />
          <Bar dataKey="volume" name="Volume" fill={theme.series[0]} fillOpacity={0.55} radius={[2, 2, 0, 0]} />
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function RsiChart({ points, height = 170 }: { points: Record<string, unknown>[]; height?: number }) {
  const theme = useChartTheme();
  const data = points.map((point) => ({ date: point.date as string, rsi: point.rsi_14 as number | null }));
  return (
    <ChartFrame height={height}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
          <CartesianGrid stroke={theme.grid} strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={axisDate}
            tick={{ fill: theme.inkMuted, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            minTickGap={40}
          />
          <YAxis
            domain={[0, 100]}
            ticks={[0, 30, 50, 70, 100]}
            tick={{ fill: theme.inkMuted, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            width={40}
          />
          {/* 30 and 70 are the conventional oversold and overbought lines. */}
          <ReferenceArea y1={30} y2={70} fill={theme.grid} fillOpacity={0.35} />
          <ReferenceLine y={70} stroke={theme.axis} strokeDasharray="4 4" />
          <ReferenceLine y={30} stroke={theme.axis} strokeDasharray="4 4" />
          <Tooltip content={<ChartTooltip formatter={(value) => number(value, 1)} />} />
          <Line
            type="monotone"
            dataKey="rsi"
            name="RSI (14)"
            stroke={theme.series[6]}
            strokeWidth={2}
            dot={false}
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function MacdChart({ points, height = 180 }: { points: Record<string, unknown>[]; height?: number }) {
  const theme = useChartTheme();
  const data = points.map((point) => ({
    date: point.date as string,
    macd: point.macd as number | null,
    signal: point.macd_signal as number | null,
    histogram: point.macd_hist as number | null,
  }));
  return (
    <ChartFrame height={height}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
          <CartesianGrid stroke={theme.grid} strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={axisDate}
            tick={{ fill: theme.inkMuted, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            minTickGap={40}
          />
          <YAxis
            tick={{ fill: theme.inkMuted, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            tickFormatter={(value: number) => number(value, 1)}
            width={48}
          />
          <ReferenceLine y={0} stroke={theme.axis} />
          <Tooltip content={<ChartTooltip formatter={(value) => number(value, 3)} />} />
          <Legend wrapperStyle={{ fontSize: 11, color: theme.ink }} />
          <Bar dataKey="histogram" name="Histogram" fill={theme.series[2]} fillOpacity={0.5} radius={[2, 2, 0, 0]} />
          <Line type="monotone" dataKey="macd" name="MACD" stroke={theme.series[0]} strokeWidth={2} dot={false} connectNulls />
          <Line
            type="monotone"
            dataKey="signal"
            name="Signal"
            stroke={theme.series[1]}
            strokeWidth={2}
            dot={false}
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function ForecastChart({
  bars,
  forecasts,
  referenceDate,
  referencePrice,
  currency,
  height = 340,
}: {
  bars: PriceBar[];
  forecasts: ForecastStep[];
  referenceDate: string;
  referencePrice: number;
  currency?: string | null;
  height?: number;
}) {
  const theme = useChartTheme();

  const historyTail = bars.slice(-90).map((bar) => ({
    date: bar.date,
    actual: bar.adj_close,
    forecast: bar.date === referenceDate ? referencePrice : null,
    bandBase: null as number | null,
    bandSpan: null as number | null,
  }));

  const forecastRows = forecasts.map((step) => ({
    date: step.target_date,
    actual: null as number | null,
    forecast: step.predicted_price,
    bandBase: step.lower_bound,
    bandSpan:
      step.lower_bound !== null && step.upper_bound !== null ? step.upper_bound - step.lower_bound : null,
  }));

  const data = [...historyTail, ...forecastRows];
  const hasBand = forecasts.some((step) => step.lower_bound !== null);

  const table = (
    <table className="w-full text-left text-xs">
      <thead className="sticky top-0 bg-surface text-ink-secondary">
        <tr>
          <th className="px-2 py-1 font-medium">Session</th>
          <th className="px-2 py-1 font-medium">Target date</th>
          <th className="px-2 py-1 text-right font-medium">Forecast</th>
          <th className="px-2 py-1 text-right font-medium">95% interval</th>
          <th className="px-2 py-1 text-right font-medium">Change</th>
        </tr>
      </thead>
      <tbody>
        {forecasts.map((step) => (
          <tr key={step.step} className="border-t border-line">
            <td className="px-2 py-1">+{step.step}</td>
            <td className="px-2 py-1">{shortDate(step.target_date)}</td>
            <td className="px-2 py-1 text-right">{money(step.predicted_price, currency)}</td>
            <td className="px-2 py-1 text-right">
              {step.lower_bound === null
                ? "not available"
                : `${money(step.lower_bound, currency)} - ${money(step.upper_bound, currency)}`}
            </td>
            <td className="px-2 py-1 text-right">{step.change_percent.toFixed(2)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  );

  return (
    <ChartFrame height={height} table={table}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
          <CartesianGrid stroke={theme.grid} strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={axisDate}
            tick={{ fill: theme.inkMuted, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            minTickGap={40}
          />
          <YAxis
            domain={["auto", "auto"]}
            tick={{ fill: theme.inkMuted, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            tickFormatter={(value: number) => number(value, 0)}
            width={60}
          />
          <Tooltip
            content={<ChartTooltip formatter={(value) => money(value, currency)} />}
            cursor={{ stroke: theme.axis, strokeDasharray: "3 3" }}
          />
          <Legend wrapperStyle={{ fontSize: 11, color: theme.ink }} />
          {/* The forecast origin: everything to its right is predicted. */}
          <ReferenceLine
            x={referenceDate}
            stroke={theme.axis}
            strokeDasharray="4 4"
            label={{ value: "forecast origin", position: "insideTopLeft", fill: theme.inkMuted, fontSize: 10 }}
          />
          {hasBand && (
            <>
              {/* Stacked pair: an invisible base plus the interval width. */}
              <Area
                dataKey="bandBase"
                stackId="band"
                stroke="none"
                fill="none"
                legendType="none"
                tooltipType="none"
                isAnimationActive={false}
              />
              <Area
                dataKey="bandSpan"
                name="95% prediction interval"
                stackId="band"
                stroke="none"
                fill={theme.band}
                isAnimationActive={false}
              />
            </>
          )}
          <Line
            type="monotone"
            dataKey="actual"
            name="Actual"
            stroke={theme.actual}
            strokeWidth={2}
            dot={false}
            connectNulls
            activeDot={{ r: 4, strokeWidth: 2, stroke: theme.surface }}
          />
          <Line
            type="monotone"
            dataKey="forecast"
            name="Forecast"
            stroke={theme.predicted}
            strokeWidth={2}
            strokeDasharray="5 4"
            dot={{ r: 3, strokeWidth: 0, fill: theme.predicted }}
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function ActualVsPredictedChart({
  series,
  currency,
  height = 360,
}: {
  series: { model: string; label: string; curve: CurvePoint[] }[];
  currency?: string | null;
  height?: number;
}) {
  const theme = useChartTheme();
  if (series.length === 0) return null;

  const byDate = new Map<string, Record<string, number | string | null>>();
  series[0].curve.forEach((point) => {
    byDate.set(point.target_date, { date: point.target_date, actual: point.actual });
  });
  series.forEach(({ model, curve }) => {
    curve.forEach((point) => {
      const row = byDate.get(point.target_date) ?? { date: point.target_date, actual: point.actual };
      row[model] = point.predicted;
      byDate.set(point.target_date, row);
    });
  });
  const data = [...byDate.values()].sort((a, b) => String(a.date).localeCompare(String(b.date)));

  const table = (
    <table className="w-full text-left text-xs">
      <thead className="sticky top-0 bg-surface text-ink-secondary">
        <tr>
          <th className="px-2 py-1 font-medium">Session</th>
          <th className="px-2 py-1 text-right font-medium">Actual</th>
          {series.map((entry) => (
            <th key={entry.model} className="px-2 py-1 text-right font-medium">
              {entry.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((row) => (
          <tr key={String(row.date)} className="border-t border-line">
            <td className="px-2 py-1">{shortDate(String(row.date))}</td>
            <td className="px-2 py-1 text-right">{money(row.actual as number, currency)}</td>
            {series.map((entry) => (
              <td key={entry.model} className="px-2 py-1 text-right">
                {money(row[entry.model] as number, currency)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );

  return (
    <ChartFrame height={height} table={table}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
          <CartesianGrid stroke={theme.grid} strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={axisDate}
            tick={{ fill: theme.inkMuted, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            minTickGap={40}
          />
          <YAxis
            domain={["auto", "auto"]}
            tick={{ fill: theme.inkMuted, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            tickFormatter={(value: number) => number(value, 0)}
            width={60}
          />
          <Tooltip
            content={<ChartTooltip formatter={(value) => money(value, currency)} />}
            cursor={{ stroke: theme.axis, strokeDasharray: "3 3" }}
          />
          <Legend wrapperStyle={{ fontSize: 11, color: theme.ink }} />
          <Line
            type="monotone"
            dataKey="actual"
            name="Actual"
            stroke={theme.ink}
            strokeWidth={2.4}
            dot={false}
            connectNulls
          />
          {series.map((entry, index) => (
            <Line
              key={entry.model}
              type="monotone"
              dataKey={entry.model}
              name={entry.label}
              stroke={theme.series[index % theme.series.length]}
              strokeWidth={2}
              strokeDasharray="5 4"
              dot={false}
              connectNulls
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function SkillBarChart({ rows, height = 260 }: { rows: LeaderboardRow[]; height?: number }) {
  const theme = useChartTheme();
  const data = rows
    .filter((row) => row.mae_vs_naive_test !== null)
    .map((row) => ({ model: row.display_name, ratio: row.mae_vs_naive_test as number }));
  if (data.length === 0) return null;

  return (
    <ChartFrame height={height}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }} layout="vertical">
          <CartesianGrid stroke={theme.grid} strokeDasharray="2 4" horizontal={false} />
          <XAxis
            type="number"
            domain={[0, (max: number) => Math.max(1.2, max * 1.1)]}
            tick={{ fill: theme.inkMuted, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            tickFormatter={(value: number) => value.toFixed(2)}
          />
          <YAxis
            type="category"
            dataKey="model"
            tick={{ fill: theme.ink, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            width={120}
          />
          <Tooltip content={<ChartTooltip formatter={(value) => `${value.toFixed(3)}x naive MAE`} />} />
          {/* Anything left of 1.0 beat the naive baseline on unseen data. */}
          <ReferenceLine
            x={1}
            stroke={theme.ink}
            strokeDasharray="4 4"
            label={{ value: "naive baseline", position: "top", fill: theme.inkMuted, fontSize: 10 }}
          />
          <Bar dataKey="ratio" name="Test MAE / naive MAE" fill={theme.series[0]} radius={[0, 4, 4, 0]} />
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function EquityCurveChart({
  points,
  height = 300,
}: {
  points: { date: string; strategy: number; benchmark: number }[];
  height?: number;
}) {
  const theme = useChartTheme();
  return (
    <ChartFrame height={height}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={points} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
          <CartesianGrid stroke={theme.grid} strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={axisDate}
            tick={{ fill: theme.inkMuted, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            minTickGap={40}
          />
          <YAxis
            domain={["auto", "auto"]}
            tick={{ fill: theme.inkMuted, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            tickFormatter={(value: number) => value.toFixed(2)}
            width={56}
          />
          <ReferenceLine y={1} stroke={theme.axis} strokeDasharray="4 4" />
          <Tooltip content={<ChartTooltip formatter={(value) => `${value.toFixed(4)}x`} />} />
          <Legend wrapperStyle={{ fontSize: 11, color: theme.ink }} />
          <Line
            type="monotone"
            dataKey="strategy"
            name="Model-driven rule"
            stroke={theme.series[0]}
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="benchmark"
            name="Buy and hold"
            stroke={theme.series[1]}
            strokeWidth={2}
            strokeDasharray="5 4"
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function ContributionChart({
  contributions,
  height = 280,
}: {
  contributions: { feature: string; contribution: number }[];
  height?: number;
}) {
  const theme = useChartTheme();
  const data = contributions.slice(0, 10).map((item) => ({ ...item }));
  return (
    <ChartFrame height={height}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} layout="vertical" margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
          <CartesianGrid stroke={theme.grid} strokeDasharray="2 4" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: theme.inkMuted, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            tickFormatter={(value: number) => value.toFixed(4)}
          />
          <YAxis
            type="category"
            dataKey="feature"
            tick={{ fill: theme.ink, fontSize: AXIS_FONT }}
            stroke={theme.axis}
            width={120}
          />
          <ReferenceLine x={0} stroke={theme.axis} />
          <Tooltip content={<ChartTooltip formatter={(value) => value.toFixed(5)} />} />
          {/* Sign is the message here, so the diverging pair carries it, with
              the axis labels naming each feature. */}
          <Bar dataKey="contribution" name="Contribution to next-session log return" radius={[0, 4, 4, 0]}>
            {data.map((item) => (
              <Cell
                key={item.feature}
                fill={item.contribution >= 0 ? theme.series[0] : theme.series[7]}
              />
            ))}
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
