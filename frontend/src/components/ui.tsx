/** Small presentational building blocks shared by every page. */
import type { ReactNode } from "react";

import { dateTime, relativeTime } from "../lib/format";
import type { Provenance } from "../lib/types";

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className = "",
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-xl border border-line bg-surface-raised p-4 shadow-sm sm:p-5 ${className}`}
    >
      {(title || actions) && (
        <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-base font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-1 text-xs text-ink-secondary">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function StatCard({
  label,
  value,
  hint,
  tone = "neutral",
  icon,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "neutral" | "up" | "down";
  icon?: ReactNode;
}) {
  const toneClass =
    tone === "up" ? "text-[var(--color-up)]" : tone === "down" ? "text-[var(--color-down)]" : "text-ink";
  return (
    <div className="rounded-xl border border-line bg-surface-raised p-4">
      <div className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-ink-muted">
        {icon}
        <span>{label}</span>
      </div>
      <div className={`mt-2 text-2xl font-semibold ${toneClass}`}>{value}</div>
      {hint && <div className="mt-1 text-xs text-ink-secondary">{hint}</div>}
    </div>
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div role="status" aria-live="polite" className="flex items-center gap-2 text-sm text-ink-secondary">
      <span
        aria-hidden="true"
        className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-line border-t-accent"
      />
      <span>{label}</span>
    </div>
  );
}

export function LoadingBlock({ label, note }: { label: string; note?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-line bg-surface p-10 text-center">
      <Spinner label={label} />
      {note && <p className="max-w-md text-xs text-ink-muted">{note}</p>}
    </div>
  );
}

export function ErrorBanner({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div
      role="alert"
      className="flex flex-wrap items-start justify-between gap-3 rounded-xl border border-[var(--color-critical)] bg-[color-mix(in_oklab,var(--color-critical)_8%,var(--color-surface))] p-4"
    >
      <p className="flex items-start gap-2 text-sm text-ink">
        <span aria-hidden="true" className="font-bold text-[var(--color-critical)]">
          !
        </span>
        <span>{message}</span>
      </p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="rounded-lg border border-line px-3 py-1.5 text-xs font-medium text-ink hover:bg-surface"
        >
          Try again
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-line bg-surface p-8 text-center">
      <p className="text-sm font-medium text-ink">{title}</p>
      {children && <div className="mx-auto mt-2 max-w-md text-xs text-ink-secondary">{children}</div>}
    </div>
  );
}

/** Keyboard- and screen-reader-accessible explanation for a metric or indicator. */
export function InfoTip({ text }: { text: string }) {
  return (
    <span className="group relative inline-flex align-middle">
      <button
        type="button"
        aria-label={text}
        className="ml-1 inline-flex h-4 w-4 items-center justify-center rounded-full border border-line text-[10px] font-semibold text-ink-muted hover:border-accent hover:text-accent"
      >
        ?
      </button>
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-2 w-60 -translate-x-1/2 rounded-lg border border-line bg-surface-raised p-2 text-left text-xs font-normal normal-case text-ink opacity-0 shadow-lg transition-opacity group-focus-within:opacity-100 group-hover:opacity-100"
      >
        {text}
      </span>
    </span>
  );
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "warning" | "critical" | "accent";
}) {
  const tones: Record<string, string> = {
    neutral: "border-line text-ink-secondary",
    accent: "border-accent text-accent",
    good: "border-[var(--color-good)] text-[var(--color-up)]",
    warning: "border-[var(--color-warning)] text-ink",
    critical: "border-[var(--color-critical)] text-[var(--color-critical)]",
  };
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

/**
 * Baseline verdict. The icon and the words carry the meaning, so the colour is
 * never the only signal.
 */
export function BaselineVerdict({ beats, ratio }: { beats: boolean | null; ratio: number | null }) {
  if (beats === null || ratio === null) {
    return <Badge>baseline</Badge>;
  }
  return beats ? (
    <Badge tone="good">
      <span aria-hidden="true">✓</span> beats naive ({ratio.toFixed(2)}×)
    </Badge>
  ) : (
    <Badge tone="critical">
      <span aria-hidden="true">✕</span> no gain ({ratio.toFixed(2)}×)
    </Badge>
  );
}

export function ProvenanceLine({ provenance }: { provenance: Provenance }) {
  return (
    <div className="space-y-1">
      <p className="text-xs text-ink-muted">
        Source: <span className="font-medium text-ink-secondary">{provenance.source}</span> · retrieved{" "}
        {dateTime(provenance.retrieved_at)} ({relativeTime(provenance.retrieved_at)})
        {provenance.from_cache && " · served from cache"}
      </p>
      {provenance.warnings.length > 0 && (
        <ul className="list-inside list-disc text-xs text-ink-secondary">
          {provenance.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function Disclaimer({ text }: { text: string }) {
  return (
    <p className="rounded-lg border border-[var(--color-warning)] bg-[color-mix(in_oklab,var(--color-warning)_10%,var(--color-surface))] p-3 text-xs text-ink">
      <span aria-hidden="true" className="mr-1 font-semibold">
        ⚠
      </span>
      {text}
    </p>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs font-medium text-ink-secondary">
      <span className="flex items-center">{label}</span>
      {children}
      {hint && <span className="font-normal text-ink-muted">{hint}</span>}
    </label>
  );
}

export const selectClass =
  "rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none";

export const buttonClass =
  "inline-flex items-center justify-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50";

export const secondaryButtonClass =
  "inline-flex items-center justify-center gap-2 rounded-lg border border-line bg-surface px-3 py-2 text-sm font-medium text-ink transition hover:border-accent disabled:cursor-not-allowed disabled:opacity-50";
