/** Stock search with keyboard navigation (FR-01). */
import { useEffect, useId, useMemo, useRef, useState } from "react";

import { StocksApi, describeError } from "../lib/api";
import type { Symbol } from "../lib/types";

export function TickerSearch({
  onSelect,
  placeholder = "Search ticker or company, e.g. AAPL or Reliance",
  autoFocus = false,
}: {
  onSelect: (ticker: string) => void;
  placeholder?: string;
  autoFocus?: boolean;
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Symbol[]>([]);
  const [open, setOpen] = useState(false);
  const [highlighted, setHighlighted] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const listId = useId();
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      return;
    }
    const timer = setTimeout(() => {
      StocksApi.search(query)
        .then((found) => {
          setResults(found);
          setError(null);
          setHighlighted(0);
        })
        .catch((cause) => setError(describeError(cause)));
    }, 180); // debounce so typing does not hit the API on every keystroke
    return () => clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    function onClickOutside(event: MouseEvent) {
      if (container.current && !container.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  const visible = useMemo(() => (open && results.length > 0 ? results : []), [open, results]);

  function choose(ticker: string) {
    onSelect(ticker.toUpperCase());
    setQuery("");
    setResults([]);
    setOpen(false);
  }

  function onKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setHighlighted((index) => Math.min(index + 1, visible.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setHighlighted((index) => Math.max(index - 1, 0));
    } else if (event.key === "Enter") {
      event.preventDefault();
      const picked = visible[highlighted];
      choose(picked ? picked.ticker : query);
    } else if (event.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <div ref={container} className="relative w-full max-w-md">
      <label className="sr-only" htmlFor={`${listId}-input`}>
        Search for a stock
      </label>
      <input
        id={`${listId}-input`}
        type="search"
        role="combobox"
        aria-expanded={visible.length > 0}
        aria-controls={listId}
        aria-autocomplete="list"
        autoFocus={autoFocus}
        value={query}
        placeholder={placeholder}
        onChange={(event) => {
          setQuery(event.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={onKeyDown}
        className="w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-muted focus:border-accent focus:outline-none"
      />
      {error && <p className="mt-1 text-xs text-[var(--color-critical)]">{error}</p>}
      {visible.length > 0 && (
        <ul
          id={listId}
          role="listbox"
          className="absolute z-30 mt-1 max-h-72 w-full overflow-auto rounded-lg border border-line bg-surface-raised py-1 shadow-xl"
        >
          {visible.map((entry, index) => (
            <li key={entry.ticker} role="option" aria-selected={index === highlighted}>
              <button
                type="button"
                onMouseEnter={() => setHighlighted(index)}
                onClick={() => choose(entry.ticker)}
                className={`flex w-full items-baseline justify-between gap-3 px-3 py-2 text-left text-sm ${
                  index === highlighted ? "bg-accent-soft" : ""
                }`}
              >
                <span className="font-semibold text-ink">{entry.ticker}</span>
                <span className="truncate text-xs text-ink-secondary">{entry.name}</span>
                <span className="shrink-0 text-[11px] text-ink-muted">{entry.exchange}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
