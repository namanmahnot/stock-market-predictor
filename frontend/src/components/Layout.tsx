/** Application shell: navigation, global ticker search and the standing disclaimer. */
import { NavLink, Outlet, useNavigate } from "react-router-dom";

import { useTheme } from "../lib/theme";
import { useWorkspace } from "../lib/workspace";
import { AccountMenu } from "./AccountMenu";
import { TickerSearch } from "./TickerSearch";

const NAV = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/analysis", label: "Stock analysis" },
  { to: "/predict", label: "Prediction lab" },
  { to: "/compare", label: "Model comparison" },
  { to: "/reports", label: "Reports & history" },
];

export function Layout() {
  const { ticker, setTicker } = useWorkspace();
  const { mode, toggle } = useTheme();
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-page">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded-lg focus:bg-surface-raised focus:px-3 focus:py-2 focus:text-sm"
      >
        Skip to content
      </a>

      <header className="border-b border-line bg-surface-raised">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-3 px-4 py-3">
          <div className="flex items-center gap-2">
            <span
              aria-hidden="true"
              className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-sm font-bold text-white"
            >
              ₹$
            </span>
            <div>
              <p className="text-sm font-semibold leading-tight text-ink">Stock Market Price Prediction</p>
              <p className="text-[11px] leading-tight text-ink-muted">
                Historical analysis · forecasting · model evaluation
              </p>
            </div>
          </div>

          <div className="order-last w-full sm:order-none sm:ml-auto sm:w-auto sm:flex-1 sm:max-w-md">
            <TickerSearch
              onSelect={(symbol) => {
                setTicker(symbol);
                navigate("/analysis");
              }}
            />
          </div>

          <span className="rounded-lg border border-line px-3 py-2 text-xs font-semibold text-ink">
            {ticker}
          </span>

          <button
            type="button"
            onClick={toggle}
            aria-label={`Switch to ${mode === "dark" ? "light" : "dark"} theme`}
            className="rounded-lg border border-line bg-surface px-3 py-2 text-xs text-ink hover:border-accent"
          >
            {mode === "dark" ? "☀ Light" : "☾ Dark"}
          </button>

          <AccountMenu />
        </div>

        <nav aria-label="Main" className="mx-auto max-w-7xl overflow-x-auto px-4">
          <ul className="flex gap-1">
            {NAV.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) =>
                    `inline-block whitespace-nowrap border-b-2 px-3 py-2 text-sm font-medium transition ${
                      isActive
                        ? "border-accent text-accent"
                        : "border-transparent text-ink-secondary hover:text-ink"
                    }`
                  }
                >
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </header>

      <main id="main" className="mx-auto max-w-7xl px-4 py-6">
        <Outlet />
      </main>

      <footer className="border-t border-line bg-surface-raised">
        <div className="mx-auto max-w-7xl px-4 py-4 text-xs text-ink-muted">
          <p>
            Educational project. Forecasts are statistical estimates from historical data, they can be wrong,
            and nothing here is investment advice or a guaranteed future price.
          </p>
        </div>
      </footer>
    </div>
  );
}
