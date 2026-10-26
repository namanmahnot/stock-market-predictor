/**
 * Theme state and the chart palette.
 *
 * The categorical hexes are assigned in a fixed order and never cycled, and
 * both sets were checked for colour-vision separation and contrast against the
 * surface they are drawn on. Light-mode slots 3 to 5 sit below 3:1 against the
 * light surface, so every chart with two or more series carries a legend and
 * offers a data-table view instead of relying on colour alone.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type ThemeMode = "light" | "dark";

export interface ChartTheme {
  series: string[];
  actual: string;
  predicted: string;
  band: string;
  grid: string;
  axis: string;
  ink: string;
  inkMuted: string;
  surface: string;
  up: string;
  down: string;
  neutral: string;
}

const LIGHT: ChartTheme = {
  series: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
  actual: "#2a78d6",
  predicted: "#eb6834",
  band: "rgba(235, 104, 52, 0.16)",
  grid: "#e1e0d9",
  axis: "#c3c2b7",
  ink: "#0b0b0b",
  inkMuted: "#898781",
  surface: "#fcfcfb",
  up: "#006300",
  down: "#d03b3b",
  neutral: "#898781",
};

const DARK: ChartTheme = {
  series: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
  actual: "#3987e5",
  predicted: "#d95926",
  band: "rgba(217, 89, 38, 0.22)",
  grid: "#263147",
  axis: "#3a4660",
  ink: "#f0efec",
  inkMuted: "#8f9bb3",
  surface: "#141b2b",
  up: "#0ca30c",
  down: "#e66767",
  neutral: "#8f9bb3",
};

const STORAGE_KEY = "spps.theme";

interface ThemeContextValue {
  mode: ThemeMode;
  chart: ChartTheme;
  toggle: () => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function readStoredMode(): ThemeMode | null {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored === "light" || stored === "dark" ? stored : null;
  } catch {
    return null;
  }
}

function systemMode(): ThemeMode {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<ThemeMode>(() => readStoredMode() ?? systemMode());

  useEffect(() => {
    document.documentElement.dataset.theme = mode;
    try {
      localStorage.setItem(STORAGE_KEY, mode);
    } catch {
      // A blocked storage API only costs the preference, not the theme itself.
    }
  }, [mode]);

  const toggle = useCallback(() => setMode((current) => (current === "dark" ? "light" : "dark")), []);

  const value = useMemo<ThemeContextValue>(
    () => ({ mode, chart: mode === "dark" ? DARK : LIGHT, toggle }),
    [mode, toggle],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used inside ThemeProvider.");
  }
  return context;
}

export function useChartTheme(): ChartTheme {
  return useTheme().chart;
}
