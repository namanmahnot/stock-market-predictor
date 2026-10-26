import { BrowserRouter, Route, Routes } from "react-router-dom";

import { Layout } from "./components/Layout";
import { SessionProvider } from "./lib/session";
import { ThemeProvider } from "./lib/theme";
import { WorkspaceProvider } from "./lib/workspace";
import { Dashboard } from "./pages/Dashboard";
import { ModelComparison } from "./pages/ModelComparison";
import { PredictionLab } from "./pages/PredictionLab";
import { ReportsHistory } from "./pages/ReportsHistory";
import { StockAnalysis } from "./pages/StockAnalysis";

export default function App() {
  return (
    <ThemeProvider>
      <SessionProvider>
        <WorkspaceProvider>
          <BrowserRouter>
            <Routes>
              <Route element={<Layout />}>
                <Route index element={<Dashboard />} />
                <Route path="analysis" element={<StockAnalysis />} />
                <Route path="predict" element={<PredictionLab />} />
                <Route path="compare" element={<ModelComparison />} />
                <Route path="reports" element={<ReportsHistory />} />
                <Route path="*" element={<Dashboard />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </WorkspaceProvider>
      </SessionProvider>
    </ThemeProvider>
  );
}
