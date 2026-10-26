/** Response shapes mirrored from the FastAPI schemas in backend/app/schemas. */

export interface Provenance {
  source: string;
  retrieved_at: string;
  from_cache: boolean;
  warnings: string[];
}

export interface Symbol {
  ticker: string;
  name: string;
  exchange: string;
  currency: string;
}

export interface PriceBar {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  adj_close: number;
  volume: number | null;
}

export interface History {
  ticker: string;
  company_name: string | null;
  currency: string | null;
  n_rows: number;
  first_date: string;
  last_date: string;
  last_close: number;
  previous_close: number | null;
  change: number | null;
  change_percent: number | null;
  provenance: Provenance;
  bars: PriceBar[];
}

export type IndicatorPoint = Record<string, number | string | null>;

export interface Indicators {
  ticker: string;
  columns: string[];
  provenance: Provenance;
  points: IndicatorPoint[];
  definitions: Record<string, string>;
}

export interface ModelInfo {
  name: string;
  display_name: string;
  family: string;
  description: string;
  available: boolean;
  unavailable_reason: string | null;
  in_default_set: boolean;
  supports_explanations: boolean;
  refit_cost: string;
  min_train_rows: number;
}

export interface Metrics {
  n_observations: number;
  mae: number | null;
  rmse: number | null;
  mape: number | null;
  directional_accuracy: number | null;
  r2: number | null;
}

export interface ForecastStep {
  step: number;
  target_date: string;
  predicted_price: number;
  lower_bound: number | null;
  upper_bound: number | null;
  change: number;
  change_percent: number;
}

export interface ShapContribution {
  feature: string;
  value: number;
  contribution: number;
}

export interface Explanation {
  method?: string;
  target?: string;
  base_value?: number;
  prediction?: number;
  contributions?: ShapContribution[];
  note?: string;
  unavailable?: string;
}

export interface DatasetInfo {
  price_rows: number;
  feature_rows: number;
  feature_set?: string;
  n_features?: number;
  notes?: string[];
  n_rows: number;
  n_train: number;
  n_validation: number;
  n_test: number;
  train_start: string;
  train_end: string;
  validation_start: string;
  validation_end: string;
  test_start: string;
  test_end: string;
}

export interface BulkDealCoverage {
  ticker: string;
  symbol: string | null;
  in_dataset: boolean;
  sessions_with_deals: number;
  deal_rows: number;
  first_deal: string | null;
  last_deal: string | null;
  dataset_start: string;
  dataset_end: string;
  sufficient: boolean;
  minimum_sessions_required: number;
}

export interface BulkDeal {
  deal_date: string;
  security_name: string;
  client_name: string;
  side: "BUY" | "SELL";
  quantity: number;
  price: number;
  turnover: number;
}

export interface BulkDealDaily {
  date: string;
  deal_count: number;
  buy_quantity: number;
  sell_quantity: number;
  net_quantity: number;
  gross_turnover: number;
  deal_vwap: number | null;
}

export interface BulkDeals {
  ticker: string;
  coverage: BulkDealCoverage;
  sources: string[];
  deals: BulkDeal[];
  daily: BulkDealDaily[];
  notes: string[];
}

export interface BulkDealSymbol {
  symbol: string;
  ticker: string;
  security_name: string;
  deal_rows: number;
  sessions_with_deals: number;
  first_deal: string;
  last_deal: string;
  gross_turnover: number;
  sufficient: boolean;
}

export interface BulkDealsFeatureInfo {
  coverage: BulkDealCoverage;
  sessions_matched: number;
  rows_before_clipping: number;
  rows_after_clipping: number;
  features: string[];
  publication_lag_sessions: number;
  notes: string[];
}

export interface Prediction {
  run_id: number | null;
  ticker: string;
  company_name: string | null;
  currency: string | null;
  model: string;
  model_display_name: string;
  auto_selected: boolean;
  selection_rule: string | null;
  forecast_horizon: number;
  reference_date: string;
  reference_price: number;
  forecasts: ForecastStep[];
  metrics: Metrics | null;
  metrics_source: string | null;
  metrics_by_horizon: Record<string, Metrics>;
  mae_vs_naive_test: number | null;
  beats_naive_on_test: boolean | null;
  baseline_metrics: Metrics | null;
  dataset: DatasetInfo;
  feature_set: string | null;
  bulk_deals: BulkDealsFeatureInfo | null;
  provenance: Provenance;
  explanation: Explanation | null;
  limitations: string[];
  runtime_seconds: number | null;
  disclaimer: string;
}

export interface LeaderboardRow {
  model: string;
  display_name: string;
  family: string;
  validation_mae: number | null;
  validation_rmse: number | null;
  test_mae: number | null;
  test_rmse: number | null;
  test_mape: number | null;
  test_directional_accuracy: number | null;
  test_r2: number | null;
  mae_vs_naive_test: number | null;
  beats_naive_on_test: boolean | null;
  training_seconds: number | null;
  is_best: boolean;
  error: string | null;
}

export interface CurvePoint {
  origin_date: string;
  target_date: string;
  reference_price: number;
  actual: number;
  predicted: number;
}

export interface SegmentEvaluation {
  segment: string;
  horizon: number;
  n_origins: number;
  refits: number;
  period_start: string | null;
  period_end: string | null;
  metrics: Metrics | null;
  metrics_by_horizon: Record<string, Metrics>;
  curve: CurvePoint[];
}

export interface ModelRunReport {
  model: string;
  display_name: string;
  family: string;
  hyperparameters: Record<string, unknown>;
  training_seconds: number | null;
  validation: SegmentEvaluation | null;
  test: SegmentEvaluation | null;
  forecast: ForecastStep[];
  mae_vs_naive_validation: number | null;
  mae_vs_naive_test: number | null;
  beats_naive_on_test: boolean | null;
  error: string | null;
}

export interface Comparison {
  ticker: string;
  forecast_horizon: number;
  reference_date: string;
  reference_price: number;
  best_model: string | null;
  selection_rule: string;
  dataset: DatasetInfo;
  feature_set: string | null;
  bulk_deals: BulkDealsFeatureInfo | null;
  provenance: Provenance;
  leaderboard: LeaderboardRow[];
  models: ModelRunReport[];
  runtime_seconds: number;
  disclaimer: string;
}

export interface StoredLeaderboard {
  segment: string;
  horizon: number;
  ticker: string | null;
  note: string;
  rows: {
    model: string;
    ticker: string;
    segment: string;
    horizon: number;
    period_start: string | null;
    period_end: string | null;
    n_observations: number;
    mae: number | null;
    rmse: number | null;
    mape: number | null;
    directional_accuracy: number | null;
    r2: number | null;
    mae_vs_naive: number | null;
    training_seconds: number | null;
    evaluated_at: string | null;
  }[];
}

export interface WatchlistQuote {
  id: number;
  ticker: string;
  note: string | null;
  company_name: string | null;
  last_close: number | null;
  change_percent: number | null;
  last_date: string | null;
  error: string | null;
}

export interface PredictionRun {
  id: number;
  ticker: string;
  model_name: string;
  forecast_horizon: number;
  reference_date: string | null;
  reference_price: number | null;
  status: string;
  auto_selected: boolean;
  data_source: string | null;
  created_at: string;
  runtime_seconds: number | null;
  forecasts: ForecastStep[];
  metrics: Metrics | null;
  error_message: string | null;
}

export interface BacktestMetrics {
  total_return_percent: number;
  annualised_return_percent: number;
  annualised_volatility_percent: number;
  sharpe_ratio: number | null;
  max_drawdown_percent: number;
  final_equity: number;
}

export interface Backtest {
  ticker: string;
  model: string;
  period_start: string;
  period_end: string;
  n_sessions: number;
  threshold_percent: number;
  transaction_cost_bps: number;
  n_trades: number;
  time_in_market_percent: number;
  strategy: BacktestMetrics;
  benchmark: BacktestMetrics;
  equity_curve: { date: string; strategy: number; benchmark: number; in_market: boolean }[];
  notes: string[];
}

export interface User {
  id: number;
  email: string;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: User;
}
