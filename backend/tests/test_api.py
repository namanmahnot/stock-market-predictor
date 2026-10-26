"""API contract, error handling and per-user data isolation (SRS 13, 16, 18)."""
from __future__ import annotations

import pytest

from tests.conftest import SYNTHETIC_TICKER

PREDICTIONS = "/api/v1/predictions"
WATCHLIST = "/api/v1/watchlist"


# -- system and stocks -----------------------------------------------------
def test_health_reports_the_configuration(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] in {"sqlite", "postgresql"}


def test_openapi_documentation_is_served(client):
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_search_returns_matching_symbols(client):
    response = client.get("/api/v1/stocks/search", params={"q": "rel"})
    assert response.status_code == 200
    assert any(entry["ticker"] == "RELIANCE.NS" for entry in response.json())


def test_history_includes_provenance_and_bars(client):
    response = client.get(f"/api/v1/stocks/{SYNTHETIC_TICKER}/history", params={"days": 365})
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == SYNTHETIC_TICKER
    assert body["n_rows"] == len(body["bars"]) > 100
    assert body["provenance"]["source"] == "csv"
    assert body["provenance"]["retrieved_at"]
    dates = [bar["date"] for bar in body["bars"]]
    assert dates == sorted(dates)
    assert body["last_close"] == pytest.approx(body["bars"][-1]["adj_close"])


def test_history_rejects_a_malformed_ticker(client):
    response = client.get("/api/v1/stocks/not a ticker/history")
    assert response.status_code == 422
    assert "valid ticker" in response.json()["detail"]


def test_history_reports_an_unavailable_symbol_without_inventing_prices(client):
    response = client.get("/api/v1/stocks/MISSING/history")
    assert response.status_code == 502
    assert "MISSING" in response.json()["detail"]


def test_history_rejects_an_out_of_range_window(client):
    assert client.get(f"/api/v1/stocks/{SYNTHETIC_TICKER}/history", params={"days": 5}).status_code == 422


def test_indicators_expose_definitions_and_warmup_nulls(client):
    response = client.get(f"/api/v1/stocks/{SYNTHETIC_TICKER}/indicators", params={"days": 365})
    assert response.status_code == 200
    body = response.json()
    assert "rsi_14" in body["columns"]
    assert body["definitions"]["rsi_14"]
    assert body["points"][0]["sma_50"] is None, "warm-up rows must be null, not fabricated"
    assert body["points"][-1]["sma_50"] is not None
    assert 0.0 <= body["points"][-1]["rsi_14"] <= 100.0


# -- models ----------------------------------------------------------------
def test_model_catalogue_lists_the_default_set(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    catalogue = response.json()
    defaults = {entry["name"] for entry in catalogue if entry["in_default_set"]}
    assert defaults == {"naive", "linear_regression", "arima", "random_forest", "xgboost", "lstm"}
    assert all(entry["description"] for entry in catalogue)


def test_compare_evaluates_models_on_identical_windows(client):
    response = client.post(
        "/api/v1/models/compare",
        json={"ticker": SYNTHETIC_TICKER, "horizon": 1, "models": ["naive", "linear_regression"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["best_model"] in {"naive", "linear_regression"}
    assert len(body["leaderboard"]) == 2
    periods = {
        (model["test"]["period_start"], model["test"]["period_end"]) for model in body["models"]
    }
    assert len(periods) == 1, "all models must be scored on the same test window"
    assert body["disclaimer"]


def test_leaderboard_returns_stored_rankings(client):
    client.post(
        "/api/v1/models/compare",
        json={"ticker": SYNTHETIC_TICKER, "horizon": 1, "models": ["naive", "linear_regression"]},
    )
    response = client.get("/api/v1/models/leaderboard", params={"ticker": SYNTHETIC_TICKER})
    assert response.status_code == 200
    body = response.json()
    assert {row["model"] for row in body["rows"]} == {"naive", "linear_regression"}
    maes = [row["mae"] for row in body["rows"]]
    assert maes == sorted(maes)


# -- predictions -----------------------------------------------------------
def test_prediction_returns_a_forecast_with_metrics_and_a_disclaimer(client):
    response = client.post(
        PREDICTIONS, json={"ticker": SYNTHETIC_TICKER, "model": "naive", "horizon": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "naive"
    assert len(body["forecasts"]) == 3
    assert body["metrics"]["mae"] is not None
    assert body["metrics_source"].startswith("Walk-forward")
    assert body["reference_price"] > 0
    assert body["dataset"]["n_test"] > 0
    assert body["limitations"]
    assert "not investment advice" in body["disclaimer"]
    assert body["run_id"] is not None


def test_auto_selection_names_its_rule(client):
    response = client.post(
        PREDICTIONS,
        json={"ticker": SYNTHETIC_TICKER, "model": "auto", "horizon": 1, "save": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["auto_selected"] is True
    assert "validation" in body["selection_rule"]
    assert body["run_id"] is None


def test_prediction_rejects_an_unknown_model(client):
    response = client.post(PREDICTIONS, json={"ticker": SYNTHETIC_TICKER, "model": "magic"})
    assert response.status_code == 422


def test_prediction_rejects_an_out_of_range_horizon(client):
    response = client.post(PREDICTIONS, json={"ticker": SYNTHETIC_TICKER, "model": "naive", "horizon": 999})
    assert response.status_code == 422


def test_prediction_rejects_a_malformed_ticker(client):
    response = client.post(PREDICTIONS, json={"ticker": "a b c", "model": "naive"})
    assert response.status_code == 422


def test_prediction_reports_an_unavailable_provider(client):
    response = client.post(PREDICTIONS, json={"ticker": "MISSING", "model": "naive"})
    assert response.status_code == 502


def test_saved_prediction_can_be_retrieved(client):
    created = client.post(
        PREDICTIONS, json={"ticker": SYNTHETIC_TICKER, "model": "naive", "horizon": 2}
    ).json()
    response = client.get(f"{PREDICTIONS}/{created['run_id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == created["run_id"]
    assert len(body["forecasts"]) == 2
    assert body["status"] == "completed"
    assert body["metrics"]["mae"] is not None


def test_retrieving_a_missing_run_is_a_404(client):
    assert client.get(f"{PREDICTIONS}/987654").status_code == 404


def test_prediction_history_requires_authentication(client):
    assert client.get(PREDICTIONS).status_code == 401


def test_prediction_history_only_returns_the_callers_runs(auth_client, client):
    auth_client.post(PREDICTIONS, json={"ticker": SYNTHETIC_TICKER, "model": "naive", "horizon": 1})
    mine = auth_client.get(PREDICTIONS)
    assert mine.status_code == 200
    assert len(mine.json()) == 1

    other = client.post(
        "/api/v1/auth/register", json={"email": "other@example.com", "password": "OtherPassword1"}
    ).json()
    headers = {"Authorization": f"Bearer {other['access_token']}"}
    assert auth_client.get(PREDICTIONS, headers=headers).json() == []


def test_another_users_run_is_not_readable(auth_client, client):
    run_id = auth_client.post(
        PREDICTIONS, json={"ticker": SYNTHETIC_TICKER, "model": "naive", "horizon": 1}
    ).json()["run_id"]
    other = client.post(
        "/api/v1/auth/register", json={"email": "snoop@example.com", "password": "SnoopPassword1"}
    ).json()
    response = auth_client.get(
        f"{PREDICTIONS}/{run_id}", headers={"Authorization": f"Bearer {other['access_token']}"}
    )
    assert response.status_code == 404


def test_explanations_are_returned_for_a_tree_model(client):
    response = client.post(
        PREDICTIONS,
        json={
            "ticker": SYNTHETIC_TICKER,
            "model": "random_forest",
            "horizon": 1,
            "save": False,
            "include_explanation": True,
        },
    )
    assert response.status_code == 200
    explanation = response.json()["explanation"]
    assert explanation and explanation["contributions"]
    assert "not establish a causal effect" in explanation["note"]


# -- auth ------------------------------------------------------------------
def test_register_then_login(client):
    registration = client.post(
        "/api/v1/auth/register", json={"email": "new@example.com", "password": "GoodPassword1"}
    )
    assert registration.status_code == 201
    assert registration.json()["token_type"] == "bearer"

    login = client.post(
        "/api/v1/auth/login", json={"email": "new@example.com", "password": "GoodPassword1"}
    )
    assert login.status_code == 200
    token = login.json()["access_token"]
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["email"] == "new@example.com"


def test_password_hash_is_never_returned(client):
    body = client.post(
        "/api/v1/auth/register", json={"email": "secret@example.com", "password": "GoodPassword1"}
    ).json()
    assert set(body["user"]) == {"id", "email", "created_at"}
    assert "password" not in body and "$2b$" not in str(body)


def test_duplicate_registration_is_rejected(client):
    payload = {"email": "dup@example.com", "password": "GoodPassword1"}
    assert client.post("/api/v1/auth/register", json=payload).status_code == 201
    assert client.post("/api/v1/auth/register", json=payload).status_code == 409


def test_login_with_a_wrong_password_is_rejected(client):
    client.post("/api/v1/auth/register", json={"email": "pw@example.com", "password": "GoodPassword1"})
    response = client.post(
        "/api/v1/auth/login", json={"email": "pw@example.com", "password": "WrongPassword1"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password."


def test_login_for_an_unknown_email_gives_the_same_message(client):
    response = client.post(
        "/api/v1/auth/login", json={"email": "ghost@example.com", "password": "GoodPassword1"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password."


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "not-an-email", "password": "GoodPassword1"},
        {"email": "short@example.com", "password": "abc"},
    ],
)
def test_registration_validates_its_input(client, payload):
    assert client.post("/api/v1/auth/register", json=payload).status_code == 422


def test_an_invalid_token_is_rejected(client):
    response = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not.a.token"})
    assert response.status_code == 401


# -- watchlist -------------------------------------------------------------
def test_watchlist_requires_authentication(client):
    assert client.get(WATCHLIST).status_code == 401
    assert client.post(WATCHLIST, json={"ticker": "AAPL"}).status_code == 401


def test_watchlist_round_trip(auth_client):
    created = auth_client.post(WATCHLIST, json={"ticker": SYNTHETIC_TICKER, "note": "demo"})
    assert created.status_code == 201
    item_id = created.json()["id"]

    listed = auth_client.get(WATCHLIST)
    assert listed.status_code == 200
    entry = listed.json()[0]
    assert entry["ticker"] == SYNTHETIC_TICKER
    assert entry["last_close"] > 0

    assert auth_client.delete(f"{WATCHLIST}/{item_id}").status_code == 204
    assert auth_client.get(WATCHLIST).json() == []


def test_watchlist_rejects_a_duplicate_ticker(auth_client):
    auth_client.post(WATCHLIST, json={"ticker": SYNTHETIC_TICKER})
    assert auth_client.post(WATCHLIST, json={"ticker": SYNTHETIC_TICKER}).status_code == 409


def test_watchlist_reports_an_unavailable_quote_per_row(auth_client):
    auth_client.post(WATCHLIST, json={"ticker": "MISSING"})
    rows = auth_client.get(WATCHLIST).json()
    assert rows[0]["error"] and rows[0]["last_close"] is None


def test_users_cannot_see_or_delete_each_others_watchlists(auth_client, client):
    item_id = auth_client.post(WATCHLIST, json={"ticker": SYNTHETIC_TICKER}).json()["id"]
    other = client.post(
        "/api/v1/auth/register", json={"email": "thief@example.com", "password": "ThiefPassword1"}
    ).json()
    headers = {"Authorization": f"Bearer {other['access_token']}"}

    assert auth_client.get(WATCHLIST, headers=headers).json() == []
    assert auth_client.delete(f"{WATCHLIST}/{item_id}", headers=headers).status_code == 404
    assert len(auth_client.get(WATCHLIST).json()) == 1


# -- reports ---------------------------------------------------------------
def test_forecast_report_downloads_as_csv(client):
    run_id = client.post(
        PREDICTIONS, json={"ticker": SYNTHETIC_TICKER, "model": "naive", "horizon": 3}
    ).json()["run_id"]
    response = client.get(f"/api/v1/reports/{run_id}", params={"format": "csv"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    text = response.text
    assert "predicted_price" in text and "disclaimer" in text
    assert text.count("\n") > 10


def test_forecast_report_downloads_as_pdf(client):
    run_id = client.post(
        PREDICTIONS, json={"ticker": SYNTHETIC_TICKER, "model": "naive", "horizon": 2}
    ).json()["run_id"]
    response = client.get(f"/api/v1/reports/{run_id}", params={"format": "pdf"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


def test_report_rejects_an_unsupported_format(client):
    assert client.get("/api/v1/reports/1", params={"format": "docx"}).status_code == 422


def test_comparison_report_downloads_as_csv(client):
    response = client.post(
        "/api/v1/reports/comparison",
        json={"ticker": SYNTHETIC_TICKER, "horizon": 1, "models": ["naive", "linear_regression"]},
    )
    assert response.status_code == 200
    assert "validation_mae" in response.text
    assert "mae_vs_naive_test" in response.text


# -- rate limiting ---------------------------------------------------------
def test_expensive_endpoints_are_rate_limited(client, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "heavy_endpoint_limit_per_minute", 1)
    payload = {"ticker": SYNTHETIC_TICKER, "model": "naive", "horizon": 1, "save": False}
    assert client.post(PREDICTIONS, json=payload).status_code == 200
    limited = client.post(PREDICTIONS, json=payload)
    assert limited.status_code == 429
    assert "Retry-After" in limited.headers


# -- bulk deals ------------------------------------------------------------
def test_bulk_deals_for_a_covered_ticker(client):
    response = client.get(f"/api/v1/bulk-deals/{SYNTHETIC_TICKER}")
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == SYNTHETIC_TICKER
    assert body["coverage"]["in_dataset"] is True
    assert body["coverage"]["sufficient"] is True
    assert body["deals"] and body["daily"]
    assert body["sources"]
    # Most recent disclosure first.
    dates = [deal["deal_date"] for deal in body["deals"]]
    assert dates == sorted(dates, reverse=True)
    first = body["deals"][0]
    assert first["side"] in {"BUY", "SELL"}
    assert first["turnover"] == pytest.approx(first["quantity"] * first["price"])


def test_bulk_deals_explains_an_absent_large_cap(client):
    response = client.get("/api/v1/bulk-deals/RELIANCE.NS")
    assert response.status_code == 200
    body = response.json()
    assert body["coverage"]["in_dataset"] is False
    assert body["deals"] == []
    assert any("0.5%" in note for note in body["notes"])


def test_bulk_deals_explains_a_non_nse_symbol(client):
    body = client.get("/api/v1/bulk-deals/%5ENSEI").json()
    assert any("NSE equities only" in note for note in body["notes"])


def test_bulk_deals_rejects_a_malformed_ticker(client):
    assert client.get("/api/v1/bulk-deals/not a ticker").status_code == 422


def test_bulk_deal_symbols_are_ranked(client):
    response = client.get("/api/v1/bulk-deals/symbols", params={"only_sufficient": True})
    assert response.status_code == 200
    rows = response.json()
    assert rows and rows[0]["ticker"] == f"{SYNTHETIC_TICKER}.NS"
    assert all(row["sufficient"] for row in rows)
    sessions = [row["sessions_with_deals"] for row in rows]
    assert sessions == sorted(sessions, reverse=True)


def test_prediction_can_use_the_bulk_deal_feature_group(client):
    response = client.post(
        PREDICTIONS,
        json={
            "ticker": SYNTHETIC_TICKER,
            "model": "random_forest",
            "horizon": 1,
            "save": False,
            "use_bulk_deals": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["feature_set"] == "price+bulk_deals"
    assert body["bulk_deals"]["coverage"]["sufficient"] is True
    assert body["dataset"]["n_features"] == 34
    assert any("clipped" in note for note in body["dataset"]["notes"])


def test_prediction_without_the_feature_group_is_unaffected(client):
    body = client.post(
        PREDICTIONS,
        json={
            "ticker": SYNTHETIC_TICKER,
            "model": "random_forest",
            "horizon": 1,
            "save": False,
            "use_bulk_deals": False,
        },
    ).json()
    assert body["feature_set"] == "price_only"
    assert body["bulk_deals"] is None
    assert body["dataset"]["n_features"] == 24


def test_the_two_feature_sets_are_cached_separately(client):
    payload = {"ticker": SYNTHETIC_TICKER, "model": "naive", "horizon": 1, "save": False}
    without = client.post(PREDICTIONS, json={**payload, "use_bulk_deals": False}).json()
    with_bulk = client.post(PREDICTIONS, json={**payload, "use_bulk_deals": True}).json()
    assert without["feature_set"] != with_bulk["feature_set"]
    # The bulk arm is clipped to the disclosure window, so it has fewer rows.
    assert with_bulk["dataset"]["feature_rows"] < without["dataset"]["feature_rows"]


def test_comparison_reports_the_feature_set(client):
    body = client.post(
        "/api/v1/models/compare",
        json={
            "ticker": SYNTHETIC_TICKER,
            "horizon": 1,
            "models": ["naive", "linear_regression"],
            "use_bulk_deals": True,
        },
    ).json()
    assert body["feature_set"] == "price+bulk_deals"
    assert body["bulk_deals"]["coverage"]["in_dataset"] is True
