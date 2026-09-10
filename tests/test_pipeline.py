"""
Tests for the parts of this project that would fail quietly.

The priority is not coverage. It is the specific mistakes this project already
made once: a leaking feature reaching a model, a benchmark trained on the future,
and a rate quoted for a product line where rates are not negotiable.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "app")]

from benchmark import (RateBenchmark, division_of, in_scope,  # noqa: E402
                       negotiated_only, walk_forward)
from clean_data import clean, to_number  # noqa: E402
from leakage_check import audit  # noqa: E402
from service import app  # noqa: E402


@pytest.fixture(scope="session")
def deals():
    return pd.read_csv(ROOT / "data" / "processed" / "deals_clean.csv",
                       parse_dates=["contract_date"])


# --- cleaning ---------------------------------------------------------------

def test_to_number_handles_spreadsheet_placeholders():
    values = pd.Series(["1,234.5", "$ -", "-", "", "42"])
    assert to_number(values).tolist()[0] == 1234.5
    assert to_number(values).isna().sum() == 3


def test_cleaning_drops_impossible_commission_rates(deals):
    assert (deals.commission_pct > 0).all()
    assert (deals.commission_pct < 1).all()


def test_cleaning_never_silently_loses_rows():
    raw = pd.read_csv(ROOT / "data" / "raw" / "textile_sales.csv")
    result = clean(raw)
    audit_log = result.audit_frame
    dropped = audit_log.query("action == 'dropped'").rows.sum()
    assert len(raw) - dropped == len(result.data), "audit log must account for every row"


def test_product_division_is_extracted(deals):
    assert set(deals.product_division) <= {"fab", "knit", "denim", "garment"}
    takko = deals[deals.customer == "Takko Fab"]
    assert (takko.product_division == "fab").all()


# --- benchmark --------------------------------------------------------------

def test_benchmark_refuses_fixed_rate_product_lines(deals):
    scored = RateBenchmark().fit(deals).score(deals)
    assert not scored.product_division.isin(["fab", "denim"]).any()


def test_scope_is_decided_by_the_customer_suffix():
    assert in_scope("Takko") and in_scope("Takko Knit")
    assert not in_scope("Takko Fab")
    assert not in_scope("Authentic Style Denim")
    assert division_of("Takko Fab") == "fab"


def test_band_is_ordered(deals):
    model = RateBenchmark().fit(deals)
    table = model.table_
    assert (table.low <= table.expected).all()
    assert (table.expected <= table.high).all()


def test_shrinkage_pulls_thin_groups_toward_the_house(deals):
    model = RateBenchmark(k=30).fit(deals)
    table = model.table_.join(
        negotiated_only(deals).groupby("supplier").commission_pct.median().rename("own"))
    thin = table[table.support <= 5]
    thick = table[table.support >= 200]
    house = model.house_["expected"]
    assert (thin.expected - house).abs().mean() < (thick.expected - house).abs().mean()


def test_unknown_supplier_falls_back_to_house_average(deals):
    band = RateBenchmark().fit(deals).lookup("A Mill That Does Not Exist")
    assert band["basis"] == "house average"
    assert band["support"] == 0


def test_walk_forward_never_uses_the_future(deals):
    # Scoring 2021 with a benchmark that has seen 2021 would inflate everything.
    # If the guard works, dropping all later years must not change 2021's scores.
    full = walk_forward(deals)
    truncated = walk_forward(deals[deals.contract_year <= 2021])

    # Deal rows are not uniquely keyed by any column, so line them up on the
    # original row index, which walk_forward preserves.
    common = full.index.intersection(truncated.index)
    assert len(common) > 100
    np.testing.assert_allclose(full.loc[common, "bench_expected"].to_numpy(),
                               truncated.loc[common, "bench_expected"].to_numpy())


def test_leakage_is_only_counted_below_the_band(deals):
    scored = walk_forward(deals)
    assert (scored.loc[scored.position != "below band", "leakage_usd"] == 0).all()
    assert (scored.leakage_usd >= 0).all()


# --- leakage guard ----------------------------------------------------------

def test_audit_catches_a_derived_column(deals):
    features = deals[["deal_value_usd", "commission_pct", "contract_qty"]]
    flagged = audit(features, deals.commission_usd)
    assert "deal_value_usd x commission_pct" in set(flagged.feature)


def test_audit_is_quiet_on_clean_features(deals):
    features = deals[["contract_qty", "unit_rate_usd", "lead_days"]]
    assert audit(features, deals.commission_pct).empty


# --- service ----------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["deals_loaded"] > 0


def test_score_deal_flags_a_rate_below_the_band(client):
    body = client.post("/score-deal", json={
        "supplier": "Samad Rubber", "customer": "Takko",
        "quantity": 20000, "unit_price_usd": 5.5, "proposed_rate_pct": 1.0}).json()
    assert body["assessment"]["position"] == "below band"
    assert body["assessment"]["gap_to_expected_usd"] > 0
    assert body["comparable_deals"], "a flag must come with evidence"


def test_score_deal_stays_quiet_on_a_normal_rate(client):
    band = client.post("/score-deal", json={
        "supplier": "Samad Rubber", "customer": "Takko",
        "quantity": 20000, "unit_price_usd": 5.5}).json()["benchmark"]
    body = client.post("/score-deal", json={
        "supplier": "Samad Rubber", "customer": "Takko", "quantity": 20000,
        "unit_price_usd": 5.5, "proposed_rate_pct": band["expected_pct"]}).json()
    assert body["assessment"]["position"] == "within band"
    assert body["assessment"]["gap_to_expected_usd"] == 0


def test_unknown_supplier_is_warned_about(client):
    body = client.post("/score-deal", json={
        "supplier": "Nonexistent Mill", "customer": "Takko",
        "quantity": 1000, "unit_price_usd": 3.0}).json()
    assert "warning" in body
    assert body["benchmark"]["confidence"] == "low"


def test_out_of_scope_deal_is_refused_not_quietly_answered(client):
    body = client.post("/score-deal", json={
        "supplier": "Samad Rubber", "customer": "Takko Fab",
        "quantity": 1000, "unit_price_usd": 3.0, "proposed_rate_pct": 2.0}).json()
    assert body["out_of_scope"] is True
    assert body["product_line"] == "fab"
    assert "benchmark" not in body, "a fabric deal must not receive a knit/garment band"


def test_rejects_a_negative_quantity(client):
    assert client.post("/score-deal", json={
        "supplier": "Rajby", "customer": "Takko",
        "quantity": -5, "unit_price_usd": 3.0}).status_code == 422
