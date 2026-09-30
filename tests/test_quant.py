import math

import pytest

from src.mrm.monitoring.model_monitor import decile_edges, psi, returns_drift, grader_score_drift
from src.mrm.validation.model_validator import basel_zone, kupiec_pof, portfolio_backtest, run_var_backtest
from src.utils import metrics
from src.utils.data import load_portfolio


def test_kupiec_known_value():
    r = kupiec_pof(250, 5, 0.01)
    assert r["lr"] == pytest.approx(1.9568, abs=1e-4)
    assert r["p_value"] == pytest.approx(math.erfc(math.sqrt(r["lr"] / 2)))
    assert 0.16 < r["p_value"] < 0.17


def test_kupiec_edge_cases_and_nonnegative():
    zero = kupiec_pof(250, 0, 0.01)
    assert zero["lr"] == pytest.approx(-2 * 250 * math.log(0.99))
    allx = kupiec_pof(10, 10, 0.01)
    assert allx["lr"] == pytest.approx(-2 * 10 * math.log(0.01))
    exact = kupiec_pof(100, 1, 0.01)  # x/T == p
    assert exact["lr"] == pytest.approx(0.0, abs=1e-12)
    assert exact["p_value"] == pytest.approx(1.0)
    for x in range(0, 30):
        assert kupiec_pof(250, x, 0.01)["lr"] >= 0
    with pytest.raises(ValueError):
        kupiec_pof(10, 11, 0.01)


def test_basel_zones():
    assert [basel_zone(x) for x in (0, 4, 5, 9, 10, 20)] == ["GREEN", "GREEN", "YELLOW", "YELLOW", "RED", "RED"]


def test_backtest_is_out_of_sample_and_long_enough():
    bt = portfolio_backtest()
    for r in bt["methods"].values():
        assert r["test_observations"] >= 250
        assert r["basel_zone"] in ("GREEN", "YELLOW", "RED")
    assert set(bt["methods"]) == {"historical_rolling", "parametric_rolling", "beta_proxy_engine"}


def test_backtest_uses_only_past_window():
    # A single huge loss on day 300 must be an exception: the VaR for day 300
    # cannot see day 300 itself.
    pnl = [((i * 7919) % 11 - 5) * 1.0 for i in range(400)]
    pnl[300] = -1e6
    bt = run_var_backtest(pnl, window=250)
    assert bt["methods"]["historical_rolling"]["exceptions"] >= 1


def test_psi_identical_is_zero_and_known_value():
    ref = list(range(100))
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-12)
    edges = decile_edges(ref)
    assert len(edges) == 9
    cur = [0.0] * 50  # everything in the first decile bin
    eps = 1e-4
    expected = (1 - 0.1) * math.log(1 / 0.1) + 9 * (eps - 0.1) * math.log(eps / 0.1)
    assert psi(ref, cur) == pytest.approx(expected, rel=1e-9)


def test_returns_drift_uses_real_windows():
    d = returns_drift()
    assert d["status"] == "OK"
    assert d["reference_n"] == 250 and d["current_n"] == 60
    assert d["psi_value"] >= 0


def test_grader_psi_insufficient_data():
    r = grader_score_drift("MRM-001", [0.9] * 5, [0.9, 0.8, 1.0] * 20)
    assert r["status"] == "INSUFFICIENT_DATA" and "n=5" in r["message"]
    assert grader_score_drift("MRM-001", [0.9] * 40, [])["status"] == "NO_REFERENCE"


def test_kpis_computed_from_data():
    pf = load_portfolio()
    k = metrics.portfolio_kpis(pf)
    assert k["n_positions"] == 15
    assert round(k["aum"] / 1e6, 2) == 32.36
    tech = sum(p["market_value"] for p in pf["positions"] if p["sector"] == "Technology")
    assert k["tech_weight_pct"] == round(tech / k["total_market_value"] * 100, 2)
    assert sum(s["weight_pct"] for s in k["sector_weights"].values()) == pytest.approx(100, abs=0.05)


def test_price_history_anchored_and_synthetic():
    from src.utils.data import load_price_history

    ph = load_price_history()
    assert ph["synthetic"] is True
    assert len(ph["dates"]) - 1 >= 500
    for p in load_portfolio()["positions"]:
        assert ph["tickers"][p["ticker"]]["prices"][-1] == p["current_price"]
