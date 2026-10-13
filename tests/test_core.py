"""Checks that the maths is right, using small made-up data. Run: pytest"""

import numpy as np
import pandas as pd
import pytest

from src import metrics, optimizer
from src.data import clean_prices


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    data = rng.normal([0.0006, 0.0003, 0.0001], [0.02, 0.012, 0.006], size=(1000, 3))
    return pd.DataFrame(data, columns=["A", "B", "C"])


def test_portfolio_performance_matches_hand_calculation():
    mu = pd.Series([0.10, 0.05], index=["X", "Y"])
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=mu.index, columns=mu.index)
    ret, vol, sharpe = metrics.portfolio_performance([0.5, 0.5], mu, cov, 0.02)
    assert ret == pytest.approx(0.075)
    assert vol == pytest.approx(np.sqrt(0.25 * 0.04 + 0.25 * 0.01))
    assert sharpe == pytest.approx((0.075 - 0.02) / vol)


def test_max_drawdown():
    r = pd.Series([0.10, -0.50, 0.20])      # 1 -> 1.1 -> 0.55 -> 0.66
    assert metrics.max_drawdown(r) == pytest.approx(-0.5)


def test_random_weights_sum_to_one(returns):
    mu, cov = metrics.annualized_returns(returns), metrics.annualized_covariance(returns)
    _, w = optimizer.simulate_random_portfolios(mu, cov, 0.02, 500)
    assert np.allclose(w.sum(axis=1), 1)
    assert (w >= 0).all().all()


def test_optimisers_beat_random_portfolios(returns):
    mu, cov = metrics.annualized_returns(returns), metrics.annualized_covariance(returns)
    res, _ = optimizer.simulate_random_portfolios(mu, cov, 0.02, 5000)
    w_s = optimizer.max_sharpe_portfolio(mu, cov, 0.02)
    w_v = optimizer.min_volatility_portfolio(mu, cov)
    assert metrics.portfolio_performance(w_s, mu, cov, 0.02)[2] >= res["Sharpe"].max() - 1e-6
    assert metrics.portfolio_performance(w_v, mu, cov, 0)[1] <= res["Volatility"].min() + 1e-6


def test_weight_cap_is_respected(returns):
    mu, cov = metrics.annualized_returns(returns), metrics.annualized_covariance(returns)
    w = optimizer.max_sharpe_portfolio(mu, cov, 0.02, max_weight=0.4)
    assert w.max() <= 0.4 + 1e-6
    assert w.sum() == pytest.approx(1)


def test_clean_prices_drops_sparse_assets():
    idx = pd.date_range("2024-01-01", periods=100)
    prices = pd.DataFrame({"OK": np.arange(100.0) + 1, "SPARSE": np.nan}, index=idx)
    prices.loc[idx[:10], "SPARSE"] = 1.0
    cleaned = clean_prices(prices)
    assert list(cleaned.columns) == ["OK"]
