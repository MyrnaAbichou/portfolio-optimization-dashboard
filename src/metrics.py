"""Step 2 - financial metrics: returns, volatility, covariance, Sharpe, drawdown."""

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS


def daily_returns(prices):
    """Simple daily return: (P_t / P_{t-1}) - 1."""
    return prices.pct_change().dropna()


def annualized_returns(returns):
    """Average daily return x 252 = expected yearly return (arithmetic)."""
    return returns.mean() * TRADING_DAYS


def annualized_volatility(returns):
    """Daily standard deviation x sqrt(252).

    Variance grows linearly with time, so standard deviation grows with the
    square root of time.
    """
    return returns.std() * np.sqrt(TRADING_DAYS)


def annualized_covariance(returns):
    """Covariance matrix of daily returns x 252."""
    return returns.cov() * TRADING_DAYS


def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    """Return (expected return, volatility, Sharpe ratio) of one portfolio.

    return     = w . mu
    volatility = sqrt(w' . Sigma . w)
    Sharpe     = (return - risk-free rate) / volatility
    """
    weights = np.asarray(weights)
    ret = float(weights @ np.asarray(mean_returns))
    vol = float(np.sqrt(weights @ np.asarray(cov_matrix) @ weights))
    sharpe = (ret - risk_free_rate) / vol
    return ret, vol, sharpe


def max_drawdown(returns_series):
    """Largest peak-to-trough loss of a return series (e.g. -0.35 = -35%)."""
    wealth = (1 + returns_series).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1
    return float(drawdown.min())


def asset_summary(returns, risk_free_rate):
    """One table with the key statistics of every individual asset."""
    mu = annualized_returns(returns)
    vol = annualized_volatility(returns)
    summary = pd.DataFrame({
        "Annual return": mu,
        "Annual volatility": vol,
        "Sharpe ratio": (mu - risk_free_rate) / vol,
        "Max drawdown": returns.apply(max_drawdown),
    })
    return summary.sort_values("Sharpe ratio", ascending=False)
