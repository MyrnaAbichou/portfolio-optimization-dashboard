"""Step 3 - Monte Carlo simulation and optimisation (Modern Portfolio Theory)."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.metrics import portfolio_performance


def simulate_random_portfolios(mean_returns, cov_matrix, risk_free_rate,
                               n_portfolios=10_000, seed=42):
    """Generate random long-only portfolios and measure each one.

    Each portfolio gets random positive weights that are rescaled to sum to 1.
    The maths is done for all portfolios at once (vectorised) instead of in a
    Python loop, which is roughly 100x faster.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(mean_returns)

    weights = rng.random((n_portfolios, n_assets))
    weights /= weights.sum(axis=1, keepdims=True)

    mu = np.asarray(mean_returns)
    sigma = np.asarray(cov_matrix)
    # einsum instead of "@": same result, and avoids false warnings that
    # NumPy 2.0 prints on Apple Silicon Macs with the "@" operator.
    rets = np.einsum("ij,j->i", weights, mu)
    # w' Sigma w for every row of `weights` at once
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, sigma, weights))
    sharpes = (rets - risk_free_rate) / vols

    results = pd.DataFrame({"Return": rets, "Volatility": vols, "Sharpe": sharpes})
    weights_df = pd.DataFrame(weights, columns=mean_returns.index)
    return results, weights_df


def _optimize(objective, n_assets, max_weight, extra_constraints=()):
    """Shared solver setup: weights between 0 and max_weight, summing to 1."""
    if max_weight * n_assets < 1:
        raise ValueError("Max weight too small: weights could never sum to 100%.")
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    constraints += list(extra_constraints)
    bounds = [(0.0, max_weight)] * n_assets
    start = np.full(n_assets, 1 / n_assets)          # equal-weight starting point
    result = minimize(objective, start, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Optimisation failed: {result.message}")
    return result.x


def max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate, max_weight=1.0):
    """Weights with the highest Sharpe ratio (the 'tangency' portfolio).

    scipy only minimises, so we minimise the *negative* Sharpe ratio.
    """
    def neg_sharpe(w):
        return -portfolio_performance(w, mean_returns, cov_matrix, risk_free_rate)[2]
    w = _optimize(neg_sharpe, len(mean_returns), max_weight)
    return pd.Series(w, index=mean_returns.index)


def min_volatility_portfolio(mean_returns, cov_matrix, max_weight=1.0):
    """Weights with the lowest possible volatility."""
    def vol(w):
        return portfolio_performance(w, mean_returns, cov_matrix, 0)[1]
    w = _optimize(vol, len(mean_returns), max_weight)
    return pd.Series(w, index=mean_returns.index)


def efficient_frontier(mean_returns, cov_matrix, n_points=40, max_weight=1.0):
    """For a range of target returns, find the least risky portfolio.

    The curve linking these points is the efficient frontier: no portfolio
    offers a higher return for the same risk.
    """
    min_vol_w = min_volatility_portfolio(mean_returns, cov_matrix, max_weight)
    low = float(min_vol_w @ mean_returns)
    high = _max_attainable_return(mean_returns, max_weight)
    frontier = []
    for target in np.linspace(low, high, n_points):
        target_constraint = {"type": "eq",
                             "fun": lambda w, t=target: w @ np.asarray(mean_returns) - t}
        try:
            w = _optimize(lambda w: portfolio_performance(w, mean_returns, cov_matrix, 0)[1],
                          len(mean_returns), max_weight, [target_constraint])
        except RuntimeError:
            continue                                  # skip unreachable targets
        _, vol, _ = portfolio_performance(w, mean_returns, cov_matrix, 0)
        frontier.append({"Return": target, "Volatility": vol})
    return pd.DataFrame(frontier)


def _max_attainable_return(mean_returns, max_weight):
    """Highest return reachable with the weight cap (fill best assets first)."""
    remaining, total = 1.0, 0.0
    for r in mean_returns.sort_values(ascending=False):
        w = min(max_weight, remaining)
        total += w * r
        remaining -= w
        if remaining <= 1e-12:
            break
    return total
