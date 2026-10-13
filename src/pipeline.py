"""Glue code: runs every step in order and returns all results in one dict."""

from src import metrics, optimizer


def run_analysis(prices, risk_free_rate, n_portfolios, max_weight=1.0, seed=42):
    returns = metrics.daily_returns(prices)
    mu = metrics.annualized_returns(returns)
    cov = metrics.annualized_covariance(returns)

    results, random_weights = optimizer.simulate_random_portfolios(
        mu, cov, risk_free_rate, n_portfolios, seed)

    w_sharpe = optimizer.max_sharpe_portfolio(mu, cov, risk_free_rate, max_weight)
    w_minvol = optimizer.min_volatility_portfolio(mu, cov, max_weight)
    frontier = optimizer.efficient_frontier(mu, cov, max_weight=max_weight)

    return {
        "returns": returns,
        "mean_returns": mu,
        "cov": cov,
        "asset_stats": metrics.asset_summary(returns, risk_free_rate),
        "random_results": results,
        "random_weights": random_weights,
        "w_sharpe": w_sharpe,
        "w_minvol": w_minvol,
        "perf_sharpe": metrics.portfolio_performance(w_sharpe, mu, cov, risk_free_rate),
        "perf_minvol": metrics.portfolio_performance(w_minvol, mu, cov, risk_free_rate),
        "frontier": frontier,
    }
