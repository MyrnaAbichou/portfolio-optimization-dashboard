"""Charts built with matplotlib, shared by the script and the Streamlit app."""

import matplotlib.pyplot as plt
import numpy as np


def plot_risk_return(results, frontier, asset_stats, perf_sharpe, perf_minvol):
    """Risk-return cloud of random portfolios + efficient frontier + key points."""
    fig, ax = plt.subplots(figsize=(10, 6.5))

    cloud = ax.scatter(results["Volatility"], results["Return"], c=results["Sharpe"],
                       cmap="viridis", s=6, alpha=0.6)
    fig.colorbar(cloud, ax=ax, label="Sharpe ratio")

    ax.plot(frontier["Volatility"], frontier["Return"], color="#d62728",
            linewidth=2.2, label="Efficient frontier")

    ax.scatter(asset_stats["Annual volatility"], asset_stats["Annual return"],
               color="black", marker="o", s=35, label="Individual assets")
    for ticker, row in asset_stats.iterrows():
        ax.annotate(ticker, (row["Annual volatility"], row["Annual return"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=9)

    ax.scatter(perf_sharpe[1], perf_sharpe[0], marker="*", s=380, color="#ff7f0e",
               edgecolor="black", zorder=5, label="Max Sharpe")
    ax.scatter(perf_minvol[1], perf_minvol[0], marker="*", s=380, color="#1f77b4",
               edgecolor="black", zorder=5, label="Min volatility")

    ax.set_xlabel("Annual volatility (risk)")
    ax.set_ylabel("Annual expected return")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(f"Risk-return space: {len(results):,} random portfolios")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_correlation(returns):
    """Heatmap of correlations between assets (low = better diversification)."""
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Correlation of daily returns")
    fig.tight_layout()
    return fig


def plot_weights(weights_by_portfolio):
    """Side-by-side bars of the weights of each optimal portfolio."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    n = len(weights_by_portfolio)
    tickers = next(iter(weights_by_portfolio.values())).index
    x = np.arange(len(tickers))
    width = 0.8 / n
    for i, (name, w) in enumerate(weights_by_portfolio.items()):
        ax.bar(x + i * width - 0.4 + width / 2, w.values, width, label=name)
    ax.set_xticks(x, tickers)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Weight")
    ax.set_title("Optimal portfolio allocations")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cumulative(returns, weights_by_portfolio):
    """Growth of $1 invested in each optimal portfolio vs an equal-weight one."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    equal = np.full(returns.shape[1], 1 / returns.shape[1])
    series = {"Equal weight": returns @ equal}
    series.update({name: returns @ w.values for name, w in weights_by_portfolio.items()})
    for name, r in series.items():
        ax.plot((1 + r).cumprod(), label=name, linewidth=1.6)
    ax.set_ylabel("Value of $1 invested")
    ax.set_title("Historical growth (in-sample: weights use the same data)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
