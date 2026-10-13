"""Charts built with matplotlib, shared by the script and the Streamlit app.

Sober report style: white background, navy as the main colour, one accent.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

# ---------- Palette ----------
BG = "#FFFFFF"
PANEL = "#FFFFFF"
TEXT = "#1F2933"
MUTED = "#5F6B76"
GRID = "#E3E7EB"
NAVY = "#1F4E79"       # main colour
ACCENT = "#C8553D"     # highlights the optimal portfolio
LIGHT_BLUE = "#8FA3B5"

SHARPE_CMAP = LinearSegmentedColormap.from_list("sharpe", ["#E4EAF0", "#8FA3B5", "#1F4E79"])
CORR_CMAP = "RdBu_r"


def _style(fig, ax):
    """Apply the shared report style to one figure."""
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#B8C2CC")
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _title(ax, text):
    ax.set_title(text, loc="left", pad=10, color=TEXT, fontsize=12)


def _legend(ax, **kwargs):
    leg = ax.legend(frameon=False, fontsize=9, **kwargs)
    for text in leg.get_texts():
        text.set_color(TEXT)


def _percent_axis(axis):
    axis.set_major_formatter(PercentFormatter(xmax=1, decimals=None))


def plot_risk_return(results, frontier, asset_stats, perf_sharpe, perf_minvol):
    """Risk-return cloud of random portfolios + efficient frontier + key points."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    _style(fig, ax)

    cloud = ax.scatter(results["Volatility"], results["Return"], c=results["Sharpe"],
                       cmap=SHARPE_CMAP, s=4, alpha=0.6, linewidths=0)
    cbar = fig.colorbar(cloud, ax=ax)
    cbar.set_label("Sharpe ratio", color=MUTED)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_edgecolor(GRID)

    ax.plot(frontier["Volatility"], frontier["Return"], color=NAVY,
            linewidth=1.8, label="Efficient frontier")

    ax.scatter(asset_stats["Annual volatility"], asset_stats["Annual return"],
               color=MUTED, s=18, label="Individual assets", zorder=4)
    for ticker, row in asset_stats.iterrows():
        ax.annotate(ticker, (row["Annual volatility"], row["Annual return"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8.5, color=MUTED)

    ax.scatter(perf_sharpe[1], perf_sharpe[0], marker="o", s=90, color=ACCENT,
               edgecolor=BG, linewidth=1.5, zorder=5, label="Max Sharpe")
    ax.scatter(perf_minvol[1], perf_minvol[0], marker="s", s=80, color=TEXT,
               edgecolor=BG, linewidth=1.5, zorder=5, label="Min volatility")

    ax.set_xlabel("Annual volatility (risk)")
    ax.set_ylabel("Annual expected return")
    _percent_axis(ax.xaxis)
    _percent_axis(ax.yaxis)
    _title(ax, f"Risk and return of {len(results):,} random portfolios")
    _legend(ax, loc="lower right")
    ax.grid(color=GRID, linewidth=0.6)
    fig.tight_layout()
    return fig


def plot_correlation(returns):
    """Heatmap of correlations between assets (low = better diversification)."""
    corr = returns.corr()
    size = max(6, 0.75 * len(corr) + 1)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    _style(fig, ax)
    im = ax.imshow(corr, cmap=CORR_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > 0.6 else TEXT)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_edgecolor(GRID)
    _title(ax, "Correlation of daily returns")
    fig.tight_layout()
    return fig


def plot_weights(weights_by_portfolio):
    """Side-by-side bars of the weights of each optimal portfolio."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _style(fig, ax)
    colors = [NAVY, LIGHT_BLUE, ACCENT]
    n = len(weights_by_portfolio)
    tickers = next(iter(weights_by_portfolio.values())).index
    x = np.arange(len(tickers))
    width = 0.8 / n
    for i, (name, w) in enumerate(weights_by_portfolio.items()):
        ax.bar(x + i * width - 0.4 + width / 2, w.values, width,
               label=name, color=colors[i % len(colors)])
    ax.set_xticks(x, tickers)
    _percent_axis(ax.yaxis)
    ax.set_ylabel("Weight")
    _title(ax, "Optimal allocations")
    _legend(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_cumulative(returns, weights_by_portfolio):
    """Growth of $1 invested in each optimal portfolio vs an equal-weight one."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    _style(fig, ax)
    equal = np.full(returns.shape[1], 1 / returns.shape[1])
    ax.plot((1 + returns @ equal).cumprod(), label="Equal weight",
            color=MUTED, linewidth=1.2, linestyle="--")
    for (name, w), color in zip(weights_by_portfolio.items(), [NAVY, ACCENT]):
        ax.plot((1 + returns @ w.values).cumprod(), label=name, color=color, linewidth=1.6)
    ax.set_ylabel("Value of $1 invested")
    _title(ax, "Growth of $1 (in-sample)")
    _legend(ax)
    ax.grid(color=GRID, linewidth=0.6)
    fig.tight_layout()
    return fig
