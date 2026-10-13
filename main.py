"""Command-line version: run the full analysis and save the charts.

Usage:
    python main.py            # analysis + charts
    python main.py --ai       # also ask Claude for a written review
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files without opening windows

from src import charts, config
from src.ai_analyst import ai_commentary, build_prompt, rule_based_summary
from src.data import download_prices
from src.pipeline import run_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai", action="store_true", help="ask Claude for a review")
    args = parser.parse_args()

    print("Downloading prices...")
    prices = download_prices(config.DEFAULT_TICKERS, start=config.START_DATE)
    print(f"{prices.shape[1]} assets, {len(prices)} trading days "
          f"({prices.index[0].date()} -> {prices.index[-1].date()})\n")

    r = run_analysis(prices, config.RISK_FREE_RATE, config.N_PORTFOLIOS,
                     config.MAX_WEIGHT, config.RANDOM_SEED)

    print("ASSET STATISTICS")
    table = r["asset_stats"].copy()
    for col in ["Annual return", "Annual volatility", "Max drawdown"]:
        table[col] = table[col].map("{:.1%}".format)
    table["Sharpe ratio"] = table["Sharpe ratio"].map("{:.2f}".format)
    print(table.to_string(), "\n")
    for name, w, perf in [("MAX SHARPE", r["w_sharpe"], r["perf_sharpe"]),
                          ("MIN VOLATILITY", r["w_minvol"], r["perf_minvol"])]:
        print(f"{name}: return {perf[0]:.2%} | volatility {perf[1]:.2%} | Sharpe {perf[2]:.2f}")
        print(w[w > 0.005].sort_values(ascending=False).map("{:.1%}".format).to_string(), "\n")

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    weights = {"Max Sharpe": r["w_sharpe"], "Min volatility": r["w_minvol"]}
    charts.plot_risk_return(r["random_results"], r["frontier"], r["asset_stats"],
                            r["perf_sharpe"], r["perf_minvol"]).savefig(out / "risk_return.png", dpi=150)
    charts.plot_correlation(r["returns"]).savefig(out / "correlation.png", dpi=150)
    charts.plot_weights(weights).savefig(out / "weights.png", dpi=150)
    charts.plot_cumulative(r["returns"], weights).savefig(out / "cumulative.png", dpi=150)
    print(f"Charts saved in {out}/")

    if args.ai:
        prompt = build_prompt(r["asset_stats"], r["w_sharpe"], r["w_minvol"],
                              r["perf_sharpe"], r["perf_minvol"],
                              r["returns"].corr(), config.RISK_FREE_RATE)
        try:
            print("\nAI REVIEW\n" + ai_commentary(prompt))
        except Exception as err:
            print(f"\nAI unavailable ({err}). Rule-based review instead:\n")
            print(rule_based_summary(r["asset_stats"], r["w_sharpe"],
                                     r["perf_sharpe"], r["perf_minvol"]))


if __name__ == "__main__":
    main()
