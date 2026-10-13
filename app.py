"""Streamlit dashboard. Run with:  streamlit run app.py"""

import os

import pandas as pd
import streamlit as st

from src import charts, config
from src.ai_analyst import ai_commentary, build_prompt, rule_based_summary
from src.data import download_prices
from src.pipeline import run_analysis

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")
st.title("📈 Portfolio Optimization & Risk-Return Dashboard")
st.caption("Modern Portfolio Theory on real market data, with an AI risk review. "
           "Educational project - not investment advice.")

# ---------- Sidebar: user inputs ----------
with st.sidebar:
    st.header("Settings")
    tickers = st.multiselect("Assets", list(config.DEFAULT_TICKERS),
                             default=list(config.DEFAULT_TICKERS),
                             format_func=lambda t: f"{t} - {config.DEFAULT_TICKERS.get(t, t)}")
    extra = st.text_input("Add tickers (comma-separated, Yahoo format)", "")
    tickers += [t.strip().upper() for t in extra.split(",") if t.strip()]

    start = st.date_input("Start date", pd.Timestamp(config.START_DATE))
    rf = st.slider("Risk-free rate", 0.0, 0.08, config.RISK_FREE_RATE, 0.0025, format="%.4f")
    n_portfolios = st.select_slider("Random portfolios", [1_000, 5_000, 10_000, 25_000],
                                    value=config.N_PORTFOLIOS)
    max_weight = st.slider("Max weight per asset", 0.10, 1.0, config.MAX_WEIGHT, 0.05,
                           help="Caps concentration: 0.30 = no asset above 30%.")

if len(tickers) < 2:
    st.warning("Pick at least two assets.")
    st.stop()


@st.cache_data(show_spinner="Downloading prices...")
def load(tickers_tuple, start_date):
    return download_prices(tickers_tuple, start=start_date)


try:
    prices = load(tuple(tickers), str(start))
    r = run_analysis(prices, rf, n_portfolios, max_weight, config.RANDOM_SEED)
except Exception as err:
    st.error(f"Could not run the analysis: {err}")
    st.stop()

# ---------- Headline numbers ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Max-Sharpe return", f"{r['perf_sharpe'][0]:.1%}")
c2.metric("Max-Sharpe volatility", f"{r['perf_sharpe'][1]:.1%}")
c3.metric("Max-Sharpe ratio", f"{r['perf_sharpe'][2]:.2f}")
c4.metric("Min-vol volatility", f"{r['perf_minvol'][1]:.1%}")

weights = {"Max Sharpe": r["w_sharpe"], "Min volatility": r["w_minvol"]}
tab1, tab2, tab3, tab4 = st.tabs(["Risk-return", "Allocations", "Assets", "🤖 AI review"])

with tab1:
    st.pyplot(charts.plot_risk_return(r["random_results"], r["frontier"], r["asset_stats"],
                                      r["perf_sharpe"], r["perf_minvol"]))
    st.markdown("Each dot is a random portfolio. The red curve is the **efficient frontier**: "
                "the lowest risk achievable for each level of return.")

with tab2:
    st.pyplot(charts.plot_weights(weights))
    table = pd.DataFrame(weights)
    st.dataframe(table.style.format("{:.1%}"), width="stretch")
    st.pyplot(charts.plot_cumulative(r["returns"], weights))
    st.info("This backtest is *in-sample*: weights were chosen using the same data, "
            "so past performance looks better than it would have been in reality.")

with tab3:
    st.dataframe(r["asset_stats"].style.format({
        "Annual return": "{:.1%}", "Annual volatility": "{:.1%}",
        "Sharpe ratio": "{:.2f}", "Max drawdown": "{:.1%}"}), width="stretch")
    st.pyplot(charts.plot_correlation(r["returns"]))
    st.line_chart(prices / prices.iloc[0])

with tab4:
    st.write("Claude reads the results and writes a critical risk review.")
    key = st.text_input("Anthropic API key (optional)", type="password",
                        value=os.getenv("ANTHROPIC_API_KEY", ""))
    if st.button("Generate review"):
        prompt = build_prompt(r["asset_stats"], r["w_sharpe"], r["w_minvol"],
                              r["perf_sharpe"], r["perf_minvol"], r["returns"].corr(), rf)
        if key:
            with st.spinner("Claude is analysing the portfolio..."):
                try:
                    st.markdown(ai_commentary(prompt, api_key=key))
                except Exception as err:
                    st.error(f"API error: {err}")
        else:
            st.markdown(rule_based_summary(r["asset_stats"], r["w_sharpe"],
                                           r["perf_sharpe"], r["perf_minvol"]))
            st.caption("No API key: showing the rule-based summary instead.")
