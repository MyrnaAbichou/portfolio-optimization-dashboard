"""Default settings for the project. Change them here, not inside the code."""

# A diversified "global mix", all quoted in US dollars so returns are comparable
# (mixing EUR and USD prices would silently add currency risk to the analysis).
DEFAULT_TICKERS = {
    "AAPL": "Apple (US tech)",
    "MSFT": "Microsoft (US tech)",
    "JPM": "JPMorgan (US banks)",
    "XOM": "ExxonMobil (US energy)",
    "EWQ": "iShares MSCI France ETF",
    "EWJ": "iShares MSCI Japan ETF",
    "TLT": "iShares 20+ Year US Treasury ETF",
    "GLD": "SPDR Gold ETF",
}

START_DATE = "2019-01-01"
TRADING_DAYS = 252          # trading days per year, used to annualise
RISK_FREE_RATE = 0.03       # 3% per year, roughly the short-term T-bill rate
N_PORTFOLIOS = 10_000       # random portfolios for the Monte Carlo simulation
RANDOM_SEED = 42            # makes the simulation reproducible
MAX_WEIGHT = 1.0            # 1.0 = no cap; e.g. 0.30 forbids >30% in one asset
