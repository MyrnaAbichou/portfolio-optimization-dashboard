"""Step 1 - retrieve and clean historical prices."""

import pandas as pd
import yfinance as yf


def download_prices(tickers, start, end=None):
    """Download daily adjusted closing prices from Yahoo Finance.

    auto_adjust=True corrects prices for dividends and stock splits, so the
    returns we compute later are total returns, not just price moves.
    """
    # threads=False downloads one ticker at a time: slower, but avoids the
    # "database is locked" error of yfinance's cache on some computers.
    raw = yf.download(list(tickers), start=start, end=end,
                      auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        raise ValueError("No data downloaded - check tickers or your connection.")

    prices = raw["Close"]
    if isinstance(prices, pd.Series):          # only one ticker requested
        prices = prices.to_frame(name=list(tickers)[0])
    return clean_prices(prices)


def clean_prices(prices, max_missing=0.05):
    """Remove unusable assets and fill small gaps.

    - Days when most assets have no price are removed. Crypto trades 7 days a
      week but stock markets close at weekends, so weekend rows would
      otherwise make every stock look like it has missing data.
    - An asset missing more than `max_missing` (5%) of the remaining days is
      dropped: it probably did not exist for the whole period, or the ticker
      is wrong.
    - Remaining gaps (e.g. a French holiday when US markets are open) are
      filled with the previous day's price ("forward fill").
    - Leading rows that are still empty are removed.
    The list of dropped assets is kept in prices.attrs["dropped"].
    """
    prices = prices.sort_index()
    prices = prices[prices.notna().mean(axis=1) >= 0.5]
    missing_share = prices.isna().mean()
    dropped = missing_share[missing_share > max_missing].index.tolist()
    if dropped:
        print(f"Dropped (too much missing data): {dropped}")
    prices = prices.drop(columns=dropped).ffill().dropna()
    prices.attrs["dropped"] = dropped
    return prices
