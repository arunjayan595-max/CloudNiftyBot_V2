"""
logic.py (v2.1)

Fixes:
- KeyError when NIFTY dataframe doesn't contain "Close"
- Robustly handles MultiIndex columns from yfinance
- Never crashes trend logic; returns NEUTRAL on bad data

Features:
- Past-date backtesting (generate_signals_for_date)
- Outcome evaluation using 1m candles (evaluate_trade_actuals)
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, date
import pytz

# ---------------- VERSION STAMP (shown in UI) ----------------
LOGIC_VERSION = "v2"

# ---------------- CONFIG ----------------
TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "TCS.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "SBIN.NS", "TITAN.NS"
]

TZ_IST = pytz.timezone("Asia/Kolkata")
DEFAULT_SCAN_TIME_IST = "10:00"


# ---------------- UTIL ----------------
def _to_ist_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure intraday index is timezone-aware IST."""
    if df is None or df.empty:
        return df
    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize("UTC").tz_convert(TZ_IST)
    else:
        df.index = df.index.tz_convert(TZ_IST)
    return df


def _parse_scan_dt_ist(trade_date: date, scan_time_ist: str) -> datetime:
    """Build IST datetime from date + 'HH:MM'."""
    hh, mm = scan_time_ist.split(":")
    return TZ_IST.localize(datetime(trade_date.year, trade_date.month, trade_date.day, int(hh), int(mm), 0))


def _extract_close_series(df: pd.DataFrame) -> pd.Series | None:
    """
    Return a Close-like series from a yfinance dataframe.
    Handles:
    - normal columns: ["Open","High","Low","Close",...]
    - MultiIndex columns: [("Close","^NSEI"), ...]
    - fallback: any column containing "close" (case-insensitive)
    """
    if df is None or df.empty:
        return None

    # Case 1: standard columns
    if "Close" in df.columns:
        return df["Close"]

    # Case 2: MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # try level 0 == "Close"
        try:
            close_df = df.xs("Close", axis=1, level=0, drop_level=False)
            # close_df could have multiple columns; take the first
            return close_df.iloc[:, 0]
        except Exception:
            pass

        # try any column where any level equals "Close"
        for col in df.columns:
            if any(str(level).lower() == "close" for level in col):
                return df[col]

    # Case 3: fallback: find a "close" column by name
    for c in df.columns:
        if "close" in str(c).lower():
            return df[c]

    return None


# ---------------- MARKET TREND ----------------
def get_market_trend(as_of_date: date | None = None) -> str:
    """
    NIFTY trend using 200 EMA of daily closes.
    Never crashes: returns NEUTRAL when data is missing/insufficient.
    """
    nifty = yf.download("^NSEI", period="36mo", interval="1d", progress=False)
