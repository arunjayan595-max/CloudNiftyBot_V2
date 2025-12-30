"""
logic.py (v2)

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
    close = _extract_close_series(nifty)

    if close is None:
        return "NEUTRAL"

    # Build clean frame
    tmp = pd.DataFrame({"Close": pd.to_numeric(close, errors="coerce")}).dropna(subset=["Close"])
    if tmp.empty:
        return "NEUTRAL"

    if as_of_date is not None:
        tmp = tmp[tmp.index.date <= as_of_date]
        if tmp.empty:
            return "NEUTRAL"

    tmp["EMA_200"] = ta.ema(tmp["Close"], length=200)
    tmp = tmp.dropna(subset=["EMA_200"])
    if tmp.empty:
        return "NEUTRAL"

    last_close = float(tmp["Close"].iloc[-1])
    last_ema = float(tmp["EMA_200"].iloc[-1])

    return "BULLISH" if last_close > last_ema else "BEARISH"


# ---------------- SIGNAL GENERATION ----------------
def generate_signals_for_date(trade_date: date, scan_time_ist: str = DEFAULT_SCAN_TIME_IST) -> list[dict]:
    """
    Generate ORB signals for a specific historical date (IST).
    Uses yfinance intraday period-based fetch for reliability.

    Note: yfinance intraday depth is limited; old dates may not be available.
    """
    trend = get_market_trend(as_of_date=trade_date)
    scan_dt_ist = _parse_scan_dt_ist(trade_date, scan_time_ist)

    signals: list[dict] = []

    for ticker in TICKERS:
        df = yf.download(
            ticker,
            period="60d",
            interval="15m",
            progress=False,
            auto_adjust=False
        )

        if df is None or df.empty:
            continue

        df = _to_ist_index(df)

        df_today = df[df.index.date == trade_date].copy()
        if df_today.empty:
            continue

        # ORB: first 15m candle
        first = df_today.iloc[0]
        orb_high = float(first["High"])
        orb_low = float(first["Low"])

        # Entry: last close up to scan time
        df_upto_scan = df_today[df_today.index <= scan_dt_ist]
        if df_upto_scan.empty:
            entry_close = float(df_today.iloc[0]["Close"])
            entry_ts = df_today.index[0]
        else:
            entry_close = float(df_upto_scan.iloc[-1]["Close"])
            entry_ts = df_upto_scan.index[-1]

        signal = None
        sl = None
        target = None
        reason = None

        if trend == "BULLISH" and entry_close > orb_high:
            signal = "BUY"
            sl = orb_low
            target = entry_close + (entry_close - orb_low) * 2
            reason = "ORB Breakout + Bullish Trend"
        elif trend == "BEARISH" and entry_close < orb_low:
            signal = "SELL"
            sl = orb_high
            target = entry_close - (orb_high - entry_close) * 2
            reason = "ORB Breakdown + Bearish Trend"

        if signal:
            signals.append({
                "date": str(trade_date),
                "ticker": ticker.replace(".NS", ""),
                "prediction": signal,
                "entry": round(entry_close, 2),
                "sl": round(float(sl), 2),
                "target": round(float(target), 2),
                "status": "OPEN",
                "actual_result": "WAITING",
                "outcome": "PENDING",
                "trend": trend,
                "reason": reason,
                "scan_time_ist": scan_time_ist,
                "entry_time_ist": entry_ts.strftime("%Y-%m-%d %H:%M:%S%z")
            })

    return signals


def generate_signals() -> list[dict]:
    """Live: generate signals for today."""
    today = datetime.now(TZ_IST).date()
    return generate_signals_for_date(today, DEFAULT_SCAN_TIME_IST)


# ---------------- ACTUAL OUTCOME CHECKER ----------------
def evaluate_trade_actuals(
    ticker_ns: str,
    trade_date: date,
    side: str,
    entry: float,
    sl: float,
    target: float,
    entry_time_ist: datetime | None = None
):
    """
    Evaluate WIN/LOSS/NOT_TRIGGERED using 1m candles after entry time.

    Uses period="7d" for reliability; very old dates may not be available from yfinance.
    """
    df = yf.download(
        ticker_ns,
        period="7d",
        interval="1m",
        progress=False,
        auto_adjust=False
    )

    if df is None or df.empty:
        return "PENDING", "No 1m data available", None, None, None

    df = _to_ist_index(df)

    df_day = df[df.index.date == trade_date].copy()
    if df_day.empty:
        return "PENDING", "No 1m data for selected date", None, None, None

    if entry_time_ist is None:
        entry_time_ist = df_day.index[0]
    else:
        if entry_time_ist.tzinfo is None:
            entry_time_ist = TZ_IST.localize(entry_time_ist)
        else:
            entry_time_ist = entry_time_ist.astimezone(TZ_IST)

    post = df_day[df_day.index >= entry_time_ist]
    if post.empty:
        post = df_day

    post_high = float(post["High"].max())
    post_low = float(post["Low"].min())
    last_close = float(post["Close"].iloc[-1])

    # Worst-case handling when both hit within same 1m candle
    if side == "BUY":
        hit_target = post_high >= target
        hit_sl = post_low <= sl
        if hit_target and hit_sl:
            return "LOSS", f"Both Target({target}) and SL({sl}) touched (1m) → LOSS (worst-case).", post_high, post_low, last_close
        if hit_target:
            return "WIN", f"Hit Target {target}", post_high, post_low, last_close
        if hit_sl:
            return "LOSS", f"Hit SL {sl}", post_high, post_low, last_close
        return "NOT_TRIGGERED", f"Neither Target({target}) nor SL({sl}) hit. Last close {last_close:.2f}", post_high, post_low, last_close

    if side == "SELL":
        hit_target = post_low <= target
        hit_sl = post_high >= sl
        if hit_target and hit_sl:
            return "LOSS", f"Both Target({target}) and SL({sl}) touched (1m) → LOSS (worst-case).", post_high, post_low, last_close
        if hit_target:
            return "WIN", f"Hit Target {target}", post_high, post_low, last_close
        if hit_sl:
            return "LOSS", f"Hit SL {sl}", post_high, post_low, last_close
        return "NOT_TRIGGERED", f"Neither Target({target}) nor SL({sl}) hit. Last close {last_close:.2f}", post_high, post_low, last_close

    return "PENDING", "Unknown side", post_high, post_low, last_close


def check_results(history_df: pd.DataFrame) -> pd.DataFrame:
    """Check SL/Target for today's OPEN trades in the CSV."""
    if history_df is None or history_df.empty:
        return history_df

    today = datetime.now(TZ_IST).date()
    today_str = str(today)

    open_trades = history_df[
        (history_df["status"] == "OPEN") &
        (history_df["date"] == today_str)
    ]

    for idx in open_trades.index:
        ticker_ns = str(history_df.loc[idx, "ticker"]) + ".NS"
        entry = float(history_df.loc[idx, "entry"])
        sl = float(history_df.loc[idx, "sl"])
        target = float(history_df.loc[idx, "target"])
        side = str(history_df.loc[idx, "prediction"])

        entry_time_ist = None
        if "entry_time_ist" in history_df.columns and pd.notna(history_df.loc[idx, "entry_time_ist"]):
            try:
                entry_time_ist = datetime.strptime(
                    str(history_df.loc[idx, "entry_time_ist"]),
                    "%Y-%m-%d %H:%M:%S%z"
                ).astimezone(TZ_IST)
            except Exception:
                entry_time_ist = None

        outcome, action, post_high, post_low, last_close = evaluate_trade_actuals(
            ticker_ns=ticker_ns,
            trade_date=today,
            side=side,
            entry=entry,
            sl=sl,
            target=target,
            entry_time_ist=entry_time_ist
        )

        if outcome in ("WIN", "LOSS"):
            history_df.loc[idx, "status"] = "CLOSED"

        history_df.loc[idx, "actual_result"] = action
        history_df.loc[idx, "outcome"] = outcome

        if post_high is not None:
            history_df.loc[idx, "post_high"] = round(float(post_high), 2)
        if post_low is not None:
            history_df.loc[idx, "post_low"] = round(float(post_low), 2)
        if last_close is not None:
            history_df.loc[idx, "post_close"] = round(float(last_close), 2)

    return history_df
