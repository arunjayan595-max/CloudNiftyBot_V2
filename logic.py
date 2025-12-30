"""
logic.py (v2)

Adds:
- Past-date backtesting support (generate_signals_for_date)
- Robust trend calc (no EMA NaN crash)
- Robust yfinance intraday fetching (period-based for reliability)
- Actual outcome evaluation from 1m data after entry time
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, date, timedelta
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
    """Ensure dataframe index is tz-aware IST."""
    if df is None or df.empty:
        return df
    if getattr(df.index, "tz", None) is None:
        # yfinance intraday is usually UTC; localize then convert
        df.index = df.index.tz_localize("UTC").tz_convert(TZ_IST)
    else:
        df.index = df.index.tz_convert(TZ_IST)
    return df


def _parse_scan_dt_ist(trade_date: date, scan_time_ist: str) -> datetime:
    """Build IST datetime from date + HH:MM string."""
    hh, mm = scan_time_ist.split(":")
    return TZ_IST.localize(datetime(trade_date.year, trade_date.month, trade_date.day, int(hh), int(mm), 0))


# ---------------- MARKET TREND ----------------
def get_market_trend(as_of_date: date | None = None) -> str:
    """
    Check NIFTY trend using 200 EMA.
    FIXED: cannot crash due to EMA NaN. If insufficient history, returns NEUTRAL.
    """
    nifty = yf.download("^NSEI", period="36mo", interval="1d", progress=False)
    if nifty is None or nifty.empty:
        return "NEUTRAL"

    nifty = nifty.dropna(subset=["Close"])
    nifty["EMA_200"] = ta.ema(nifty["Close"], length=200)

    # Use data only up to the chosen historical date (backtesting)
    if as_of_date is not None:
        nifty = nifty[nifty.index.date <= as_of_date]
        if nifty.empty:
            return "NEUTRAL"

    # IMPORTANT: remove NaN EMA rows (happens until 200 bars exist)
    nifty = nifty.dropna(subset=["EMA_200"])
    if nifty.empty:
        return "NEUTRAL"

    last_close = float(nifty["Close"].iloc[-1])
    last_ema = float(nifty["EMA_200"].iloc[-1])

    return "BULLISH" if last_close > last_ema else "BEARISH"


# ---------------- SIGNAL GENERATION ----------------
def generate_signals_for_date(trade_date: date, scan_time_ist: str = DEFAULT_SCAN_TIME_IST) -> list[dict]:
    """
    Generate ORB signals for a specific historical date (IST).

    Data source note:
    - Uses period-based intraday fetch (more reliable than start/end for many users).
    - This means very old dates may not be available in yfinance intraday.
    """
    trend = get_market_trend(as_of_date=trade_date)
    scan_dt_ist = _parse_scan_dt_ist(trade_date, scan_time_ist)

    signals: list[dict] = []

    for ticker in TICKERS:
        df = yf.download(
            ticker,
            period="60d",      # reliable window; older dates may not be present
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

        # ORB candle = first 15m candle of the day
        first = df_today.iloc[0]
        orb_high = float(first["High"])
        orb_low = float(first["Low"])

        # Entry close = last available 15m close up to scan time
        df_upto_scan = df_today[df_today.index <= scan_dt_ist]
        if df_upto_scan.empty:
            close = float(df_today.iloc[0]["Close"])
            entry_ts = df_today.index[0]
        else:
            close = float(df_upto_scan.iloc[-1]["Close"])
            entry_ts = df_upto_scan.index[-1]

        signal = None
        sl = None
        target = None
        reason = None

        # Same logic as your original code, with trend filter
        if trend == "BULLISH" and close > orb_high:
            signal = "BUY"
            sl = orb_low
            target = close + (close - orb_low) * 2
            reason = "ORB Breakout + Bullish Trend"
        elif trend == "BEARISH" and close < orb_low:
            signal = "SELL"
            sl = orb_high
            target = close - (orb_high - close) * 2
            reason = "ORB Breakdown + Bearish Trend"

        if signal:
            signals.append({
                "date": str(trade_date),
                "ticker": ticker.replace(".NS", ""),
                "prediction": signal,
                "entry": round(close, 2),
                "sl": round(float(sl), 2),
                "target": round(float(target), 2),
                "status": "OPEN",
                "actual_result": "WAITING",
                "outcome": "PENDING",

                # extra helpful fields (for UI / comparison)
                "trend": trend,
                "reason": reason,
                "scan_time_ist": scan_time_ist,
                "entry_time_ist": entry_ts.strftime("%Y-%m-%d %H:%M:%S%z")
            })

    return signals


def generate_signals() -> list[dict]:
    """Live mode: generate for today's IST date."""
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
    Fetch 1-minute data and evaluate what happened AFTER entry time.

    Outcome:
    - WIN / LOSS / NOT_TRIGGERED
    - If both SL and Target appear in same 1m candle, we mark LOSS (worst-case)
    """
    df = yf.download(
        ticker_ns,
        period="7d",   # reliable; older days may not exist
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
    """
    Live daily evaluation:
    Checks SL/Target for today's OPEN trades in trade_history.csv.
    """
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
