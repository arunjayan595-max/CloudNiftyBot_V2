"""
logic.py (v3)

Fixes:
- TypeError in get_market_trend (forces DataFrame -> Series conversion)
- Robust handling of yfinance MultiIndex returns
- Prevents crashes on empty or malformed data
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, date
import pytz

# ---------------- VERSION STAMP ----------------
LOGIC_VERSION = "v3"

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
    Robustly extract a 1D Close Series from a yfinance DataFrame.
    """
    if df is None or df.empty:
        return None

    # 1. Check for MultiIndex columns (common in yfinance > 0.2)
    if isinstance(df.columns, pd.MultiIndex):
        # Try to find "Close" in the top level
        try:
            # xs returns a DataFrame if drop_level=False, or if multiple columns match
            # We use drop_level=True to try and get a simplified frame/series
            close_data = df.xs("Close", axis=1, level=0, drop_level=True)
            
            # If result is DataFrame, take first column
            if isinstance(close_data, pd.DataFrame):
                return close_data.iloc[:, 0]
            return close_data
        except Exception:
            pass
            
        # Fallback: Check if any level contains "Close"
        for col in df.columns:
            # col is a tuple like ('Close', 'RELIANCE.NS')
            if any(str(x).lower() == "close" for x in col):
                data = df[col]
                if isinstance(data, pd.DataFrame):
                    return data.iloc[:, 0]
                return data

    # 2. Check for standard flat columns
    if "Close" in df.columns:
        data = df["Close"]
        if isinstance(data, pd.DataFrame):
            return data.iloc[:, 0]
        return data

    # 3. Case-insensitive fallback
    for c in df.columns:
        if "close" in str(c).lower():
            data = df[c]
            if isinstance(data, pd.DataFrame):
                return data.iloc[:, 0]
            return data

    return None


# ---------------- MARKET TREND ----------------
def get_market_trend(as_of_date: date | None = None) -> str:
    """
    NIFTY trend using 200 EMA of daily closes.
    """
    # Download NIFTY data
    nifty = yf.download("^NSEI", period="36mo", interval="1d", progress=False)
    
    # Extract Close column
    close = _extract_close_series(nifty)

    # SAFETY CHECK: Ensure we have data
    if close is None:
        return "NEUTRAL"
    
    # SAFETY CHECK: If it is still a DataFrame (rare), force it to Series
    if isinstance(close, pd.DataFrame):
        if close.empty:
            return "NEUTRAL"
        close = close.iloc[:, 0]

    # Convert to numeric, coercing errors
    close = pd.to_numeric(close, errors="coerce")
    
    # Drop NaNs
    tmp = pd.DataFrame({"Close": close}).dropna(subset=["Close"])
    
    if tmp.empty:
        return "NEUTRAL"

    # Filter by date if provided
    if as_of_date is not None:
        tmp = tmp[tmp.index.date <= as_of_date]
        if tmp.empty:
            return "NEUTRAL"

    # Calculate EMA
    tmp["EMA_200"] = ta.ema(tmp["Close"], length=200)
    tmp = tmp.dropna(subset=["EMA_200"])
    
    if tmp.empty:
        # Not enough data for 200 EMA
        return "NEUTRAL"

    last_close = float(tmp["Close"].iloc[-1])
    last_ema = float(tmp["EMA_200"].iloc[-1])

    return "BULLISH" if last_close > last_ema else "BEARISH"


# ---------------- SIGNAL GENERATION ----------------
def generate_signals_for_date(trade_date: date, scan_time_ist: str = DEFAULT_SCAN_TIME_IST) -> list[dict]:
    """
    Generate ORB signals for a specific historical date.
    """
    trend = get_market_trend(as_of_date=trade_date)
    scan_dt_ist = _parse_scan_dt_ist(trade_date, scan_time_ist)

    signals: list[dict] = []

    for ticker in TICKERS:
        try:
            df = yf.download(
                ticker,
                period="60d",
                interval="15m",
                progress=False,
                auto_adjust=False
            )

            if df is None or df.empty:
                continue

            # Standardize Index
            df = _to_ist_index(df)

            # Filter for specific date
            df_today = df[df.index.date == trade_date].copy()
            if df_today.empty:
                continue

            # Handle MultiIndex for High/Low/Close extraction
            # We create a temporary flat dataframe for easier logic
            flat_df = pd.DataFrame(index=df_today.index)
            
            # Helper to safely grab columns
            c_series = _extract_close_series(df_today)
            if c_series is None: continue
            flat_df["Close"] = c_series

            # Extract High (similar logic to Close)
            if "High" in df_today.columns:
                h = df_today["High"]
                flat_df["High"] = h.iloc[:, 0] if isinstance(h, pd.DataFrame) else h
            elif isinstance(df_today.columns, pd.MultiIndex):
                try: flat_df["High"] = df_today.xs("High", axis=1, level=0, drop_level=True)
                except: flat_df["High"] = flat_df["Close"] # Fallback
            
            # Extract Low
            if "Low" in df_today.columns:
                l = df_today["Low"]
                flat_df["Low"] = l.iloc[:, 0] if isinstance(l, pd.DataFrame) else l
            elif isinstance(df_today.columns, pd.MultiIndex):
                try: flat_df["Low"] = df_today.xs("Low", axis=1, level=0, drop_level=True)
                except: flat_df["Low"] = flat_df["Close"] # Fallback

            # Ensure numeric
            flat_df = flat_df.apply(pd.to_numeric, errors='coerce').dropna()
            if flat_df.empty: continue

            # ORB Logic
            first = flat_df.iloc[0]
            orb_high = float(first["High"])
            orb_low = float(first["Low"])

            # Entry Logic
            df_upto_scan = flat_df[flat_df.index <= scan_dt_ist]
            if df_upto_scan.empty:
                entry_close = float(flat_df.iloc[0]["Close"])
                entry_ts = flat_df.index[0]
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
        except Exception:
            continue

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
    Evaluate WIN/LOSS/NOT_TRIGGERED using 1m candles.
    """
    try:
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

        # Build clean 1m data
        clean_df = pd.DataFrame(index=df_day.index)
        
        # High
        if isinstance(df_day.columns, pd.MultiIndex):
            try: clean_df["High"] = df_day.xs("High", axis=1, level=0, drop_level=True).iloc[:, 0]
            except: clean_df["High"] = df_day.iloc[:, 0] # Fallback
            try: clean_df["Low"] = df_day.xs("Low", axis=1, level=0, drop_level=True).iloc[:, 0]
            except: clean_df["Low"] = df_day.iloc[:, 0]
            try: clean_df["Close"] = df_day.xs("Close", axis=1, level=0, drop_level=True).iloc[:, 0]
            except: clean_df["Close"] = df_day.iloc[:, 0]
        else:
            clean_df["High"] = df_day["High"]
            clean_df["Low"] = df_day["Low"]
            clean_df["Close"] = df_day["Close"]

        # Handle time
        if entry_time_ist is None:
            entry_time_ist = df_day.index[0]
        else:
            if entry_time_ist.tzinfo is None:
                entry_time_ist = TZ_IST.localize(entry_time_ist)
            else:
                entry_time_ist = entry_time_ist.astimezone(TZ_IST)

        post = clean_df[clean_df.index >= entry_time_ist]
        if post.empty:
            post = clean_df

        post_high = float(post["High"].max())
        post_low = float(post["Low"].min())
        last_close = float(post["Close"].iloc[-1])

        # Evaluation Logic
        if side == "BUY":
            hit_target = post_high >= target
            hit_sl = post_low <= sl
            if hit_target and hit_sl:
                return "LOSS", f"Both Target({target}) and SL({sl}) touched (1m) → LOSS (worst-case).", post_high, post_low, last_close
            if hit_target:
                return "WIN", f"Hit Target {target}", post_high, post_low, last_close
            if hit_sl:
                return "LOSS", f"Hit SL {sl}", post_high, post_low, last_close
            return "NOT_TRIGGERED", f"Neither hit. Last close {last_close:.2f}", post_high, post_low, last_close

        if side == "SELL":
            hit_target = post_low <= target
            hit_sl = post_high >= sl
            if hit_target and hit_sl:
                return "LOSS", f"Both Target({target}) and SL({sl}) touched (1m) → LOSS (worst-case).", post_high, post_low, last_close
            if hit_target:
                return "WIN", f"Hit Target {target}", post_high, post_low, last_close
            if hit_sl:
                return "LOSS", f"Hit SL {sl}", post_high, post_low, last_close
            return "NOT_TRIGGERED", f"Neither hit. Last close {last_close:.2f}", post_high, post_low, last_close

        return "PENDING", "Unknown side", post_high, post_low, last_close
        
    except Exception as e:
        return "PENDING", f"Error in eval: {str(e)}", None, None, None


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
