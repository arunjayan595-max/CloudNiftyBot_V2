import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, date, timedelta
import pytz

# ---------------- CONFIG ----------------
TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "TCS.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "SBIN.NS", "TITAN.NS"
]

TZ_IST = pytz.timezone("Asia/Kolkata")

# If you run the scan at ~10:00 AM IST, the first 15m candle is usually 09:15-09:30.
# Your original code used "first candle of the day" as ORB candle, we keep that.
# For historical "entry", we take the last available 15m close up to scan_time.
DEFAULT_SCAN_TIME_IST = "10:00"


# ---------------- UTIL ----------------
def _to_ist(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return TZ_IST.localize(dt)
    return dt.astimezone(TZ_IST)

def _parse_scan_dt_ist(trade_date: date, scan_time_ist: str) -> datetime:
    hh, mm = scan_time_ist.split(":")
    dt = datetime(trade_date.year, trade_date.month, trade_date.day, int(hh), int(mm), 0)
    return TZ_IST.localize(dt)

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


# ---------------- MARKET TREND ----------------
def get_market_trend(as_of_date: date | None = None) -> str:
    """
    Check NIFTY trend using 200 EMA.
    If as_of_date is provided, trend is computed using data up to that date (last available daily bar <= as_of_date).
    """
    nifty = yf.download("^NSEI", period="12mo", interval="1d", progress=False)
    if nifty.empty:
        return "NEUTRAL"

    nifty = nifty.dropna()
    nifty["EMA_200"] = ta.ema(nifty["Close"], length=200)

    if as_of_date is not None:
        nifty = nifty[nifty.index.date <= as_of_date]
        if nifty.empty:
            return "NEUTRAL"

    last_close = _safe_float(nifty["Close"].iloc[-1])
    last_ema = _safe_float(nifty["EMA_200"].iloc[-1])

    if pd.isna(last_close) or pd.isna(last_ema):
        return "NEUTRAL"

    return "BULLISH" if last_close > last_ema else "BEARISH"


# ---------------- SIGNAL GENERATION (LIVE + HISTORICAL) ----------------
def generate_signals_for_date(trade_date: date, scan_time_ist: str = DEFAULT_SCAN_TIME_IST):
    """
    Generate ORB-based intraday signals for a specific IST date.

    Logic preserved from your original:
    - ORB candle = first 15m candle of that day
    - Determine signal using trend + price vs ORB levels
    - Entry = last available 15m close up to scan_time_ist (historical analog of "current close")

    Returns:
        list[dict] rows matching your CSV schema + extra fields helpful for comparison UI.
    """
    trend = get_market_trend(as_of_date=trade_date)
    signals = []

    start_dt_ist = TZ_IST.localize(datetime(trade_date.year, trade_date.month, trade_date.day, 0, 0, 0))
    end_dt_ist = start_dt_ist + timedelta(days=1)

    # Download enough intraday data to include that day (15m)
    # yfinance supports "start"/"end" for intraday.
    data = yf.download(
        tickers=TICKERS,
        interval="15m",
        start=start_dt_ist.astimezone(pytz.UTC),
        end=end_dt_ist.astimezone(pytz.UTC),
        group_by="ticker",
        progress=False,
        auto_adjust=False
    )

    scan_dt_ist = _parse_scan_dt_ist(trade_date, scan_time_ist)

    for ticker in TICKERS:
        try:
            df = data[ticker].copy()
        except Exception:
            continue

        if df is None or df.empty:
            continue

        # yfinance returns index tz-aware sometimes; normalize to IST for filtering by date/time
        idx = df.index
        if getattr(idx, "tz", None) is None:
            # assume UTC if naive (yfinance usually returns UTC-aware; but be defensive)
            df.index = df.index.tz_localize("UTC").tz_convert(TZ_IST)
        else:
            df.index = df.index.tz_convert(TZ_IST)

        df_today = df[df.index.date == trade_date].copy()
        if df_today.empty:
            continue

        # ORB = first 15m candle of the day
        first = df_today.iloc[0]
        orb_high = _safe_float(first["High"])
        orb_low = _safe_float(first["Low"])

        # Entry close = last bar close up to scan time
        df_upto_scan = df_today[df_today.index <= scan_dt_ist]
        if df_upto_scan.empty:
            # if scan time is before first bar timestamp, fall back to first close
            close = _safe_float(df_today.iloc[0]["Close"])
            entry_ts = df_today.index[0]
        else:
            close = _safe_float(df_upto_scan.iloc[-1]["Close"])
            entry_ts = df_upto_scan.index[-1]

        if pd.isna(close) or pd.isna(orb_high) or pd.isna(orb_low):
            continue

        signal = None
        sl = None
        target = None
        reason = None

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

                # extras (safe to keep in CSV; app will use them)
                "trend": trend,
                "reason": reason,
                "scan_time_ist": scan_time_ist,
                "entry_time_ist": entry_ts.strftime("%Y-%m-%d %H:%M:%S%z")
            })

    return signals


def generate_signals():
    """Original behavior: generate for today (IST), using default scan time."""
    today = datetime.now(TZ_IST).date()
    return generate_signals_for_date(today, DEFAULT_SCAN_TIME_IST)


# ---------------- ACTUAL OUTCOME CHECKER (LIVE + HISTORICAL) ----------------
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
    Fetch 1m data for the trade date, and evaluate what happened AFTER entry.
    Returns:
        outcome: WIN / LOSS / NOT_TRIGGERED
        actual_result: human readable
        post_high, post_low: for UI
        last_close: last close of day
    """
    start_dt_ist = TZ_IST.localize(datetime(trade_date.year, trade_date.month, trade_date.day, 0, 0, 0))
    end_dt_ist = start_dt_ist + timedelta(days=1)

    df = yf.download(
        ticker_ns,
        interval="1m",
        start=start_dt_ist.astimezone(pytz.UTC),
        end=end_dt_ist.astimezone(pytz.UTC),
        progress=False,
        auto_adjust=False
    )

    if df is None or df.empty:
        return "PENDING", "No 1m data available", None, None, None

    # Normalize index to IST
    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize("UTC").tz_convert(TZ_IST)
    else:
        df.index = df.index.tz_convert(TZ_IST)

    if entry_time_ist is None:
        # default: start of day (not ideal, but keeps it robust)
        entry_time_ist = start_dt_ist

    # Only consider bars at/after entry
    post = df[df.index >= entry_time_ist]
    if post.empty:
        post = df

    post_high = float(post["High"].max())
    post_low = float(post["Low"].min())
    last_close = float(post["Close"].iloc[-1])

    # Determine if target/sl ever hit after entry (order doesn't matter here because
    # we are not simulating exact tick sequence; with 1m OHLC, both could hit same bar.
    # We'll assume worst-case for the trade: if both hit in same minute, mark LOSS.
    outcome = "NOT_TRIGGERED"
    action = f"High={post_high:.2f}, Low={post_low:.2f}, Close={last_close:.2f}"

    if side == "BUY":
        hit_target = post_high >= target
        hit_sl = post_low <= sl
        if hit_target and hit_sl:
            outcome = "LOSS"
            action = f"Both Target({target}) and SL({sl}) touched (1m) → Marked LOSS (worst-case)."
        elif hit_target:
            outcome = "WIN"
            action = f"Hit Target {target}"
        elif hit_sl:
            outcome = "LOSS"
            action = f"Hit SL {sl}"
        else:
            outcome = "NOT_TRIGGERED"
            action = f"Neither Target({target}) nor SL({sl}) hit. Last close {last_close:.2f}"

    elif side == "SELL":
        hit_target = post_low <= target
        hit_sl = post_high >= sl
        if hit_target and hit_sl:
            outcome = "LOSS"
            action = f"Both Target({target}) and SL({sl}) touched (1m) → Marked LOSS (worst-case)."
        elif hit_target:
            outcome = "WIN"
            action = f"Hit Target {target}"
        elif hit_sl:
            outcome = "LOSS"
            action = f"Hit SL {sl}"
        else:
            outcome = "NOT_TRIGGERED"
            action = f"Neither Target({target}) nor SL({sl}) hit. Last close {last_close:.2f}"

    return outcome, action, post_high, post_low, last_close


def check_results(history_df: pd.DataFrame):
    """
    Original behavior: check SL/Target for today's OPEN trades.
    Updated: uses entry_time_ist if present, and writes extra actual metrics.
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
            # entry_time_ist stored like "YYYY-MM-DD HH:MM:SS+0530"
            try:
                entry_time_ist = datetime.strptime(str(history_df.loc[idx, "entry_time_ist"]), "%Y-%m-%d %H:%M:%S%z")
                entry_time_ist = _to_ist(entry_time_ist)
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

        # Close only if decisive
        if outcome in ("WIN", "LOSS"):
            history_df.loc[idx, "status"] = "CLOSED"

        history_df.loc[idx, "actual_result"] = action
        history_df.loc[idx, "outcome"] = outcome

        # store actual metrics for tiles
        if post_high is not None:
            history_df.loc[idx, "post_high"] = round(float(post_high), 2)
        if post_low is not None:
            history_df.loc[idx, "post_low"] = round(float(post_low), 2)
        if last_close is not None:
            history_df.loc[idx, "post_close"] = round(float(last_close), 2)

    return history_df
