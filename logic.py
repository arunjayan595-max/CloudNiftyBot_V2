import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, date, timedelta
import pytz

TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "TCS.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "SBIN.NS", "TITAN.NS"
]

TZ_IST = pytz.timezone("Asia/Kolkata")
DEFAULT_SCAN_TIME_IST = "10:00"


def _to_ist_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if getattr(df.index, "tz", None) is None:
        # yfinance intraday is typically UTC; be defensive
        df.index = df.index.tz_localize("UTC").tz_convert(TZ_IST)
    else:
        df.index = df.index.tz_convert(TZ_IST)
    return df


def _parse_scan_dt_ist(trade_date: date, scan_time_ist: str) -> datetime:
    hh, mm = scan_time_ist.split(":")
    return TZ_IST.localize(datetime(trade_date.year, trade_date.month, trade_date.day, int(hh), int(mm), 0))


# ---------------- MARKET TREND ----------------
def get_market_trend(as_of_date: date | None = None) -> str:
    """
    Check NIFTY trend using 200 EMA.
    Safe against insufficient history / NaNs.
    """
    nifty = yf.download("^NSEI", period="24mo", interval="1d", progress=False)
    if nifty is None or nifty.empty:
        return "NEUTRAL"

    nifty = nifty.dropna()

    # Compute EMA_200
    nifty["EMA_200"] = ta.ema(nifty["Close"], length=200)

    # If a date is provided, keep data up to that date
    if as_of_date is not None:
        nifty = nifty[nifty.index.date <= as_of_date]
        if nifty.empty:
            return "NEUTRAL"

    # Drop rows where EMA is NaN (not enough history)
    nifty = nifty.dropna(subset=["EMA_200"])
    if nifty.empty:
        return "NEUTRAL"

    last_close = float(nifty["Close"].iloc[-1])
    last_ema = float(nifty["EMA_200"].iloc[-1])

    return "BULLISH" if last_close > last_ema else "BEARISH"


# ---------------- SIGNAL GENERATION ----------------
def generate_signals_for_date(trade_date: date, scan_time_ist: str = DEFAULT_SCAN_TIME_IST):
    trend = get_market_trend(as_of_date=trade_date)
    signals = []

    scan_dt_ist = _parse_scan_dt_ist(trade_date, scan_time_ist)

    # IMPORTANT: fetch per-ticker with a longer period (more reliable than start/end for intraday)
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

        # ORB = first 15m candle
        first = df_today.iloc[0]
        orb_high = float(first["High"])
        orb_low = float(first["Low"])

        # Entry close = last 15m close up to scan time
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
                "trend": trend,
                "reason": reason,
                "scan_time_ist": scan_time_ist,
                "entry_time_ist": entry_ts.strftime("%Y-%m-%d %H:%M:%S%z")
            })

    return signals


def generate_signals():
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
    # Fetch per-ticker 1m data with period to increase reliability
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


def check_results(history_df: pd.DataFrame):
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
                entry_time_ist = datetime.strptime(str(history_df.loc[idx, "entry_time_ist"]), "%Y-%m-%d %H:%M:%S%z")
                entry_time_ist = entry_time_ist.astimezone(TZ_IST)
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
