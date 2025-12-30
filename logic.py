"""
logic.py (v5)

Improvements:
- End of Day Logic: If Target/SL isn't hit, check if we closed in profit (Green) or loss (Red).
- Keeps v4 Strategy: ORB + RSI + Volume.
- Keeps 20 Nifty Stocks.
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, date
import pytz

# ---------------- VERSION STAMP ----------------
LOGIC_VERSION = "v5 (End-of-Day Logic)"

# ---------------- CONFIG ----------------
TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "TCS.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "SBIN.NS",
    "BHARTIARTL.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "M&M.NS",
    "MARUTI.NS", "TITAN.NS", "SUNPHARMA.NS", "TATASTEEL.NS",
    "ULTRACEMCO.NS", "TATAMOTORS.NS", "NTPC.NS"
]

TZ_IST = pytz.timezone("Asia/Kolkata")
DEFAULT_SCAN_TIME_IST = "10:00"

# Strategy Settings
RSI_PERIOD = 14
RSI_BUY_MIN = 55   
RSI_BUY_MAX = 70   
RSI_SELL_MAX = 45  
RSI_SELL_MIN = 30  
VOL_MA = 20        

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
    hh, mm = scan_time_ist.split(":")
    return TZ_IST.localize(datetime(trade_date.year, trade_date.month, trade_date.day, int(hh), int(mm), 0))

def _get_1d_series(df: pd.DataFrame, col_name: str) -> pd.Series | None:
    if df is None or df.empty: return None
    def to_series(data):
        if isinstance(data, pd.Series): return data
        if isinstance(data, pd.DataFrame): return data.iloc[:, 0] if not data.empty else None
        return None
    col_lower = col_name.lower()
    if col_name in df.columns: return to_series(df[col_name])
    if isinstance(df.columns, pd.MultiIndex):
        try: return to_series(df.xs(col_name, axis=1, level=0, drop_level=True))
        except: pass
    for c in df.columns:
        if isinstance(c, tuple):
            if any(str(part).lower() == col_lower for part in c): return to_series(df[c])
        elif str(c).lower() == col_lower: return to_series(df[c])
    return None

# ---------------- MARKET TREND ----------------
def get_market_trend(as_of_date: date | None = None) -> str:
    nifty = yf.download("^NSEI", period="36mo", interval="1d", progress=False)
    close = _get_1d_series(nifty, "Close")
    if close is None or close.empty: return "NEUTRAL"
    tmp = pd.DataFrame({"Close": pd.to_numeric(close, errors="coerce")}).dropna()
    if as_of_date:
        tmp = tmp[tmp.index.date <= as_of_date]
        if tmp.empty: return "NEUTRAL"
    tmp["EMA_200"] = ta.ema(tmp["Close"], length=200)
    tmp = tmp.dropna(subset=["EMA_200"])
    if tmp.empty: return "NEUTRAL"
    last_close = float(tmp["Close"].iloc[-1])
    last_ema = float(tmp["EMA_200"].iloc[-1])
    return "BULLISH" if last_close > last_ema else "BEARISH"

# ---------------- SIGNAL GENERATION ----------------
def generate_signals_for_date(trade_date: date, scan_time_ist: str = DEFAULT_SCAN_TIME_IST) -> list[dict]:
    trend = get_market_trend(as_of_date=trade_date)
    scan_dt_ist = _parse_scan_dt_ist(trade_date, scan_time_ist)
    
    signals: list[dict] = []

    for ticker in TICKERS:
        try:
            df = yf.download(ticker, period="60d", interval="15m", progress=False, auto_adjust=False)
            if df is None or df.empty: continue
            df = _to_ist_index(df)
            
            c_s = _get_1d_series(df, "Close")
            h_s = _get_1d_series(df, "High")
            l_s = _get_1d_series(df, "Low")
            v_s = _get_1d_series(df, "Volume") 

            if any(x is None for x in [c_s, h_s, l_s, v_s]): continue

            data = pd.DataFrame({"Close": c_s, "High": h_s, "Low": l_s, "Volume": v_s}).dropna()
            data["RSI"] = ta.rsi(data["Close"], length=RSI_PERIOD)
            data["Vol_SMA"] = ta.sma(data["Volume"], length=VOL_MA)

            df_day = data[data.index.date == trade_date].copy()
            if df_day.empty: continue

            first = df_day.iloc[0]
            orb_high, orb_low = float(first["High"]), float(first["Low"])
            
            df_upto_scan = df_day[df_day.index <= scan_dt_ist]
            if df_upto_scan.empty: continue
            
            current_bar = df_upto_scan.iloc[-1]
            curr_close = float(current_bar["Close"])
            curr_rsi = float(current_bar["RSI"]) if pd.notna(current_bar["RSI"]) else 50.0
            curr_vol = float(current_bar["Volume"])
            vol_avg = float(current_bar["Vol_SMA"]) if pd.notna(current_bar["Vol_SMA"]) else 0.0
            entry_ts = df_upto_scan.index[-1]

            signal = None
            reason = []

            if trend == "BULLISH" and curr_close > orb_high:
                if RSI_BUY_MIN <= curr_rsi <= RSI_BUY_MAX:
                    if vol_avg > 0 and curr_vol > (vol_avg * 0.8):
                        signal = "BUY"
                        sl = orb_low
                        target = curr_close + (curr_close - orb_low) * 2
                        reason.append(f"RSI({int(curr_rsi)}) OK")
                        reason.append("Vol OK")

            elif trend == "BEARISH" and curr_close < orb_low:
                if RSI_SELL_MIN <= curr_rsi <= RSI_SELL_MAX:
                    if vol_avg > 0 and curr_vol > (vol_avg * 0.8):
                        signal = "SELL"
                        sl = orb_high
                        target = curr_close - (orb_high - curr_close) * 2
                        reason.append(f"RSI({int(curr_rsi)}) OK")
                        reason.append("Vol OK")

            if signal:
                signals.append({
                    "date": str(trade_date),
                    "ticker": ticker.replace(".NS", ""),
                    "prediction": signal,
                    "entry": round(curr_close, 2),
                    "sl": round(float(sl), 2),
                    "target": round(float(target), 2),
                    "status": "OPEN",
                    "actual_result": "WAITING",
                    "outcome": "PENDING",
                    "trend": trend,
                    "reason": " + ".join(reason),
                    "scan_time_ist": scan_time_ist,
                    "entry_time_ist": entry_ts.strftime("%Y-%m-%d %H:%M:%S%z")
                })
        except Exception:
            continue
    return signals

def generate_signals() -> list[dict]:
    today = datetime.now(TZ_IST).date()
    return generate_signals_for_date(today, DEFAULT_SCAN_TIME_IST)

# ---------------- UPDATED OUTCOME CHECKER ----------------
def evaluate_trade_actuals(ticker_ns, trade_date, side, entry, sl, target, entry_time_ist=None):
    """
    Improved Logic:
    1. Checks if Target or SL was hit.
    2. If neither was hit, checks the Closing Price.
       - If Close > Entry (Buy) = WIN (Profit)
       - If Close < Entry (Buy) = LOSS
    """
    try:
        df = yf.download(ticker_ns, period="7d", interval="1m", progress=False, auto_adjust=False)
        if df is None or df.empty: return "PENDING", "No 1m data", None, None, None
        df = _to_ist_index(df)
        df_day = df[df.index.date == trade_date].copy()
        if df_day.empty: return "PENDING", "Date not in 7d range", None, None, None

        c_s, h_s, l_s = _get_1d_series(df_day, "Close"), _get_1d_series(df_day, "High"), _get_1d_series(df_day, "Low")
        clean = pd.DataFrame({"Close": c_s, "High": h_s, "Low": l_s}).dropna()
        
        if entry_time_ist:
            if entry_time_ist.tzinfo is None: entry_time_ist = TZ_IST.localize(entry_time_ist)
            clean = clean[clean.index >= entry_time_ist]

        if clean.empty: return "PENDING", "No data after entry", None, None, None
        
        ph, pl, lc = float(clean["High"].max()), float(clean["Low"].min()), float(clean["Close"].iloc[-1])

        # --- EVALUATION LOGIC ---
        if side == "BUY":
            # 1. Did we hit Target?
            if ph >= target: 
                return "WIN", f"Target Hit ({target})", ph, pl, lc
            # 2. Did we hit SL?
            if pl <= sl: 
                return "LOSS", f"SL Hit ({sl})", ph, pl, lc
            
            # 3. Neither hit? Check End-of-Day status
            if lc > entry:
                return "WIN", f"Day End Profit (+{round(lc - entry, 2)})", ph, pl, lc
            else:
                return "LOSS", f"Day End Loss ({round(lc - entry, 2)})", ph, pl, lc

        elif side == "SELL":
            if pl <= target: 
                return "WIN", f"Target Hit ({target})", ph, pl, lc
            if ph >= sl: 
                return "LOSS", f"SL Hit ({sl})", ph, pl, lc
            
            if lc < entry:
                return "WIN", f"Day End Profit (+{round(entry - lc, 2)})", ph, pl, lc
            else:
                return "LOSS", f"Day End Loss ({round(entry - lc, 2)})", ph, pl, lc
            
        return "PENDING", "Error", ph, pl, lc
    except Exception:
        return "PENDING", "Eval Crash", None, None, None

def check_results(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df is None or history_df.empty: return history_df
    today = datetime.now(TZ_IST).date()
    open_trades = history_df[(history_df["status"] == "OPEN") & (history_df["date"] == str(today))]
    for idx in open_trades.index:
        t_ns = f"{history_df.loc[idx, 'ticker']}.NS"
        args = {
            "ticker_ns": t_ns, "trade_date": today, "side": str(history_df.loc[idx, "prediction"]),
            "entry": float(history_df.loc[idx, "entry"]), "sl": float(history_df.loc[idx, "sl"]),
            "target": float(history_df.loc[idx, "target"])
        }
        try:
            eti = history_df.loc[idx, "entry_time_ist"]
            if pd.notna(eti): args["entry_time_ist"] = datetime.strptime(str(eti), "%Y-%m-%d %H:%M:%S%z").astimezone(TZ_IST)
        except: pass
        
        outcome, act, ph, pl, lc = evaluate_trade_actuals(**args)
        if outcome in ("WIN", "LOSS"): history_df.loc[idx, "status"] = "CLOSED"
        history_df.loc[idx, "actual_result"] = act
        history_df.loc[idx, "outcome"] = outcome
        if ph: history_df.loc[idx, "post_high"] = ph
        if pl: history_df.loc[idx, "post_low"] = pl
        if lc: history_df.loc[idx, "post_close"] = lc
    return history_df
