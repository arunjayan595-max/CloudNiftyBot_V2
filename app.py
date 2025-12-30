"""
app.py (v2)

- Shows logic version stamp in UI
- Screen 1: three tiles (Prediction / Actual / Comparison)
- Past-date backtesting: run strategy for any selected date and compare with reality
- Screen 2: historical archive from trade_history.csv
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
import pytz

import logic
from logic import generate_signals_for_date, evaluate_trade_actuals

TZ_IST = pytz.timezone("Asia/Kolkata")
CSV_FILE = "trade_history.csv"

st.set_page_config(layout="wide", page_title="Nifty Auto Bot v2")

# --- Header with v2 stamps ---
st.title("Nifty Intraday Auto Bot")
st.caption("Cloud automated via GitHub Actions + Past-Date Backtesting (Prediction vs Reality)")
st.write(f"UI version: v2 | logic.py version: {logic.LOGIC_VERSION}")  # <-- prints v2 in UI


def color_outcome(outcome: str) -> str:
    if outcome == "WIN":
        return "#1b5e20"
    if outcome == "LOSS":
        return "#b71c1c"
    if outcome == "NOT_TRIGGERED":
        return "#616161"
    return "#37474f"


def load_history() -> pd.DataFrame:
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame()
    df = pd.read_csv(CSV_FILE)
    if not df.empty and "date" in df.columns:
        df = df.sort_values(["date", "ticker"], ascending=[False, True])
    return df


history_df = load_history()

tab1, tab2 = st.tabs(["Screen 1: Dashboard", "Screen 2: Historical Archive"])

# ---------------- Screen 1 ----------------
with tab1:
    st.subheader("Past-Date Backtesting & Reality Comparison (v2)")

    cA, cB, cC = st.columns([1.2, 1, 1])

    with cA:
        backtest_date = st.date_input(
            "Select date (IST) to run the strategy as if it was that day",
            value=(datetime.now(TZ_IST).date())
        )

    with cB:
        scan_time_ist = st.selectbox(
            "Scan time (IST)",
            options=["09:30", "10:00", "10:15", "11:00", "12:00", "14:00"],
            index=1
        )

    with cC:
        run_btn = st.button("Run Backtest", use_container_width=True)

    st.divider()

    if run_btn:
        with st.spinner("Running historical scan and fetching actual outcomes..."):
            signals = generate_signals_for_date(backtest_date, scan_time_ist=scan_time_ist)

            st.write("Signals generated:", len(signals))  # helpful debug

            if not signals:
                st.info(
                    "No signals generated for this date/time using the current logic. "
                    "This can also happen if intraday data is not available via yfinance for that date."
                )
            else:
                # Compute actuals for each signal (1m)
                rows = []
                for s in signals:
                    ticker_ns = f"{s['ticker']}.NS"
                    side = s["prediction"]
                    entry = float(s["entry"])
                    sl = float(s["sl"])
                    target = float(s["target"])

                    # Parse entry_time_ist string like "YYYY-MM-DD HH:MM:SS+0530"
                    entry_time_ist = None
                    try:
                        entry_time_ist = datetime.strptime(s["entry_time_ist"], "%Y-%m-%d %H:%M:%S%z").astimezone(TZ_IST)
                    except Exception:
                        entry_time_ist = None

                    outcome, action, post_high, post_low, post_close = evaluate_trade_actuals(
                        ticker_ns=ticker_ns,
                        trade_date=backtest_date,
                        side=side,
                        entry=entry,
                        sl=sl,
                        target=target,
                        entry_time_ist=entry_time_ist
                    )

                    s2 = dict(s)
                    s2["outcome"] = outcome
                    s2["actual_result"] = action
                    s2["post_high"] = post_high
                    s2["post_low"] = post_low
                    s2["post_close"] = post_close
                    rows.append(s2)

                bt_df = pd.DataFrame(rows)

                chosen = st.selectbox("Select a trade to view tiles", options=bt_df["ticker"].tolist())
                row = bt_df[bt_df["ticker"] == chosen].iloc[0].to_dict()

                t1, t2, t3 = st.columns(3)

                # Tile 1: Prediction
                with t1:
                    st.markdown("### Prediction")
                    st.metric("Ticker", row["ticker"])
                    st.metric("Side", row["prediction"])
                    st.metric("Entry", row["entry"])
                    st.metric("Target", row["target"])
                    st.metric("SL", row["sl"])
                    st.caption(f"Trend: {row.get('trend', 'NA')} | Reason: {row.get('reason', 'NA')}")
                    st.caption(f"Entry time (IST): {row.get('entry_time_ist', 'NA')}")

                # Tile 2: Actual
                with t2:
                    st.markdown("### Actual Market Outcome")
                    ph = row.get("post_high", None)
                    pl = row.get("post_low", None)
                    pc = row.get("post_close", None)
                    st.metric("Post High", f"{ph:.2f}" if isinstance(ph, (int, float)) and ph is not None else "NA")
                    st.metric("Post Low", f"{pl:.2f}" if isinstance(pl, (int, float)) and pl is not None else "NA")
                    st.metric("Post Close", f"{pc:.2f}" if isinstance(pc, (int, float)) and pc is not None else "NA")
                    st.caption(row.get("actual_result", ""))

                # Tile 3: Comparison
                with t3:
                    outcome = row.get("outcome", "PENDING")
                    bg = color_outcome(outcome)
                    st.markdown(
                        f"""
                        <div style="padding:16px;border-radius:12px;background:{bg};color:white;">
                          <h3 style="margin:0 0 8px 0;">Comparison & Analysis</h3>
                          <div><b>Outcome:</b> {outcome}</div>
                          <div style="margin-top:8px;"><b>Assumption:</b> {row.get('reason', 'NA')}</div>
                          <div><b>Trend:</b> {row.get('trend', 'NA')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.divider()
                st.markdown("### All signals for selected date")
                st.dataframe(
                    bt_df[[
                        "date", "ticker", "prediction", "entry", "sl", "target",
                        "trend", "reason", "outcome", "actual_result"
                    ]],
                    use_container_width=True
                )

    st.divider()
    st.subheader("Live / Journal View (trade_history.csv)")

    if history_df.empty:
        st.warning("No trade_history.csv found yet. Run GitHub Actions once to generate data.")
    else:
        today_str = str(datetime.now(TZ_IST).date())
        today_df = history_df[history_df["date"] == today_str].copy()
        st.dataframe(today_df, use_container_width=True)


# ---------------- Screen 2 ----------------
with tab2:
    st.subheader("Historical Archive (from trade_history.csv)")

    if history_df.empty:
        st.warning("No history available yet.")
    else:
        available_dates = sorted(history_df["date"].dropna().unique().tolist(), reverse=True)
        selected_date_str = st.selectbox("Select a date", options=available_dates)

        day_df = history_df[history_df["date"] == selected_date_str].copy()

        for col in ["entry", "sl", "target", "post_high", "post_low", "post_close"]:
            if col in day_df.columns:
                day_df[col] = pd.to_numeric(day_df[col], errors="coerce")

        def style_outcome_row(r):
            bg = color_outcome(str(r.get("outcome", "")))
            return [f"background-color: {bg}; color: white" if c == "outcome" else "" for c in r.index]

        st.dataframe(
            day_df[[
                "date", "ticker", "prediction", "entry", "sl", "target",
                "trend", "reason",
                "post_high", "post_low", "post_close",
                "outcome", "actual_result"
            ]].style.apply(style_outcome_row, axis=1),
            use_container_width=True
        )
