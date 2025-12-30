"""
daily_runner.py (v2)

- Ensures schema compatibility (new columns added safely)
- Updates today's OPEN trades with outcomes
- Generates today's signals and appends if not already present
"""

import pandas as pd
import os
from logic import generate_signals, check_results, LOGIC_VERSION

CSV_FILE = "trade_history.csv"

COLUMNS = [
    "date", "ticker", "prediction",
    "entry", "sl", "target",
    "status", "actual_result", "outcome",
    "trend", "reason", "scan_time_ist", "entry_time_ist",
    "post_high", "post_low", "post_close"
]

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[COLUMNS]

if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
else:
    df = pd.DataFrame(columns=COLUMNS)

df = ensure_columns(df)

df = check_results(df)

signals = generate_signals()
if signals:
    new_df = ensure_columns(pd.DataFrame(signals))

    for _, row in new_df.iterrows():
        exists = not df[
            (df["date"] == row["date"]) &
            (df["ticker"] == row["ticker"])
        ].empty
        if not exists:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

df = ensure_columns(df)
df.to_csv(CSV_FILE, index=False)
print(f"Trade history updated ({LOGIC_VERSION}).")
