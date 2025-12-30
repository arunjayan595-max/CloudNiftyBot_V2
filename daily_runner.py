import pandas as pd
import os
from logic import generate_signals, check_results

CSV_FILE = "trade_history.csv"

COLUMNS = [
    "date", "ticker", "prediction",
    "entry", "sl", "target",
    "status", "actual_result", "outcome",
    # new fields (safe if blank)
    "trend", "reason", "scan_time_ist", "entry_time_ist",
    "post_high", "post_low", "post_close"
]

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[COLUMNS]

# Load or initialize CSV
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
else:
    df = pd.DataFrame(columns=COLUMNS)

df = ensure_columns(df)

# Step 1: Update previous trades (today's OPEN trades)
df = check_results(df)

# Step 2: Generate new signals (today)
signals = generate_signals()

if signals:
    new_df = pd.DataFrame(signals)
    new_df = ensure_columns(new_df)

    # Prevent duplicates: date+ticker
    for _, row in new_df.iterrows():
        exists = not df[
            (df["date"] == row["date"]) &
            (df["ticker"] == row["ticker"])
        ].empty

        if not exists:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

# Step 3: Save
df = ensure_columns(df)
df.to_csv(CSV_FILE, index=False)
print("Trade history updated.")
