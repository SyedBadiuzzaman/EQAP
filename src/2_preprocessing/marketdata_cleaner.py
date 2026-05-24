import pandas as pd
import os
from pathlib import Path


# ============================
# CONFIGURATION
# ============================

DESKTOP_PATH = Path(os.path.expanduser("~/Desktop"))
PROJECT_ROOT = DESKTOP_PATH / "Equinox Quantitative Analytics and Predictive Platform/data"

RAW_FOLDER   = PROJECT_ROOT / "raw_data"
CLEAN_FOLDER = PROJECT_ROOT / "processed_data"

os.makedirs(CLEAN_FOLDER, exist_ok=True)

# Each interval maps to its raw filenames per symbol
INTERVAL_FILES = {
    "1d": {
        "AAPL":   "AAPL_1d.csv",
        "BTC":    "BTC-USD_1d.csv",
        "EURUSD": "EURUSD_1d.csv",
        "XAUUSD": "GC_1d.csv",
    },
    "1h": {
        "AAPL":   "AAPL_1h.csv",
        "BTC":    "BTC-USD_1h.csv",
        "EURUSD": "EURUSD_1h.csv",
        "XAUUSD": "GC_1h.csv",
    },
    "1m": {
        "AAPL":   "AAPL_1m.csv",
        "BTC":    "BTC-USD_1m.csv",
        "EURUSD": "EURUSD_1m.csv",
        "XAUUSD": "GC_1m.csv",
    },
}

CORRECT_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Patterns that indicate a spurious header row injected by yfinance
HEADER_NOISE_PATTERN = "Price|Ticker|AAPL|BTC|Time|GC|EURUSD|XAUUSD"


# ============================
# SHARED CLEAN FUNCTION
# ============================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Universal cleaner for daily, hourly, and minute OHLCV data.

    Steps:
        1. Drop fully-empty rows.
        2. Remove spurious header rows sometimes injected by yfinance.
        3. Normalise column names and keep the first 6 columns.
        4. Parse the Date column (preserves full timestamp precision).
        5. Sort by datetime, drop duplicates, forward/back-fill gaps.
    """
    # 1. Drop fully-empty rows
    df = df.dropna(how="all")

    if df.empty:
        return df

    # 2. Remove spurious header rows
    if df.iloc[0].astype(str).str.contains(HEADER_NOISE_PATTERN).any():
        df = df.iloc[1:].reset_index(drop=True)

    # 3. Normalise column names
    df.columns = df.columns.str.strip()

    if len(df.columns) >= 6:
        df = df.iloc[:, :6].copy()
        df.columns = CORRECT_COLUMNS

    # 4. Parse Date — keeps full H:M:S precision for intraday data
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # 5. Sort, deduplicate, fill gaps
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.ffill().bfill()

    return df


# ============================
# PROCESS ONE INTERVAL
# ============================

def process_interval(interval: str, file_names: dict) -> dict:
    """
    Load, clean, and save all symbols for a given interval.
    Returns a dict of { symbol: cleaned_DataFrame }.
    """
    processed = {}

    label = {
        "1d": "Daily",
        "1h": "Hourly",
        "1m": "1-Minute",
    }.get(interval, interval.upper())

    print(f"\n{'='*80}")
    print(f"  Starting {label} Data Preprocessing")
    print(f"{'='*80}")

    for symbol, filename in file_names.items():
        print(f"\n  Processing -> {symbol}  ({filename})")

        file_path = RAW_FOLDER / filename

        if not file_path.exists():
            print(f"  File not found: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            print(f"  Error loading {filename}: {exc}")
            continue

        df = clean_dataframe(df)

        if df.empty:
            print(f"   No valid rows after cleaning — skipping {symbol} {interval}")
            continue

        save_path = CLEAN_FOLDER / f"{symbol}_{interval}_cleaned.csv"
        df.to_csv(save_path, index=False)
        processed[symbol] = df

        print(f"  Saved -> {save_path}  ({len(df):,} rows)")

    print(f"\n{'='*80}")
    print(f"  {label} Preprocessing Complete — files saved in: processed_data/")
    print(f"{'='*80}")

    return processed


# ============================
# RUN ALL INTERVALS
# ============================

def preprocess_all() -> dict:
    """
    Run preprocessing for every configured interval and return all results.
    """
    all_results = {}

    for interval, file_names in INTERVAL_FILES.items():
        all_results[interval] = process_interval(interval, file_names)

    print("\n All intervals preprocessed successfully.\n")
    return all_results


# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":
    results = preprocess_all()

    # Quick summary table
    print(f"\n{'='*80}")
    print("  Summary")
    print(f"{'='*80}")
    for interval, symbols in results.items():
        for symbol, df in symbols.items():
            print(f"  {symbol:8s} | {interval:3s} | {len(df):>7,} rows")
    print(f"{'='*80}\n")