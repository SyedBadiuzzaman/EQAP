import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta


# ============================
# CONFIGURATION
# ============================

TICKERS = {
    "Apple":   "AAPL",
    "Bitcoin": "BTC-USD",
    "Gold":    "GC=F",
    "EUR/USD": "EURUSD=X"
}

INTERVALS_TO_FETCH = ["1m","1h","1d"]

PROJECT_RELATIVE_PATH = "Desktop/Equinox Quantitative Analytics and Predictive Platform/data"
HOME_DIR = os.path.expanduser("~")
BASE_DIR = os.path.join(HOME_DIR, PROJECT_RELATIVE_PATH)
OUTPUT_DIR = "raw_data"
FULL_OUTPUT_PATH = os.path.join(BASE_DIR, OUTPUT_DIR)


# ============================
# DATA FETCH FUNCTION
# ============================

def fetch_data(ticker: str, interval: str) -> pd.DataFrame:
    """
    Fetch OHLCV data for a given ticker and interval.

    Supports:
        - Minute intervals : "1m", "2m", "5m", "15m", "30m"
        - Hourly intervals : "60m", "1h"
        - 4-Hour interval  : "4h"  (fetched as 1h then resampled)
        - Daily interval   : "1d"

    Yahoo Finance limits intraday history (e.g. 7 days for 1m),
    so the lookback window is capped accordingly.
    """

    # Maximum lookback (days) allowed by Yahoo Finance per interval
    interval_limits = {
        "1m":  7,
        "1h":  365,
        "1d":  3650,
    }

    # 4h is not a native yfinance interval — fetch 1h and resample
    max_days = interval_limits.get(interval, 365)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=max_days)

    print(f"  Fetching {interval} data for {ticker} "
          f"from {start_date.date()} to {end_date.date()}")

    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
        auto_adjust=True
    )

    # Guard: yfinance can return None for unsupported ticker/interval combos
    if df is None:
        print(f"  [WARN] yfinance returned None for {ticker} @ {interval}. Skipping.")
        return pd.DataFrame()

    # Flatten multi-level columns (yfinance ≥ 0.2.x returns MultiIndex headers)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    
    return df


# ============================
# DOWNLOAD & SAVE ALL DATA
# ============================

def download_all_data():
    """
    Loops through every ticker × interval combination,
    fetches OHLCV data, and saves each result as a CSV file
    inside the configured output directory.
    """
    os.makedirs(FULL_OUTPUT_PATH, exist_ok=True)
    print(f"Output directory: {FULL_OUTPUT_PATH}\n")

    for asset_name, ticker_symbol in TICKERS.items():
        print(f"=== {asset_name} ({ticker_symbol}) ===")

        for interval in INTERVALS_TO_FETCH:
            try:
                df = fetch_data(ticker_symbol, interval)

                if df.empty:
                    print(f"  [SKIP] No data returned for {ticker_symbol} @ {interval}")
                    continue

                # Build a clean filename, stripping Yahoo suffix characters
                clean_symbol = ticker_symbol.replace("=X", "").replace("=F", "")
                filename = f"{clean_symbol}_{interval}.csv"
                filepath = os.path.join(FULL_OUTPUT_PATH, filename)

                df.to_csv(filepath)
                print(f"  [OK]   Saved -> {filepath}")

            except Exception as exc:
                print(f"  [ERROR] {ticker_symbol} @ {interval}: {exc}")

        print()

    print("All data fetching complete.")


# ============================
# QUICK PREVIEW (optional)
# ============================

def preview_data(interval: str = "1h"):
    """Print the first few rows for every ticker at a given interval."""
    print(f"\n--- Preview ({interval}) ---")
    for asset_name, ticker_symbol in TICKERS.items():
        df = fetch_data(ticker_symbol, interval)
        print(f"\n{asset_name}:\n{df.head()}")


# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":
    download_all_data()
    # Uncomment to preview data after downloading:
    # preview_data("1h")