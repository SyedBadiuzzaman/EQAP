import finnhub
import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv


# ============================
# CONFIGURATION
# ============================

DESKTOP_PATH    = Path(os.path.expanduser("~/Desktop"))
PROJECT_ROOT    = DESKTOP_PATH / "Equinox Quantitative Analytics and Predictive Platform" / "data"
SAVE_PATH       = PROJECT_ROOT / "raw_data"

# FIX 1: point load_dotenv() at the .env FILE, not the folder
load_dotenv(DESKTOP_PATH / "Equinox Quantitative Analytics and Predictive Platform" / ".env")

API_KEY = os.getenv("api_key")
if not API_KEY:
    raise EnvironmentError(
        "❌ API key not found. Make sure 'api_key' is set inside your .env file."
    )

# FIX 3: create output folder if it doesn't exist yet
os.makedirs(SAVE_PATH, exist_ok=True)

# Finnhub client
finnhub_client = finnhub.Client(api_key=API_KEY)

# FIX 2: map each symbol to the correct Finnhub fetch strategy
# - Stocks  → company_news(symbol, from, to)
# - Crypto  → general_news("crypto")
# - Forex   → general_news("forex")
SYMBOLS = {
    "AAPL":   {"type": "stock",  "query": "AAPL"},
    "BTC":    {"type": "crypto", "query": "crypto"},
    "EURUSD": {"type": "forex",  "query": "forex"},
    "XAUUSD": {"type": "forex",  "query": "forex"},
}


# ============================
# FETCH FUNCTION
# ============================

def fetch_and_save(symbol: str, config: dict):
    """
    Fetch news from Finnhub using the correct endpoint per asset type,
    then append results to a CSV file.
    """
    print(f"\nFetching news for {symbol}  (type: {config['type']})…")

    try:
        if config["type"] == "stock":
            # company_news requires a date range — fetch last 30 days
            date_to   = datetime.today().strftime("%Y-%m-%d")
            date_from = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
            data = finnhub_client.company_news(config["query"], _from=date_from, to=date_to)

        else:
            # general_news accepts a category: "general" | "forex" | "crypto" | "merger"
            data = finnhub_client.general_news(config["query"], min_id=0)

    except Exception as exc:
        print(f"  Finnhub API error for {symbol}: {exc}")
        return

    if not data:
        print(f"   No news returned for {symbol}. Skipping.")
        return

    df = pd.DataFrame(data)
    df["symbol"]     = symbol          # tag which asset this row belongs to
    df["fetched_at"] = datetime.utcnow()

    file_path = SAVE_PATH / f"{symbol}_news.csv"

    # Append if file already exists so we accumulate history
    if file_path.exists():
        df.to_csv(file_path, mode="a", index=False, header=False)
    else:
        df.to_csv(file_path, index=False)

    print(f"  Saved {len(df)} articles -> {file_path}")


# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":
    for symbol, config in SYMBOLS.items():
        fetch_and_save(symbol, config)

    print("\ All news fetching complete.")