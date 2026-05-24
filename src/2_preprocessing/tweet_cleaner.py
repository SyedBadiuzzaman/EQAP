import pandas as pd
import re
import os

# Folder configuration
INPUT_FOLDER  = r"C:\Users\user\Desktop\Equinox Quantitative Analytics and Predictive Platform\data\raw_data"
OUTPUT_FOLDER = r"C:\Users\user\Desktop\Equinox Quantitative Analytics and Predictive Platform\data\processed_data"

# Only these four files will be processed
TARGET_FILES = [
    "XAUUSD_news.csv",
    "EURUSD_news.csv",
    "BTC_news.csv",
    "AAPL_news.csv",
]
TARGET_FILES_cleaned = [
    "XAUUSD_news_cleaned.csv",
    "EURUSD_news_cleaned.csv",
    "BTC_news_cleaned.csv",
    "AAPL_news_cleaned.csv",
]

EXPECTED_COLUMNS = [
    "category", "datetime", "headline", "id", "image",
    "related", "source", "summary", "url", "symbol", "fetched_at"
]

def fix_encoding(text):
    """Fix UTF-8 text that was mis-read as latin-1 (mojibake), then clean whitespace."""
    if not isinstance(text, str):
        return text
    try:
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def parse_datetime(val):
    """Convert Unix timestamp (seconds) or any date string to datetime."""
    try:
        ts = float(val)
        if ts > 1_000_000_000:
            return pd.to_datetime(ts, unit='s', utc=True)
    except (ValueError, TypeError):
        pass
    return pd.to_datetime(val, errors='coerce', utc=True)

def clean_dataframe(df):
    # 1. Keep only known columns that actually exist
    cols_present = [c for c in EXPECTED_COLUMNS if c in df.columns]
    df = df[cols_present].copy()

    # 2. Drop fully duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"    Removed {before - len(df)} duplicate rows")

    # 3. Drop rows that are entirely empty
    df.dropna(how='all', inplace=True)

    # 4. Fix datetime column
    if 'datetime' in df.columns:
        df['datetime'] = df['datetime'].apply(parse_datetime)

    # 5. Fix fetched_at column
    if 'fetched_at' in df.columns:
        df['fetched_at'] = pd.to_datetime(df['fetched_at'], errors='coerce', utc=True)

    # 6. Fix encoding in text columns
    text_cols = ['category', 'headline', 'summary', 'source', 'related']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(fix_encoding)

    # 7. Strip whitespace from all remaining string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    # 8. Clean id column (should be integer)
    if 'id' in df.columns:
        df['id'] = pd.to_numeric(df['id'], errors='coerce').astype('Int64')

    # 9. Drop columns that ended up entirely NaN
    df.dropna(axis=1, how='all', inplace=True)

    # 10. Reset index
    df.reset_index(drop=True, inplace=True)

    return df

def read_file(filepath):
    """Try tab-separated first, then comma-separated."""
    for sep in ['\t', ',']:
        try:
            df = pd.read_csv(filepath, sep=sep, encoding='utf-8', on_bad_lines='warn')
            if df.shape[1] > 2:
                return df
        except Exception:
            continue
    raise ValueError(f"Could not parse file: {filepath}")

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"Looking for files in '{INPUT_FOLDER}/'\n")

    for fname,cname in zip(TARGET_FILES , TARGET_FILES_cleaned):
        filepath = os.path.join(INPUT_FOLDER, fname  )

        if not os.path.exists(filepath):
            print(f"  SKIPPED (not found): {fname}\n")
            continue

        print(f"Processing: {fname}")
        try:
            df = read_file(filepath)
            print(f"  Rows before cleaning: {len(df)}")

            df = clean_dataframe(df)
            print(f"  Rows after cleaning: {len(df)}")

            out_path = os.path.join(OUTPUT_FOLDER, cname)
            df.to_csv(out_path, index=False)
            print(f"  Saved to: {out_path}\n")

        except Exception as e:
            print(f"  ERROR processing {fname}: {e}\n")

    print("All done! Cleaned files are in the 'output/' folder.")

if __name__ == "__main__":
    main()