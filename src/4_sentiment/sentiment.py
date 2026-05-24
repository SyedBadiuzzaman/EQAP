import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import nltk
import os
from pathlib import Path

# Download lexicon (runs once)
nltk.download('vader_lexicon')

# ==============================
# 1. Initialize VADER
# ==============================
sia = SentimentIntensityAnalyzer()

# ==============================
# 2. Load CSV
# ==============================
DESKTOP_PATH = Path(os.path.expanduser("~/Desktop"))
PROJECT_ROOT = DESKTOP_PATH / "Equinox Quantitative Analytics and Predictive Platform/data"

OUTPUT = PROJECT_ROOT / "features"
INPUT  = PROJECT_ROOT / "processed_data"
os.makedirs(OUTPUT, exist_ok=True)

def apply_sentiment(files):
    file_path = INPUT / files  # <-- CHANGE THIS
    df = pd.read_csv(file_path)

    # Combine headline + summary
    df["text"] = df["headline"].fillna("") + ". " + df["summary"].fillna("")

    # ==============================
    # 3. Sentiment Function
    # ==============================

    def get_sentiment(text):
        if pd.isna(text) or text.strip() == "":
            return {
                "label": "neutral",
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0
            }

        scores = sia.polarity_scores(text)

        compound = scores["compound"]

        # Standard VADER thresholds
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "compound": compound,
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"]
        }

    # ==============================
    # 4. Apply Sentiment
    # ==============================

    tqdm.pandas()

    results = df["text"].progress_apply(get_sentiment)

    df["sentiment_label"] = results.apply(lambda x: x["label"])
    df["sentiment_compound"] = results.apply(lambda x: x["compound"])
    df["sentiment_positive"] = results.apply(lambda x: x["positive"])
    df["sentiment_negative"] = results.apply(lambda x: x["negative"])
    df["sentiment_neutral"] = results.apply(lambda x: x["neutral"])

    # ==============================
    # 5. Save Output
    # ==============================

    output_path = OUTPUT / files.replace("cleaned.csv", "sentiment.csv")
    df.to_csv(output_path, index=False)

def main():
    TARGET_FILES_cleaned = [
        "XAUUSD_news_cleaned.csv",
        "EURUSD_news_cleaned.csv",
        "BTC_news_cleaned.csv",
        "AAPL_news_cleaned.csv",
    ]
    for i in TARGET_FILES_cleaned:
        apply_sentiment(i)
    
    print(f"Sentiment analysis for  complete.")
    print(f"Saved to: {OUTPUT}")
if __name__ == "__main__":
    main()