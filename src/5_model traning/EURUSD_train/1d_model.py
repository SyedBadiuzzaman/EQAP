import os
import pandas as pd
import joblib
import logging
from typing import Tuple, List
from xgboost import XGBClassifier
from sklearn.metrics import  classification_report

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = r"C:\Users\syedb\Desktop\Equinox Quantitative Analytics and Predictive Platform"
DATA_PATH = os.path.join(BASE_DIR, "data", "features", "EURUSD_1d_features.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, r"models\EUR_USD model", "xgb_1d_model.joblib")

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the feature dataset from a CSV file."""
    if not os.path.exists(file_path):
        logger.error(f"Data file not found at: {file_path}")
        raise FileNotFoundError(f"Missing data file: {file_path}")
    
    logger.info(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame, target_col: str = "target") -> Tuple[pd.DataFrame, pd.Series]:
    """Handles date conversion, sorting, and splitting features/target."""
    logger.info("Preprocessing data...")
    
    # 1. Date Handling
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    
    # 2. Drop Missing Values
    initial_len = len(df)
    df = df.dropna()
    logger.info(f"Dropped {initial_len - len(df)} rows with missing values.")
    
    # 3. Define Features & Target
    feature_cols = [col for col in df.columns if col not in ["Date", target_col]]
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8) -> Tuple:
    """Performs a time-based split."""
    split_idx = int(len(X) * train_ratio)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, X_test, y_train, y_test

def train_xgb_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Trains an XGBClassifier with class balancing."""
    logger.info("Training XGBoost model for 1D timeframe...")
    
    # Calculate scale_pos_weight for imbalanced data
    counts = y_train.value_counts()
    scale_pos_weight = counts[0] / counts[1] if 1 in counts and counts[1] > 0 else 1.0
    logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series, thresholds: List[float] = [0.35, 0.4, 0.45, 0.5]):
    """Evaluates the model across different probability thresholds."""
    logger.info("Evaluating 1D model performance...")
    y_prob = model.predict_proba(X_test)[:, 1]
    
    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        print(f"\n--- Threshold: {t} ---")
        print(classification_report(y_test, y_pred))

def save_model(model: XGBClassifier, path: str):
    """Saves the trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"1D Model successfully saved to: {path}")

def main():
    try:
        # Load
        df = load_data(DATA_PATH)
        
        # Preprocess
        X, y = preprocess_data(df)
        
        # Split
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Train
        model = train_xgb_model(X_train, y_train)
        
        # Evaluate
        evaluate(model, X_test, y_test)
        
        # Save
        save_model(model, MODEL_SAVE_PATH)
        
    except Exception as e:
        logger.exception(f"An error occurred during the 1D training pipeline: {e}")

if __name__ == "__main__":
    main()
