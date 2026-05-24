import pandas as pd
from sklearn.metrics import  classification_report
from lightgbm import LGBMClassifier

# =========================
# 1. Load Data
# =========================
df = pd.read_csv(r"C:\Users\syedb\Desktop\Equinox Quantitative Analytics and Predictive Platform\data\features\AAPL_1d_features.csv")

# =========================
# 2. Date Handling (IMPORTANT)
# =========================
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# =========================
# 3. Drop Missing Values
# =========================
df = df.dropna()

# =========================
# 4. Define Features & Target
# =========================
target_col = "target" # Replace with your actual target column name
# Automatically select feature columns (avoid hardcoding)
feature_cols = [col for col in df.columns if col not in ["Date", target_col]]

X = df[feature_cols]
y = df[target_col]

# 5. Time-Based Split
# =========================
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# 6. Train Model
# =========================
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

model = LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

model.fit(X_train, y_train)

# Get probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Define confidence filter
high_conf_mask = (y_prob > 0.6) | (y_prob < 0.4)

# Apply filter
y_prob_filtered = y_prob[high_conf_mask]
y_test_filtered = y_test[high_conf_mask]

# Convert to predictions
y_pred_filtered = (y_prob_filtered > 0.5).astype(int)

print("Filtered Results:\n")
print(classification_report(y_test_filtered, y_pred_filtered))




# --- Added by conversion script ---
import joblib
import os
os.makedirs(r'models/Apple_model', exist_ok=True)
joblib.dump(model, r'models/Apple_model/1d_model.pkl')
