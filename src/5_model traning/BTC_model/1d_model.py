import pandas as pd
from sklearn.metrics import accuracy_score ,classification_report
from xgboost import XGBClassifier
import joblib
import os
# =========================
# 1. Load Data
# =========================
df = pd.read_csv(r"C:\Users\syedb\Desktop\Equinox Quantitative Analytics and Predictive Platform\data\features\BTC_1d_features.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)


df =df.dropna()

target_col = 'target'
feature_cols = [col for col in df.columns if col not in ['Date', target_col]]
X = df[feature_cols]
y = df[target_col]


split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]


scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr'   
)
model.fit(X_train, y_train)

yprobs = model.predict_proba(X_test)[:, 1]
highconfidence_preds = (yprobs > 0.6)|(yprobs < 0.4)
#apply filter to predictions and true labels
yprobsfiltered = yprobs[highconfidence_preds]
ytestfiltered = y_test[highconfidence_preds]
y_pred = (yprobsfiltered > 0.5).astype(int)

print("filtered accuracy:")
print(f"{accuracy_score(ytestfiltered, y_pred):.3f}")
print("Classification Report:")
print(classification_report(ytestfiltered, y_pred))




# --- Added by conversion script ---

os.makedirs(r'models/BTC_model', exist_ok=True)
joblib.dump(model, r'models/BTC_model/1d_model.pkl')
