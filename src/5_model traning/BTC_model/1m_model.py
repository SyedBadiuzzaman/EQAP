import pandas as pd
from sklearn.metrics import accuracy_score ,classification_report
from xgboost import XGBClassifier
import joblib
import os
# =========================
# 1. Load Data
# =========================
df = pd.read_csv(r"C:\Users\syedb\Desktop\Equinox Quantitative Analytics and Predictive Platform\data\features\BTC_1m_features.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df.dropna()


target_val = "target"
feature_cols = [col for col in df.columns if col not in ["Date", target_val]]
X = df[feature_cols]
y = df[target_val]

split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]


scale_pos_weights = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weights,
    eval_metric='aucpr'   
)
model.fit(X_train, y_train)

yprob = model.predict_proba(X_test)[:, 1]
highconfidence_preds = (yprob > 0.6)|(yprob < 0.4)
yprobfiltered = yprob[highconfidence_preds]
ytestfiltered = y_test[highconfidence_preds]
ypred = (yprobfiltered > 0.5).astype(int)

#report
#classification report
print("Classification Report:")
print(classification_report(ytestfiltered, ypred))
print("Accuracy:", accuracy_score(ytestfiltered, ypred))




# --- Added by conversion script ---

os.makedirs(r'models/BTC_model', exist_ok=True)
joblib.dump(model, r'models/BTC_model/1m_model.pkl')
