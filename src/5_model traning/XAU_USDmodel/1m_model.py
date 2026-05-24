import pandas as pd
from sklearn.metrics import accuracy_score,classification_report
from xgboost import XGBClassifier
import joblib
import os
# Load the dataset
df= pd.read_csv(r"C:\Users\syedb\Desktop\Equinox Quantitative Analytics and Predictive Platform\data\features\XAUUSD_1m_features.csv")


df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

df =df.dropna()

target_col = 'target'
feature_cols = [col for col in df.columns if col not in ['Date', target_col]]
X = df[feature_cols]
y = df[target_col]  

#timebased split
split=int(0.8*len(df))
xtrain, xtest = X.iloc[:split], X.iloc[split:]
ytrain, ytest = y.iloc[:split], y.iloc[split:]
scale_pos_weights = len(ytrain[ytrain == 0]) / len(ytrain[ytrain == 1])

model = XGBClassifier(
    eval_metric="aucpr",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weights,
)
model.fit(xtrain, ytrain)

yprobs = model.predict_proba(xtest)[:, 1]
high_confidence_mask = (yprobs >= 0.6) | (yprobs <= 0.4)
yprobs_filtered = yprobs[high_confidence_mask]
ytest_filtered = ytest[high_confidence_mask]
ypreds= (yprobs_filtered >= 0.3).astype(int)

print("filtered accuracy:")
print(f"{accuracy_score(ytest_filtered, ypreds):.3f}")
print("Classification Report:")
print(classification_report(ytest_filtered, ypreds))




# --- Added by conversion script ---

os.makedirs(r'models/XAU_USDmodel', exist_ok=True)
joblib.dump(model, r'models/XAU_USDmodel/1m_model.pkl')
