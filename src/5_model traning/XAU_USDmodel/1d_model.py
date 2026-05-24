import pandas as pd
from sklearn.metrics import accuracy_score,classification_report
from xgboost import XGBClassifier
import joblib
import os
# Load the dataset
df= pd.read_csv(r"C:\Users\syedb\Desktop\Equinox Quantitative Analytics and Predictive Platform\data\features\XAUUSD_1d_features.csv")


df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

df =df.dropna()

target_col = 'target'
feature_cols = [col for col in df.columns if col not in ['Date', target_col]]
X = df[feature_cols]
y = df[target_col]

#timebased split
split= int(0.8*len(df))
xtrain, xtest = X.iloc[:split], X.iloc[split:]
ytrain, ytest = y.iloc[:split], y.iloc[split:]
ytrain.head()

scale_pos_weights = len(ytrain[ytrain == 0]) / len(ytrain[ytrain == 1])
model = XGBClassifier(  n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weights,
    eval_metric='aucpr')
model.fit(xtrain, ytrain)

yprobs = model.predict_proba(xtest)[:, 1]
highconfidence_preds = (yprobs > 0.6)|(yprobs < 0.4)
#apply filter to predictions and true labels
yprobsfiltered = yprobs[highconfidence_preds]
ytestfiltered = ytest[highconfidence_preds]
y_pred = (yprobsfiltered > 0.5).astype(int)

print("filtered accuracy:")
print(f"{accuracy_score(ytestfiltered, y_pred):.3f}")
print("Classification Report:")
print(classification_report(ytestfiltered, y_pred))




# --- Added by conversion script ---

os.makedirs(r'models/XAU_USDmodel', exist_ok=True)
joblib.dump(model, r'models/XAU_USDmodel/1d_model.pkl')
