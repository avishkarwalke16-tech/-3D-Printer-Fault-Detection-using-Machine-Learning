import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score

data = pd.read_csv("dataset.csv")
data['Error_found'] = data['Error_found'].map({'no': 0, 'yes': 1})

X = data.drop('Error_found', axis=1)
y = data['Error_found']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# FIX: LogisticRegression in sklearn 1.8+ removed 'multi_class' param
# Use solver='lbfgs' which works cleanly without that attribute
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "SVC": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "NaiveBayes": GaussianNB()
}

results = []
print("Training models...")
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    results.append((name, acc, roc, model))
    print(f"  {name}: Acc={acc:.4f}, ROC-AUC={roc:.4f}")

best = max(results, key=lambda x: x[1])
best_name, best_acc, best_roc, best_model = best
print(f"\nBest: {best_name} | Acc: {best_acc:.4f} | ROC-AUC: {best_roc:.4f}")

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Saved model.pkl and scaler.pkl")
