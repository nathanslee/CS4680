# main.py
# Train + evaluate REGRESSION (Final_Exam_Score) and CLASSIFICATION (Pass_Fail)
# on student_performance_dataset.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# ---------- 1) LOAD ----------
df = pd.read_csv("student_performance_dataset.csv")

# Drop non-predictive ID
if "Student_ID" in df.columns:
    df = df.drop(columns=["Student_ID"])

# Targets
REG_TGT = "Final_Exam_Score"
CLS_TGT = "Pass_Fail"

# Features
feature_cols = [c for c in df.columns if c not in [REG_TGT, CLS_TGT]]
numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

# ---------- 2) PREPROCESS ----------
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols)
])

# Stratify by Pass/Fail for fair split
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[CLS_TGT]
)

X_train = train_df[feature_cols]
X_test  = test_df[feature_cols]

# ---------- 3) REGRESSION (predict Final_Exam_Score) ----------
y_train_reg = train_df[REG_TGT]
y_test_reg  = test_df[REG_TGT]

reg_lin = Pipeline([
    ("prep", preprocess),
    ("model", LinearRegression())
])

reg_rf = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42))
])

reg_lin.fit(X_train, y_train_reg)
reg_rf.fit(X_train, y_train_reg)

pred_lin = reg_lin.predict(X_test)
pred_rf  = reg_rf.predict(X_test)

# Compute RMSE manually (no squared=)
mse_lin = mean_squared_error(y_test_reg, pred_lin)
rmse_lin = np.sqrt(mse_lin)
r2_lin = r2_score(y_test_reg, pred_lin)

mse_rf = mean_squared_error(y_test_reg, pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test_reg, pred_rf)

print("\n=== Regression Metrics (Final_Exam_Score) ===")
print(f"Linear Regression -> MSE: {mse_lin:.3f}  RMSE: {rmse_lin:.3f}  R2: {r2_lin:.3f}")
print(f"Random Forest     -> MSE: {mse_rf:.3f}  RMSE: {rmse_rf:.3f}  R2: {r2_rf:.3f}")

# ---------- 4) CLASSIFICATION (predict Pass/Fail) ----------
label_map = {"Fail": 0, "Pass": 1}
y_train_cls = train_df[CLS_TGT].map(label_map)
y_test_cls  = test_df[CLS_TGT].map(label_map)

cls_log = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=2000, solver="lbfgs"))
])

cls_rf = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42))
])

cls_log.fit(X_train, y_train_cls)
cls_rf.fit(X_train, y_train_cls)

pred_log = cls_log.predict(X_test)
pred_rf_cls = cls_rf.predict(X_test)

# Probabilities for ROC/AUC
prob_log = cls_log.predict_proba(X_test)[:, 1]
prob_rf  = cls_rf.predict_proba(X_test)[:, 1]

def summarize_cls(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    print(f"\n{name}")
    print(f"Accuracy: {acc:.3f}  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}  ROC AUC: {auc:.3f}")
    print(classification_report(y_true, y_pred, target_names=['Fail','Pass'], zero_division=0))
    return f1

print("\n=== Classification Metrics (Pass_Fail) ===")
f1_log = summarize_cls("Logistic Regression", y_test_cls, pred_log, prob_log)
f1_rf  = summarize_cls("Random Forest Classifier", y_test_cls, pred_rf_cls, prob_rf)

# ---------- 5) PLOTS: Confusion Matrix + ROC (best by F1) ----------
best_is_rf = f1_rf >= f1_log
best_name = "Random Forest Classifier" if best_is_rf else "Logistic Regression"
best_pred = pred_rf_cls if best_is_rf else pred_log
best_prob = prob_rf if best_is_rf else prob_log

cm = confusion_matrix(y_test_cls, best_pred)
plt.figure()
plt.imshow(cm, interpolation='nearest')
plt.title(f'Confusion Matrix – {best_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0,1], ["Fail","Pass"])
plt.yticks([0,1], ["Fail","Pass"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y_test_cls, best_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC – {best_name}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title(f'ROC Curve – {best_name}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()
