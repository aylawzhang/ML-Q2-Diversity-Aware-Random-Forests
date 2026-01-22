import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    cohen_kappa_score
)

DATA_PATH = "student-performance-disc.csv"
LABEL_COL = "performance_category"
# LABEL_COL = "pass_fail"

# Load data 
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL]

# Split train/val/test
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y if y.nunique() > 1 else None
)

X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp_raw, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp if y_temp.nunique() > 1 else None
)


X_train = X_train_raw.copy()
X_val = X_val_raw.copy()
X_test = X_test_raw.copy()

encoders = {}
for col in X_train.columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_val[col] = le.transform(X_val[col])
    X_test[col] = le.transform(X_test[col])
    encoders[col] = le

X_train_np = X_train.to_numpy()
X_val_np = X_val.to_numpy()
X_test_np = X_test.to_numpy()

# Train 
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    criterion="entropy",
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)

trees = rf.estimators_
classes = rf.classes_


# Helpers
def weighted_predict_proba(estimators, w, X_in_np, class_labels):
    proba_sum = np.zeros((X_in_np.shape[0], len(class_labels)), dtype=float)
    for est, wt in zip(estimators, w):
        proba_sum += wt * est.predict_proba(X_in_np)
    return proba_sum

def predict_from_proba(proba, class_labels):
    return class_labels[np.argmax(proba, axis=1)]

def print_metrics(name, y_true, y_pred, y_proba, labels):
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Balanced accuracy:", balanced_accuracy_score(y_true, y_pred))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
    print("Weighted F1:", f1_score(y_true, y_pred, average="weighted"))
    print("Macro Precision:", precision_score(y_true, y_pred, average="macro", zero_division=0))
    print("Macro Recall:", recall_score(y_true, y_pred, average="macro", zero_division=0))
    print("Cohen's kappa:", cohen_kappa_score(y_true, y_pred))
    if y_proba is not None:
        print("Log loss:", log_loss(y_true, y_proba, labels=labels))


# uniform voting
y_pred_base = rf.predict(X_test)
acc_base = accuracy_score(y_test, y_pred_base)
cm_base = confusion_matrix(y_test, y_pred_base, labels=classes)
proba_base = rf.predict_proba(X_test)


# prediction agreement
val_preds = np.column_stack([t.predict(X_val_np) for t in trees])  # (n_val, n_trees)
n_trees = len(trees)


sim_agree = np.zeros((n_trees, n_trees), dtype=float)
for i in range(n_trees):
    for j in range(n_trees):
        sim_agree[i, j] = np.mean(val_preds[:, i] == val_preds[:, j])

div_agree = np.zeros(n_trees, dtype=float)
for i in range(n_trees):
    others = [j for j in range(n_trees) if j != i]
    mean_sim = np.mean(sim_agree[i, others]) if others else 1.0
    div_agree[i] = 1.0 - mean_sim

eps = 1e-12
w_agree = np.maximum(div_agree, 0.0) + eps
w_agree = w_agree / w_agree.sum()


proba_agree = weighted_predict_proba(trees, w_agree, X_test_np, classes)
y_pred_agree = predict_from_proba(proba_agree, classes)
acc_agree = accuracy_score(y_test, y_pred_agree)
cm_agree = confusion_matrix(y_test, y_pred_agree, labels=classes)


# structural similarity
n_features = X_train.shape[1]

def tree_feature_split_counts(tree, n_feat):
    feat = tree.tree_.feature 
    counts = np.zeros(n_feat, dtype=float)
    for f in feat:
        if f >= 0:
            counts[f] += 1.0
    s = counts.sum()
    if s > 0:
        counts /= s
    return counts

fingerprints = np.vstack([tree_feature_split_counts(t, n_features) for t in trees])

def cosine_sim(u, v, eps=1e-12):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < eps or nv < eps:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))

sim_struct = np.zeros((n_trees, n_trees), dtype=float)
for i in range(n_trees):
    for j in range(n_trees):
        sim_struct[i, j] = cosine_sim(fingerprints[i], fingerprints[j])

div_struct = np.zeros(n_trees, dtype=float)
for i in range(n_trees):
    others = [j for j in range(n_trees) if j != i]
    mean_sim = np.mean(sim_struct[i, others]) if others else 1.0
    div_struct[i] = 1.0 - mean_sim


w_struct = np.maximum(div_struct, 0.0) + eps
w_struct = w_struct / w_struct.sum()

proba_struct = weighted_predict_proba(trees, w_struct, X_test_np, classes)
y_pred_struct = predict_from_proba(proba_struct, classes)
acc_struct = accuracy_score(y_test, y_pred_struct)
cm_struct = confusion_matrix(y_test, y_pred_struct, labels=classes)




print("\n----Random Forest Comparison-----")
print("\nBaseline (uniform vote) accuracy:", acc_base)
print("Agreement diversity weighted accuracy:", acc_agree)
print("Structural diversity weighted accuracy:", acc_struct)

print("\nConfusion Matrices (labels order: high, low, medium) (columns=predicted, rows=true)")
print("Baseline:")
print(cm_base)
print("Agreement-diversity weighted:")
print(cm_agree)
print("Structural-diversity weighted:")
print(cm_struct)

print_metrics("Baseline (uniform vote)", y_test, y_pred_base, proba_base, classes)
print_metrics("Agreement-diversity weighted", y_test, y_pred_agree, proba_agree, classes)
print_metrics("Structural-diversity weighted", y_test, y_pred_struct, proba_struct, classes)

example = X_test.iloc[0]
true_label = y_test.iloc[0]

pred_base_ex = y_pred_base[0]
pred_agree_ex = y_pred_agree[0]
pred_struct_ex = y_pred_struct[0]

print("\nExample Test Instance:")
print(example.to_dict())
print("True Label:", true_label)
print("Predictions -> Baseline:", pred_base_ex,
      "| Agreement-weighted:", pred_agree_ex,
      "| Structural-weighted:", pred_struct_ex)

