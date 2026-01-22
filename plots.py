import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

# 1) Load data (same files)
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL]

# 2) Split train/val/test
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


# 3) Label encode (fit ONLY on train split)
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

# 4) Train Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    criterion="entropy",
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)

trees = rf.estimators_
classes = rf.classes_


# Helpers functions for weighted voting
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

    # log_loss needs probabilities in same class order
    if y_proba is not None:
        print("Log loss:", log_loss(y_true, y_proba, labels=labels))

    print("\nPer-class report (precision/recall/F1):")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))


# Method 1: uniform voting
y_pred_base = rf.predict(X_test)
acc_base = accuracy_score(y_test, y_pred_base)
cm_base = confusion_matrix(y_test, y_pred_base, labels=classes)
proba_base = rf.predict_proba(X_test)



# Method 2: prediction agreement (on validation)
# diversity(i) = 1 - mean_j!=i agreement(i,j)
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
alpha = 2
w_agree = np.maximum(div_agree, 0.0) + eps
w_agree = w_agree ** alpha
w_agree = w_agree / w_agree.sum()


proba_agree = weighted_predict_proba(trees, w_agree, X_test_np, classes)
y_pred_agree = predict_from_proba(proba_agree, classes)
acc_agree = accuracy_score(y_test, y_pred_agree)
cm_agree = confusion_matrix(y_test, y_pred_agree, labels=classes)


# Method 3: structural similarity

n_features = X_train.shape[1]

def tree_feature_split_counts(tree, n_feat):
    feat = tree.tree_.feature  # -2 are leaves
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
w_struct = w_struct ** alpha
w_struct = w_struct / w_struct.sum()



proba_struct = weighted_predict_proba(trees, w_struct, X_test_np, classes)
y_pred_struct = predict_from_proba(proba_struct, classes)
acc_struct = accuracy_score(y_test, y_pred_struct)
cm_struct = confusion_matrix(y_test, y_pred_struct, labels=classes)




# Print results 
print("\n=== Random Forest Comparison (same split/encoders/test set) ===")
print("Baseline (uniform vote) accuracy:", acc_base)
print("Agreement-diversity weighted accuracy:", acc_agree)
print("Structural-diversity weighted accuracy:", acc_struct)

print("\n--- Confusion Matrices (labels order = rf.classes_) ---")
print("\nBaseline:")
print(cm_base)
print("\nAgreement-diversity weighted:")
print(cm_agree)
print("\nStructural-diversity weighted:")
print(cm_struct)

print("\nClass labels (row/column order):", classes)

print_metrics("Baseline (uniform vote)", y_test, y_pred_base, proba_base, classes)
print_metrics("Agreement-diversity weighted", y_test, y_pred_agree, proba_agree, classes)
print_metrics("Structural-diversity weighted", y_test, y_pred_struct, proba_struct, classes)


# Example 
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


#SHOWING HOW BALANCED ACCURACY CHANGES WHEN DIVERSITY SCORE WEIGHT CHANGES
alphas = [0.2, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20]
bal_acc_struct = []
bal_acc_agree = []

for alpha in alphas:
    # Agreement weights
    w_a = (np.maximum(div_agree, 0.0) + eps) ** alpha
    w_a = w_a / w_a.sum()
    proba_a = weighted_predict_proba(trees, w_a, X_test_np, classes)
    y_a = predict_from_proba(proba_a, classes)
    bal_acc_agree.append(balanced_accuracy_score(y_test, y_a))

    # Structural weights
    w_s = (np.maximum(div_struct, 0.0) + eps) ** alpha
    w_s = w_s / w_s.sum()
    proba_s = weighted_predict_proba(trees, w_s, X_test_np, classes)
    y_s = predict_from_proba(proba_s, classes)
    bal_acc_struct.append(balanced_accuracy_score(y_test, y_s))

# Plot
plt.figure()
plt.plot(alphas, bal_acc_agree, marker='o', label="Agreement-weighted")
plt.plot(alphas, bal_acc_struct, marker='s', label="Structural-weighted")
plt.axhline(balanced_accuracy_score(y_test, y_pred_base), linestyle='--', label="Baseline")
plt.xlabel("Alpha (diversity weight exponent)")
plt.ylabel("Balanced Accuracy")
plt.title("Effect of Diversity Weight Sharpening on Balanced Accuracy")
plt.legend()
plt.grid(True)
plt.show()

def make_weights(div_scores, alpha, eps=1e-12):
    w = (np.maximum(div_scores, 0.0) + eps) ** alpha
    return w / w.sum()

def per_class_recall_f1(y_true, y_pred, labels):
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1  = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return rec, f1


rec_base, f1_base = per_class_recall_f1(y_test, y_pred_base, classes)
bal_base = balanced_accuracy_score(y_test, y_pred_base)

agree_rec_by_alpha = []
agree_f1_by_alpha  = []
agree_bal_by_alpha = []

struct_rec_by_alpha = []
struct_f1_by_alpha  = []
struct_bal_by_alpha = []

for alpha in alphas:
    # Agreement-weighted
    w_a = make_weights(div_agree, alpha, eps)
    proba_a = weighted_predict_proba(trees, w_a, X_test_np, classes)
    y_a = predict_from_proba(proba_a, classes)

    rec_a, f1_a = per_class_recall_f1(y_test, y_a, classes)
    agree_rec_by_alpha.append(rec_a)
    agree_f1_by_alpha.append(f1_a)
    agree_bal_by_alpha.append(balanced_accuracy_score(y_test, y_a))

    # Structural-weighted
    w_s = make_weights(div_struct, alpha, eps)
    proba_s = weighted_predict_proba(trees, w_s, X_test_np, classes)
    y_s = predict_from_proba(proba_s, classes)

    rec_s, f1_s = per_class_recall_f1(y_test, y_s, classes)
    struct_rec_by_alpha.append(rec_s)
    struct_f1_by_alpha.append(f1_s)
    struct_bal_by_alpha.append(balanced_accuracy_score(y_test, y_s))

agree_rec_by_alpha = np.array(agree_rec_by_alpha)  # shape (len(alphas), n_classes)
agree_f1_by_alpha  = np.array(agree_f1_by_alpha)
struct_rec_by_alpha = np.array(struct_rec_by_alpha)
struct_f1_by_alpha  = np.array(struct_f1_by_alpha)

# Plot Recall vs alpha (one plot per class, per method)
for k, cls in enumerate(classes):
    plt.figure()
    plt.plot(alphas, agree_rec_by_alpha[:, k], marker="o", label="Agreement-weighted")
    plt.plot(alphas, struct_rec_by_alpha[:, k], marker="s", label="Structural-weighted")
    plt.axhline(rec_base[k], linestyle="--", label="Baseline")
    plt.xlabel("Alpha (diversity weight exponent)")
    plt.ylabel(f"Recall for class = {cls}")
    plt.title(f"Recall vs Alpha ({cls})")
    plt.grid(True)
    plt.legend()
    plt.show()

# Plot F1 vs alpha (one plot per class, per method)
for k, cls in enumerate(classes):
    plt.figure()
    plt.plot(alphas, agree_f1_by_alpha[:, k], marker="o", label="Agreement-weighted")
    plt.plot(alphas, struct_f1_by_alpha[:, k], marker="s", label="Structural-weighted")
    plt.axhline(f1_base[k], linestyle="--", label="Baseline")
    plt.xlabel("Alpha (diversity weight exponent)")
    plt.ylabel(f"F1 for class = {cls}")
    plt.title(f"F1-score vs Alpha ({cls})")
    plt.grid(True)
    plt.legend()
    plt.show()



alphas_to_show = [0.2, 1.0, 3.0, 10.0]  # edit as desired
tree_idx = np.arange(len(trees))

# Agreement weights: plot weights across tree index for each alpha
plt.figure()
for alpha in alphas_to_show:
    w = make_weights(div_agree, alpha, eps)
    plt.plot(tree_idx, w, marker="o", label=f"alpha={alpha}")
plt.xlabel("Tree index")
plt.ylabel("Weight")
plt.title("Agreement-weighted: Tree weights vs Tree index")
plt.grid(True)
plt.legend()
plt.show()

# Structural weights: plot weights across tree index for each alpha
plt.figure()
for alpha in alphas_to_show:
    w = make_weights(div_struct, alpha, eps)
    plt.plot(tree_idx, w, marker="s", label=f"alpha={alpha}")
plt.xlabel("Tree index")
plt.ylabel("Weight")
plt.title("Structural-weighted: Tree weights vs Tree index")
plt.grid(True)
plt.legend()
plt.show()


# --- Agreement: compute pairwise disagreement matrix on validation preds ---
# val_preds shape: (n_val, n_trees) of hard predictions
n_trees = val_preds.shape[1]
disagree = np.zeros((n_trees, n_trees), dtype=float)
for i in range(n_trees):
    for j in range(n_trees):
        disagree[i, j] = np.mean(val_preds[:, i] != val_preds[:, j])  # disagreement rate

# weighted average pairwise disagreement
def weighted_pairwise_metric(M, w):
    # M is NxN symmetric, w is N
    # compute sum_{i<j} w_i w_j M_ij / sum_{i<j} w_i w_j
    W = np.outer(w, w)
    upper = np.triu_indices(len(w), k=1)
    num = np.sum(W[upper] * M[upper])
    den = np.sum(W[upper])
    return float(num / den) if den > 0 else 0.0

agree_div_metric = []
struct_div_metric = []

n_features = X_train_np.shape[1]

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

cos_dist = np.zeros((n_trees, n_trees), dtype=float)
for i in range(n_trees):
    for j in range(n_trees):
        cos_dist[i, j] = 1.0 - cosine_sim(fingerprints[i], fingerprints[j])  # distance

# compute diversity metric vs balanced accuracy points
for alpha in alphas:
    w_a = make_weights(div_agree, alpha, eps)
    w_s = make_weights(div_struct, alpha, eps)

    agree_div_metric.append(weighted_pairwise_metric(disagree, w_a))
    struct_div_metric.append(weighted_pairwise_metric(cos_dist, w_s))

# Scatter: diversity vs balanced accuracy (agreement)
plt.figure()
plt.scatter(agree_div_metric, agree_bal_by_alpha)
for x, y, a in zip(agree_div_metric, agree_bal_by_alpha, alphas):
    plt.annotate(str(a), (x, y), textcoords="offset points", xytext=(5, 5))
plt.axhline(bal_base, linestyle="--", label="Baseline")
plt.xlabel("Weighted avg pairwise disagreement (validation)")
plt.ylabel("Balanced Accuracy (test)")
plt.title("Agreement diversity vs Balanced Accuracy")
plt.grid(True)
plt.legend()
plt.show()

# Scatter: diversity vs balanced accuracy (structural)
plt.figure()
plt.scatter(struct_div_metric, struct_bal_by_alpha)
for x, y, a in zip(struct_div_metric, struct_bal_by_alpha, alphas):
    plt.annotate(str(a), (x, y), textcoords="offset points", xytext=(5, 5))
plt.axhline(bal_base, linestyle="--", label="Baseline")
plt.xlabel("Weighted avg pairwise structural distance")
plt.ylabel("Balanced Accuracy (test)")
plt.title("Structural diversity vs Balanced Accuracy")
plt.grid(True)
plt.legend()
plt.show()
