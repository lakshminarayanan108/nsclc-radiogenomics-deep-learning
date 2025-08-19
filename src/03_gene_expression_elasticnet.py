import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve



# Load data
# -----------------------------

data = pd.read_csv("/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/data/processed/omics/processed_gene_expr_with_labels.csv")

# Encode gender (if categorical)

X = data.drop(columns=["label", "histology_label", "histology_label_simplified", "patientID"])
y = data["label"]

print("gender" in X.columns)

y.value_counts(normalize=True)


# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Elastic Net Logistic Regression (CV)
# -----------------------------
logreg_cv = LogisticRegressionCV(
    Cs=10,
    cv=5,
    penalty="elasticnet",
    solver="saga",
    l1_ratios=[0.1, 0.5, 0.9],
    max_iter=5000,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42
)

logreg_cv.fit(X_train_scaled, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = logreg_cv.predict(X_test_scaled)
y_prob = logreg_cv.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# -----------------------------
# Confusion Matrix Plot
# -----------------------------
cm = confusion_matrix(y_test, y_pred, normalize="true")

plt.figure(figsize=(6, 5), dpi=150)
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Adenocarcinoma", "Squamous"],
            yticklabels=["Adenocarcinoma", "Squamous"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Elastic Net)")
plt.tight_layout()
plt.savefig("/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/results/figures/elasticnet_confusion.png", dpi=300)
plt.close()

# -----------------------------
# ROC Curve Plot
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 5), dpi=150)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Elastic Net)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/results/figures/elasticnet_roc.png", dpi=300)
plt.close()

# -----------------------------
# Coefficient Importance Plot
# -----------------------------
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": logreg_cv.coef_[0]
})
coef_df["abs_coef"] = np.abs(coef_df["Coefficient"])
top20 = coef_df.sort_values("abs_coef", ascending=False).head(20)

plt.figure(figsize=(8, 6), dpi=150)
sns.barplot(data=top20, x="Coefficient", y="Feature", palette="coolwarm")
plt.title("Top 20 Features (Elastic Net Coefficients)")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/results/figures/elasticnet_top_features.png", dpi=300)
plt.close()

