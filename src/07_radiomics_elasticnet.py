
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve


def simplify_histology(label):
    if "Adenocarcinoma" in label:
        return "Adenocarcinoma"
    elif "Squamous Cell Carcinoma" in label:
        return "Squamous Cell Carcinoma"
    else:
        return None 

# Load data
# -----------

radiomics_features_file_path = '/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/results/radiomics_features_dedup.csv'

meta_data_dir = '/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/data/raw/metadata/'

meta_data_path = meta_data_dir + 'Lung3.metadata.csv'
metadata_df = pd.read_csv(meta_data_path)

# keep only sample.name, gender, histology

meta_df = metadata_df[["sample.name", "characteristics.tag.gender", "characteristics.tag.histology"]]

# rename columns for clarity

meta_df = meta_df.rename(columns={
    "sample.name": "PatientID",
    "characteristics.tag.gender": "gender",
    "characteristics.tag.histology": "histology_label"
})

meta_df["histology_label_simplified"] = meta_df["histology_label"].apply(simplify_histology)

meta_df = meta_df[meta_df["histology_label_simplified"].notna()]

# gender encoding
meta_df["gender"] = meta_df["gender"].map({"M": 0, "F": 1})

# label encoding

label_map = {"Adenocarcinoma": 0, "Squamous Cell Carcinoma": 1}
meta_df["label"] = meta_df["histology_label_simplified"].map(label_map)

# --- Step 1: Load data ---
X = pd.read_csv(radiomics_features_file_path)          # radiomics features
labels_df = meta_df.copy(deep=True)           # labels dataframe

# Merge on PatientID
merged_df = X.merge(labels_df, on="PatientID", how="inner")


# Drop identifiers, keep features

X = merged_df.drop(columns=["PatientID", "label", "histology_label", "histology_label_simplified"])

y = merged_df["label"]

print("X shape:", X.shape)

print("y distribution:\n", y.value_counts(normalize=True))


# Train-test split
# -----------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# Scale features
# ---------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# Elastic Net Logistic Regression (CV)
# -------------------------------------

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


# Predictions
# ------------

y_pred = logreg_cv.predict(X_test_scaled)

y_prob = logreg_cv.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Confusion Matrix Plot
# ----------------------

cm = confusion_matrix(y_test, y_pred, normalize="true")

plt.figure(figsize=(6, 5), dpi=300)

sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Adenocarcinoma", "Squamous"],
            yticklabels=["Adenocarcinoma", "Squamous"])

plt.xlabel("Predicted")

plt.ylabel("True")

plt.title("Confusion Matrix (Elastic Net Radiomics)")

plt.tight_layout()

plt.savefig("/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/results/figures/radiomics_elasticnet_confusion.png", dpi=300)

plt.close()


# ROC Curve Plot
# ---------------

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 5), dpi=300)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Elastic Net Radiomics)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/results/figures/radiomics_elasticnet_roc.png", dpi=300)
plt.close()


# Coefficient Importance Plot
# -----------------------------

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": logreg_cv.coef_[0]
})
coef_df["abs_coef"] = np.abs(coef_df["Coefficient"])
top20 = coef_df.sort_values("abs_coef", ascending=False).head(20)

plt.figure(figsize=(8, 6), dpi=300)
sns.barplot(data=top20, x="Coefficient", y="Feature", palette="coolwarm")
plt.title("Top 20 Radiomics Features (Elastic Net Coefficients)")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/results/figures/radiomics_elasticnet_top_features.png", dpi=300)
plt.close()

