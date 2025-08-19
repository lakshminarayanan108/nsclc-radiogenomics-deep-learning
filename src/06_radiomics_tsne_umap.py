#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap.umap_ as umap
import os

def simplify_histology(label):
    if "Adenocarcinoma" in label:
        return "Adenocarcinoma"
    elif "Squamous Cell Carcinoma" in label:
        return "Squamous Cell Carcinoma"
    else:
        return None   # drop other rare types

# --- Input files ---

input_radiomics_file = "/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/results/radiomics_features_dedup.csv"


# meta data and labels
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
X = pd.read_csv(input_radiomics_file)          # radiomics features
labels_df = meta_df.copy(deep=True)           # labels dataframe

# Merge on PatientID
merged_df = X.merge(labels_df, on="PatientID", how="inner")

# Labels for plotting
y = merged_df["label"].values   # 0 = Adeno, 1 = Squamous

# Feature matrix (drop metadata columns)
feature_cols = [c for c in merged_df.columns if c not in 
                ["PatientID", "gender", "histology_label", "histology_label_simplified", "label"]]
X = merged_df[feature_cols]

# --- Step 3: Scale ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 4a: t-SNE ---
tsne = TSNE(n_components=2, random_state=42, perplexity=15, n_iter_without_progress=2000)
X_tsne = tsne.fit_transform(X_scaled)

# --- Step 4b: UMAP ---
umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.3)
X_umap = umap_model.fit_transform(X_scaled)

# --- Ensure results folder exists ---
fig_dir = "/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/results/figures/"
os.makedirs(fig_dir, exist_ok=True)

# --- Step 5a: Plot t-SNE ---
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette="coolwarm", s=80, alpha=0.9, edgecolor="k")
plt.title("t-SNE projection of radiomics features", fontsize=14)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Class", labels=["Adenocarcinoma", "Squamous"])
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "tsne_radiomics.png"), dpi=300)
plt.close()

# --- Step 5b: Plot UMAP ---
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=y, palette="coolwarm", s=80, alpha=0.9, edgecolor="k")
plt.title("UMAP projection of radiomics features", fontsize=14)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(title="Class", labels=["Adenocarcinoma", "Squamous"])
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "umap_radiomics.png"), dpi=300)
plt.close()

print(f"✅ Done. {len(merged_df)} samples included (Adeno + Squamous).")
print("t-SNE and UMAP plots saved to:", fig_dir)

