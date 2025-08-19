import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap.umap_ as umap
import os

input_expr_data_file = '/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/data/processed/omics/processed_gene_expr_with_labels.csv'

merged_df = pd.read_csv(input_expr_data_file)


# --- Step 1: Features & labels ---

X = merged_df.drop(columns=["patientID", "gender", "histology_label", 
                            "histology_label_simplified", "label"])

y = merged_df["label"].values  # 0 = Adeno, 1 = Squamous

# --- Step 2: Scale ---

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# --- Step 3a: t-SNE ---

tsne = TSNE(n_components=2, random_state=42, perplexity=15, n_iter_without_progress=2000)

X_tsne = tsne.fit_transform(X_scaled)

# --- Step 3b: UMAP ---

umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.3)

X_umap = umap_model.fit_transform(X_scaled)

# --- Ensure results folder exists ---

os.makedirs("/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/results/figures/", exist_ok=True)

# --- Step 4a: Plot t-SNE ---
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette="coolwarm", s=80, alpha=0.9, edgecolor="k")
plt.title("t-SNE projection of gene expression", fontsize=14)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Class", labels=["Adenocarcinoma", "Squamous"])
plt.tight_layout()
plt.savefig("/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/results/figures/tsne_gene_expression.png", dpi=300)
plt.close()

# --- Step 4b: Plot UMAP ---
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=y, palette="coolwarm", s=80, alpha=0.9, edgecolor="k")
plt.title("UMAP projection of gene expression", fontsize=14)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(title="Class", labels=["Adenocarcinoma", "Squamous"])
plt.tight_layout()
plt.savefig("/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/results/figures/umap_gene_expression.png", dpi=300)
plt.close()

