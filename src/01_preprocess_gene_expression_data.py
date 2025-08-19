import GEOparse
import pandas as pd


# Collapse histology labels into two main groups

def simplify_histology(label):
    if "Adenocarcinoma" in label:
        return "Adenocarcinoma"
    elif "Squamous Cell Carcinoma" in label:
        return "Squamous Cell Carcinoma"
    else:
        return None   # drop other rare types


raw_omics_data_dir = '/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/data/raw/omics/'

# 1. Load platform annotation (probe ↔ gene symbol mapping)
# ============================================================

soft_file_path = raw_omics_data_dir + "GPL15048_family.soft"

gpl = GEOparse.get_GEO(filepath=soft_file_path)

platform_table = gpl.table

# keep only probe_id and gene_symbol
annot_df = platform_table[["ID", "GeneSymbol"]].rename(columns={"ID": "probe_id", "GeneSymbol": "gene_symbol"})


# 2. Load expression data (series matrix)
# ========================================

matrix_file_path = raw_omics_data_dir + "GSE58661_series_matrix.txt"

expr_df = pd.read_csv(matrix_file_path, sep="\t", comment="!", index_col=0)

expr_df = expr_df.reset_index().rename(columns={"ID_REF": "probe_id"})

print(expr_df.shape)
print(expr_df.head())


#gse = GEOparse.get_GEO(filepath=matrix_file_path)
# expression data (probes × GSMs)
#expr_df = gse.pivot_samples("VALUE")
# ensure probe IDs are in a column
#expr_df = expr_df.reset_index().rename(columns={"ID_REF": "probe_id"})


# 3. Map patient IDs ↔ GEO sample IDs
# =====================================

patient_geo_map_file_path = raw_omics_data_dir + "LUNG3_patient_ID_GEO_Map.parquet"

patientID_geo_accession_map = pd.read_parquet(patient_geo_map_file_path)

# structure of the data: patient_map = pd.DataFrame({ "patientID": [], "geo_accession": []})


# rename expression columns from GSM IDs → patient IDs
expr_df = expr_df.rename(columns=dict(zip(patientID_geo_accession_map.geo_accession,
                                          patientID_geo_accession_map.patientID)))


# 4. Merge probes ↔ gene symbols
# ================================

expr_df = expr_df.merge(annot_df, on="probe_id", how="left")

print(expr_df.head())

# 5. Collapse probes → gene symbols
# ===================================

# Drop probe_id column, average across probes per gene
expr_gene_df = expr_df.drop(columns="probe_id") \
                      .groupby("gene_symbol") \
                      .mean(numeric_only=True)

# check for NaN values and redundant gene symbols

print(expr_gene_df.head())

#print(expr_gene_df.isna().sum())

#print(len(expr_gene_df['gene_symbol'].unique()))



# 6. merge with gender, histology classification label from meta data file
# =========================================================================

meta_data_dir = '/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/data/raw/metadata/'

meta_data_path = meta_data_dir + 'Lung3.metadata.csv'

metadata_df = pd.read_csv(meta_data_path)

# keep only sample.name, gender, histology

meta_df = metadata_df[["title", "characteristics.tag.gender", "characteristics.tag.histology"]]

# rename columns for clarity
meta_df = meta_df.rename(columns={
    "title": "patientID",
    "characteristics.tag.gender": "gender",
    "characteristics.tag.histology": "histology_label"
})

meta_df["histology_label_simplified"] = meta_df["histology_label"].apply(simplify_histology)

meta_df = meta_df[meta_df["histology_label_simplified"].notna()]


expr_gene_t = expr_gene_df.T.reset_index().rename(columns={"index": "patientID"})

merged_df = expr_gene_t.merge(meta_df, on="patientID", how="inner")

print(merged_df.shape)

print(merged_df.head())


print(merged_df["histology_label"].value_counts())

print(merged_df["histology_label_simplified"].value_counts())


main_labels = ["Adenocarcinoma", "Squamous Cell Carcinoma"]

filtered_df = merged_df[merged_df["histology_label_simplified"].isin(main_labels)].copy()

print(filtered_df.head())

# gender encoding
filtered_df["gender"] = filtered_df["gender"].map({"M": 0, "F": 1})

# label encoding
label_map = {"Adenocarcinoma": 0, "Squamous Cell Carcinoma": 1}
filtered_df["label"] = filtered_df["histology_label_simplified"].map(label_map)

print(filtered_df.head())


# 7. Save final gene × patient matrix
# ====================================

gene_expression_results_file = '/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/data/processed/omics/' + 'processed_gene_expr_with_labels.csv'

filtered_df.to_csv(gene_expression_results_file)

print("Final matrix shape:", filtered_df.shape)

print(filtered_df.head())

