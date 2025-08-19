
import pandas as pd

def simplify_histology(label):
    if "Adenocarcinoma" in label:
        return "Adenocarcinoma"
    elif "Squamous Cell Carcinoma" in label:
        return "Squamous Cell Carcinoma"
    else:
        return None   # drop other rare types


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

# gender encoding
meta_df["gender"] = meta_df["gender"].map({"M": 0, "F": 1})

# label encoding

label_map = {"Adenocarcinoma": 0, "Squamous Cell Carcinoma": 1}
meta_df["label"] = meta_df["histology_label_simplified"].map(label_map)

print(meta_df)
