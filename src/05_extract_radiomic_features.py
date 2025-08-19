
import os
import glob
import pandas as pd
from radiomics import featureextractor
import SimpleITK as sitk


# Configurations
# ---------------

CT_DIR = "/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/data/processed/nifti/"          # directory of CT scans

MASK_DIR = "/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/data/processed/masks/"     # directory with *_nodules_* and *_vessels_* masks

OUTPUT_CSV = "/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/results/radiomics_features.csv"

# Choose which mask type to use: "nodules" or "vessels"

MASK_TYPE = "nodules"   # change to "vessels" if you want vessel masks

# Optional: YAML parameters file for reproducibility

PARAMS_FILE = "/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/config/radiomics_params.yaml"


# Initialize Feature Extractor
# -----------------------------

if os.path.exists(PARAMS_FILE):
    extractor = featureextractor.RadiomicsFeatureExtractor(PARAMS_FILE)
else:
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    extractor.enableImageTypes(Original={}, Wavelet={}, LoG={})

print("✅ Radiomics feature extractor initialized")
print("Settings:", extractor.settings)


# Run feature extraction
# ------------------------

records = []

ct_files = sorted(glob.glob(os.path.join(CT_DIR, "*/*.nii.gz")))

if not ct_files:
    raise FileNotFoundError(f"No CT scans found in {CT_DIR}")


# --- Full extraction loop ---
for ct_path in ct_files:
    patient_id = os.path.basename(os.path.dirname(ct_path))
    mask_path = os.path.join(MASK_DIR, f"{patient_id}_lung_{MASK_TYPE}_aligned.nii.gz")

    if not os.path.exists(mask_path):
        print(f"⚠️ No {MASK_TYPE} mask found for {patient_id}, skipping...")
        continue

    print(f"➡️ Extracting {MASK_TYPE} features for {patient_id}...")

    ct_image = sitk.ReadImage(ct_path)
    mask_image = sitk.ReadImage(mask_path)
    feature_vector = extractor.execute(ct_image, mask_image)

    features = {k: v for k, v in feature_vector.items() if not k.startswith("diagnostics")}
    features["PatientID"] = patient_id
    records.append(features)
    

# Save results
# ------------------------

if records:
    df = pd.DataFrame(records).set_index("PatientID")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV)
    print(f"✅ Radiomics features saved to {OUTPUT_CSV}")
    print(f"Total patients processed: {len(df)}")
    print(f"Total features per patient: {df.shape[1]}")
else:
    print("⚠️ No features extracted. Check your CT and mask directories.")

