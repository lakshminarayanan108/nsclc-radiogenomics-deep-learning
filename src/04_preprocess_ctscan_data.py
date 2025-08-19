import os
import subprocess
import SimpleITK as sitk


# CONFIG
# --------------------------

RAW_DICOM_DIR = "/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/data/raw/dicom"
NIFTI_DIR     = "/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/data/processed/nifti"
SEG_DIR       = "/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/data/processed/segmentation"
MASKS_DIR     = "/home/lakshminarayanan_evolution/nsclc-radiogenomics-deep-learning/data/processed/masks"

# ensure output dirs exist
os.makedirs(NIFTI_DIR, exist_ok=True)
os.makedirs(SEG_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)


# FUNCTIONS
# --------------------------
"""
def convert_dicom_to_nifti(dicom_dir, out_dir):
    
    subprocess.run([
        "dcm2niix", "-z", "y", "-o", out_dir, dicom_dir
    ], check=True)
"""

def convert_dicom_to_nifti(dicom_dir, out_dir):
    try:
        subprocess.run(
            ["dcm2niix", "-z", "y", "-o", out_dir, dicom_dir],
            check=True
        )
    except subprocess.CalledProcessError:
        print(f"⚠️ Skipping {dicom_dir}, conversion failed (missing slices).")


def run_totalsegmentator(ct_path, out_dir, task):
    """Run TotalSegmentator for a given task (e.g., lung_nodules / lung_vessels)"""
    subprocess.run([
        "TotalSegmentator", "-i", ct_path, "-o", out_dir, "--task", task
    ], check=True)

def align_mask_to_ct(ct_path, mask_path, out_path):
    """Resample mask to match CT voxel grid (safety check)"""
    img  = sitk.ReadImage(ct_path)
    mask = sitk.ReadImage(mask_path)
    mask_res = sitk.Resample(mask, img, sitk.Transform(),
                             sitk.sitkNearestNeighbor, 0,
                             mask.GetPixelID())
    sitk.WriteImage(mask_res, out_path)


# MAIN LOOP
# --------------------------

patients = sorted([p for p in os.listdir(RAW_DICOM_DIR) if p.startswith("LUNG3")])

for pid in patients:

    out_file = os.path.join(MASKS_DIR, f"{pid}_lung_vessels_aligned.nii.gz")

    if os.path.exists(out_file):
        print(f"✅ Skipping {pid}, already processed")
        continue

    print(f"\n=== Processing {pid} ===")
    dicom_dir = os.path.join(RAW_DICOM_DIR, pid)
    out_nifti_dir = os.path.join(NIFTI_DIR, pid)
    out_seg_dir   = os.path.join(SEG_DIR, pid)
    os.makedirs(out_nifti_dir, exist_ok=True)
    os.makedirs(out_seg_dir, exist_ok=True)

    # Step 1: DICOM → NIfTI
    convert_dicom_to_nifti(dicom_dir, out_nifti_dir)

    # Find CT file (first .nii.gz in converted dir)
    ct_files = [f for f in os.listdir(out_nifti_dir) if f.endswith(".nii.gz")]
    if not ct_files:
        print(f"⚠️ No NIfTI found for {pid}, skipping.")
        continue
    ct_path = os.path.join(out_nifti_dir, ct_files[0])

    # Step 2: Segmentation (both nodules + vessels)
    for task in ["lung_nodules", "lung_vessels"]:
        print(f"  → Running TotalSegmentator task: {task}")
        task_out_dir = os.path.join(out_seg_dir, task)
        os.makedirs(task_out_dir, exist_ok=True)

        run_totalsegmentator(ct_path, task_out_dir, task)

        # Step 3: Extract & align mask
        mask_file = os.path.join(task_out_dir, f"{task}.nii.gz")
        if not os.path.exists(mask_file):
            print(f"⚠️ No {task} mask found for {pid}, skipping.")
            continue

        out_mask = os.path.join(MASKS_DIR, f"{pid}_{task}_aligned.nii.gz")
        align_mask_to_ct(ct_path, mask_file, out_mask)

        print(f"  ✅ Saved {task} mask: {out_mask}")

    print(f"✅ Finished {pid}: CT={ct_path}")
    

