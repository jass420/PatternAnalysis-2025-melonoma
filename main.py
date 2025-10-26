import os, sys
from pathlib import Path
from glob import glob
import pandas as pd

# ---- Paths from your screenshot ----
BASE       = Path(r"C:\Users\mjas0\OneDrive\Desktop\courses\COMP3710\Alzheimer-s")
CSV_A_PATH = BASE / "ISIC_2020_Training_GroundTruth.csv"        # labels
CSV_B_PATH = BASE / "ISIC_2020_Training_GroundTruth_v2.csv"     # metadata (your naming)
DICOM_DIR  = BASE / "ISIC_2020_Train_DICOM_corrected"           # NOTE: 'Train' not 'Training'

def must_exist(p: Path, kind: str):
    if not p.exists():
        print(f"❌ Missing {kind}: {p}")
        sys.exit(1)
    print(f"✅ Found {kind}: {p}")

must_exist(CSV_A_PATH, "CSV_A")
must_exist(CSV_B_PATH, "CSV_B")
must_exist(DICOM_DIR,  "DICOM_DIR")

def load_norm(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'isic_id' in df.columns: df.rename(columns={'isic_id':'image_name'}, inplace=True)
    if 'image'   in df.columns: df.rename(columns={'image':'image_name'}, inplace=True)
    if 'image_name' in df.columns:
        df['image_name'] = (df['image_name'].astype(str)
                                            .str.strip()
                                            .str.replace('.dcm','', regex=False))
    return df

def detect_role(df: pd.DataFrame) -> str:
    gt_keys   = {'target','benign_malignant','diagnosis'}
    meta_keys = {'patient_id','lesion_id','sex','age','anatom_site_general'}
    has_gt   = len(gt_keys.intersection(df.columns))   > 0
    has_meta = len(meta_keys.intersection(df.columns)) > 0
    if has_gt and not has_meta:  return 'gt'
    if has_meta and not has_gt:  return 'meta'
    if has_gt and has_meta:      return 'mixed'
    return 'unknown'

A = load_norm(CSV_A_PATH)
B = load_norm(CSV_B_PATH)
role_A, role_B = detect_role(A), detect_role(B)
print(f"CSV A ({CSV_A_PATH.name}) role → {role_A}")
print(f"CSV B ({CSV_B_PATH.name}) role → {role_B}")

# Decide which df to use for labels vs metadata
if role_A == 'gt' and role_B in ('meta','mixed'):
    gt, meta = A, B
elif role_B == 'gt' and role_A in ('meta','mixed'):
    gt, meta = B, A
else:
    # both 'mixed' is fine — just pick A for labels, B for meta
    gt, meta = A, B

# Ensure a single binary label exists in GT
if 'target' not in gt.columns:
    if 'benign_malignant' in gt.columns:
        gt['target'] = gt['benign_malignant'].astype(str).str.lower().map({'malignant':1,'benign':0})
    elif 'diagnosis' in gt.columns:
        gt['target'] = gt['diagnosis'].astype(str).str.lower().str.contains('melanoma').astype(int)
    else:
        raise ValueError("GroundTruth needs 'target' or 'benign_malignant' or 'diagnosis'.")

# 🚫 If metadata ALSO has a target-like column, drop it to avoid target_x/target_y
for col in ['target','target_x','target_y']:
    if col in meta.columns:
        meta = meta.drop(columns=[col])
        break

# Merge labels into metadata -> ensure we end with exactly one 'target'
df = meta.merge(gt[['image_name','target']], on='image_name', how='inner')
print("Rows after metadata+GT merge (expect ~33126):", len(df))

# Recursively gather all TRAIN DICOMs (.dcm and .DCM)
DICOM_DIR = DICOM_DIR.resolve()
all_dcm = glob(str(DICOM_DIR / '**' / '*.dcm'), recursive=True) + \
          glob(str(DICOM_DIR / '**' / '*.DCM'), recursive=True)
print("Total DICOM files found recursively in TRAIN:", len(all_dcm))

id_to_path = {Path(p).stem.strip(): str(Path(p).resolve()) for p in all_dcm}
df['dcm_path'] = df['image_name'].map(id_to_path)

missing = df['dcm_path'].isna().sum()
print(f"Matched files: {len(df) - missing} / {len(df)}")
if missing:
    print("Example missing IDs:", df.loc[df['dcm_path'].isna(), 'image_name'].head(10).tolist())
    if len(df) - missing == 0:
        sys.exit("No TRAIN DICOMs matched. Check DICOM_DIR & that files are extracted locally.")

# Keep only the columns that actually exist
keep = ['image_name','dcm_path','target']
for c in ['patient_id','lesion_id']:
    if c in df.columns: keep.append(c)
df = df[keep].reset_index(drop=True)

print("Class counts:")
print(df['target'].value_counts())

out = BASE / "train_mapping.csv"
df.to_csv(out, index=False)
print("✅ Wrote:", out)
