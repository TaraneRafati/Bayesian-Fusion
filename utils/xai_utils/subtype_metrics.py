import os
import pandas as pd

from .cam_metrics import compute_segmentation_metrics

SUBTYPE_COLUMN_MAP = {
    "IVH": "IVH",
    "ICH": "ICH",
    "SDH": "SDH",
    "EDH": "EPH",
    "SAH": "SAH",
}

POSITIVE_VALUES = {"1", "1.0", "true", "yes"}


def _parse_patient_and_slice(filename):
    stem = os.path.splitext(filename)[0]
    if "_" in stem:
        patient_id, _, slice_part = stem.rpartition("_")
    else:
        patient_id, slice_part = stem, "0"
    try:
        slice_idx = int(slice_part)
    except ValueError:
        slice_idx = None
    return patient_id, slice_idx


def _load_subtype_lookup(cfg):
    csv_paths = [p for p in (cfg.get("train_csv_path"), cfg.get("test_csv_path")) if p]
    if not csv_paths:
        print("[WARN] calc_subtypes_metrics=True but cfg['train_csv_path']/['test_csv_path'] "
              "are not set; skipping subtype metrics.")
        return None

    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)

    if "patient_name" not in df.columns or "slice_number" not in df.columns:
        print(f"[WARN] subtype csv is missing 'patient_name'/'slice_number'. "
              f"Columns found: {list(df.columns)}. Skipping subtype metrics.")
        return None

    df["patient_name"] = df["patient_name"].astype(str).str.strip()
    df["slice_number"] = pd.to_numeric(df["slice_number"], errors="coerce")

    n_before = len(df)
    df = df.drop_duplicates(subset=["patient_name", "slice_number"], keep="first")
    if len(df) < n_before:
        print(f"[WARN] Dropped {n_before - len(df)} duplicate (patient_name, slice_number) "
              f"rows in the subtype csv(s).")

    return df.set_index(["patient_name", "slice_number"])


def _subtype_membership(result, lookup, col):
    filename = result.get("filename", "")
    patient_id, slice_idx = _parse_patient_and_slice(filename)
    if slice_idx is None:
        return False
    try:
        row = lookup.loc[(patient_id, slice_idx)]
    except KeyError:
        return False
    return str(row[col]).strip().lower() in POSITIVE_VALUES


def compute_subtype_segmentation_metrics(results, cfg):
    if not cfg.get("calc_subtypes_metrics", False):
        return

    lookup = _load_subtype_lookup(cfg)
    if lookup is None:
        return

    subtype_cfg = dict(cfg)
    subtype_cfg["calc_subtypes_metrics"] = False

    n_unmatched = 0
    for r in results:
        patient_id, slice_idx = _parse_patient_and_slice(r.get("filename", ""))
        if slice_idx is None or (patient_id, slice_idx) not in lookup.index:
            n_unmatched += 1
    if n_unmatched:
        print(f"[WARN] {n_unmatched} / {len(results)} slices had no matching row in the subtype csv(s).")

    for display_name, col in SUBTYPE_COLUMN_MAP.items():
        if col not in lookup.columns:
            print(f"[WARN] Column '{col}' (for subtype '{display_name}') not found in subtype csv — skipping.")
            continue

        subset = [r for r in results if _subtype_membership(r, lookup, col)]

        print("\n" + "#" * 55)
        print(f"[SUBTYPE METRICS: {display_name}]  (n={len(subset)})")
        print("#" * 55)

        if not subset:
            print(f"[INFO] No slices found for subtype '{display_name}' — skipping.")
            continue

        compute_segmentation_metrics(subset, subtype_cfg)
