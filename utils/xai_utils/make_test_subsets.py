FRACTIONS = [0.10, 0.25, 0.50]
N_CHUNKS = 4
SEED = 51

import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--out_dir", required=True)
args = parser.parse_args()
FULL_CSV = args.csv
OUT_DIR = args.out_dir

PID = "patient_name"
LAB = "hemorrhage_binary"

os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(FULL_CSV)
pids = sorted(df[PID].unique())

print(f"full: {len(df)} slices, {len(pids)} patients, "
      f"{int(df[LAB].sum())} positive ({100 * df[LAB].mean():.1f}%)\n")


def report(name, sub, path):
    print(f"  {name:22s} {len(sub):>6} slices  {sub[PID].nunique():>3} patients  "
          f"{100 * sub[LAB].mean():>5.1f}% pos   {path}")


print("PATIENT-LEVEL FRACTIONS (use these for reported metrics)")
rng = np.random.RandomState(SEED)
shuffled = list(rng.permutation(pids))
for f in FRACTIONS:
    k = max(1, int(round(len(pids) * f)))
    keep = set(shuffled[:k])
    sub = df[df[PID].isin(keep)].reset_index(drop=True)
    p = os.path.join(OUT_DIR, f"test_frac{int(f * 100):02d}_patient_seed{SEED}.csv")
    sub.to_csv(p, index=False)
    report(f"fraction {f:.2f}", sub, p)

print("\nSTRATIFIED SLICE FRACTIONS (smoke tests only, all patients seen)")
rng = np.random.RandomState(SEED)
for f in FRACTIONS:
    parts = []
    for _, g in df.groupby(LAB):
        k = max(1, int(round(len(g) * f)))
        idx = rng.permutation(len(g))[:k]
        parts.append(g.iloc[idx])
    sub = pd.concat(parts).sort_values([PID, "slice_number"]).reset_index(drop=True)
    p = os.path.join(OUT_DIR, f"test_frac{int(f * 100):02d}_slice_seed{SEED}.csv")
    sub.to_csv(p, index=False)
    report(f"fraction {f:.2f}", sub, p)

print(f"\nCHUNKS ({N_CHUNKS} runs cover every patient exactly once -> full-dataset metrics)")
total = 0
for i in range(N_CHUNKS):
    keep = set(pids[i::N_CHUNKS])
    sub = df[df[PID].isin(keep)].reset_index(drop=True)
    p = os.path.join(OUT_DIR, f"test_chunk{i}of{N_CHUNKS}.csv")
    sub.to_csv(p, index=False)
    report(f"chunk {i}/{N_CHUNKS}", sub, p)
    total += len(sub)

assert total == len(df), f"chunks cover {total} of {len(df)} slices"
print(f"\n  chunks verified: {total} slices == full dataset, no overlap")
