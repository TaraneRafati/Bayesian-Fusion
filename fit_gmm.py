import argparse
import os

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from utils.xai_utils.gmm_params import TISSUES, save_gmm_parameters

SOFT_TISSUE_MIN_HU = 0
SOFT_TISSUE_MAX_HU = 100
BONE_MIN_HU = 200


def parse_args():
    parser = argparse.ArgumentParser(description="Fit the per-class HU Gaussian mixtures on the training split.")
    parser.add_argument("--original_ct_dir", required=True)
    parser.add_argument("--stripped_ct_dir", required=True)
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--output", default="gmm_parameters.csv")
    parser.add_argument("--n_components", type=int, default=2)
    parser.add_argument("--sample_size", type=int, default=1_000_000)
    parser.add_argument("--skull_sample_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=51)
    return parser.parse_args()


def train_filenames(csv_path):
    df = pd.read_csv(csv_path)
    return sorted(f"{p}_{s}.npy" for p, s in zip(df["patient_name"], df["slice_number"]))


def is_soft_tissue(ct):
    return (ct > SOFT_TISSUE_MIN_HU) & (ct < SOFT_TISSUE_MAX_HU)


def draw(values, taken, cap, rng):
    if cap is None:
        return values
    n = min(values.size, cap - taken)
    if n <= 0:
        return values[:0]
    return rng.choice(values, size=n, replace=False)


def collect_samples(args):
    rng = np.random.default_rng(args.seed)
    caps = {
        "hemorrhage": args.sample_size,
        "background": args.sample_size,
        "skull": args.skull_sample_size or None,
    }
    samples = {t: [] for t in TISSUES}
    taken = {t: 0 for t in TISSUES}

    for filename in train_filenames(args.train_csv):
        if all(caps[t] is not None and taken[t] >= caps[t] for t in TISSUES):
            break

        paths = [
            os.path.join(args.original_ct_dir, filename),
            os.path.join(args.stripped_ct_dir, filename),
            os.path.join(args.masks_dir, filename),
        ]
        if not all(os.path.exists(p) for p in paths):
            continue

        original, stripped, mask = (np.load(p).astype(np.int64) for p in paths)
        healthy = mask == 0

        pools = {
            "hemorrhage": original[(mask != 0) & is_soft_tissue(original)],
            "background": stripped[healthy & is_soft_tissue(stripped)],
            "skull": original[(original >= BONE_MIN_HU) & (stripped <= 0) & healthy],
        }
        for tissue, values in pools.items():
            chosen = draw(values, taken[tissue], caps[tissue], rng)
            if chosen.size:
                samples[tissue].append(chosen.astype(np.float32))
                taken[tissue] += chosen.size

    for tissue in TISSUES:
        if not samples[tissue]:
            raise ValueError(f"No '{tissue}' pixels found in the training split")
    return {t: np.concatenate(samples[t]) for t in TISSUES}


def fit_mixture(values, n_components, seed):
    gmm = GaussianMixture(n_components=n_components, random_state=seed)
    gmm.fit(values.reshape(-1, 1))
    means = gmm.means_.ravel()
    order = np.argsort(means)
    return {
        "weights": gmm.weights_[order],
        "means": means[order],
        "stds": np.sqrt(gmm.covariances_.ravel()[order]),
    }


def main():
    args = parse_args()
    samples = collect_samples(args)

    params = {}
    for tissue in TISSUES:
        print(f"Fitting {args.n_components}-component mixture for {tissue} on {samples[tissue].size:,} pixels")
        params[tissue] = fit_mixture(samples[tissue], args.n_components, args.seed)
        for k, (w, m, s) in enumerate(zip(*params[tissue].values())):
            print(f"  component {k}: weight={w:.4f} mean={m:.2f} HU std={s:.2f}")

    save_gmm_parameters(params, args.output)
    print(f"Saved parameters to {args.output}")


if __name__ == "__main__":
    main()
