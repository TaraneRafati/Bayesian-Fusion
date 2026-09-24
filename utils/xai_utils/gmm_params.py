import numpy as np
import pandas as pd

TISSUES = ("hemorrhage", "skull", "background")


def save_gmm_parameters(params, path):
    rows = []
    for tissue in TISSUES:
        p = params[tissue]
        for k, (w, m, s) in enumerate(zip(p["weights"], p["means"], p["stds"])):
            rows.append({"tissue": tissue, "component": k, "weight": w, "mean": m, "std": s})
    pd.DataFrame(rows).to_csv(path, index=False)


def load_gmm_parameters(path):
    df = pd.read_csv(path)
    params = {}
    for tissue in TISSUES:
        sub = df[df["tissue"] == tissue].sort_values("mean")
        if sub.empty:
            raise ValueError(f"No '{tissue}' rows in {path}")
        params[tissue] = {
            "weights": sub["weight"].to_numpy(dtype=float).tolist(),
            "means": sub["mean"].to_numpy(dtype=float).tolist(),
            "stds": sub["std"].to_numpy(dtype=float).tolist(),
        }
        if not np.isclose(sum(params[tissue]["weights"]), 1.0, atol=1e-3):
            raise ValueError(f"'{tissue}' weights in {path} do not sum to 1")
    return params
