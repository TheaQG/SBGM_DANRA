# sbgm/evaluate/evaluate_prcp/eval_features/metrics_features.py
from __future__ import annotations
import numpy as np


def compute_sal(hr: np.ndarray, gen: np.ndarray, lr: np.ndarray | None = None) -> dict:
    """
    Compute the Structure–Amplitude–Location (SAL) metrics as in
    Wernli et al. (2008), QJRMS.
    Works on 2D or 3D arrays (D,H,W).
    Returns per-source metrics and combined differences.
    """

    def _sal_pair(ref: np.ndarray, test: np.ndarray) -> dict:
        """Compute SAL components for one pair."""
        # Amplitude
        A_ref = np.mean(ref)
        A_test = np.mean(test)
        A = 2 * (A_test - A_ref) / (A_test + A_ref + 1e-8)

        # Structure (std difference)
        S_ref = np.std(ref)
        S_test = np.std(test)
        S = 2 * (S_test - S_ref) / (S_test + S_ref + 1e-8)

        # Location (centroid distance)
        def centroid(x):
            total = np.sum(x)
            if total <= 0:
                return np.array([np.nan, np.nan])
            coords = np.indices(x.shape)
            return np.array([np.sum(coords[0] * x) / total, np.sum(coords[1] * x) / total])

        c_ref, c_test = centroid(ref), centroid(test)
        L = np.sqrt(np.sum((c_test - c_ref) ** 2)) / np.sqrt(np.sum(np.array(ref.shape) ** 2))

        return dict(A=A, S=S, L=L)

    res = {}
    if hr is not None and gen is not None:
        res["GEN_vs_HR"] = _sal_pair(hr, gen)
    if hr is not None and lr is not None:
        res["LR_vs_HR"] = _sal_pair(hr, lr)

    for key in res:
        res[key]["SAL"] = np.sqrt(res[key]["A"] ** 2 + res[key]["S"] ** 2 + res[key]["L"] ** 2)

    res["summary"] = {
        "A_GEN": res.get("GEN_vs_HR", {}).get("A", np.nan),
        "S_GEN": res.get("GEN_vs_HR", {}).get("S", np.nan),
        "L_GEN": res.get("GEN_vs_HR", {}).get("L", np.nan),
        "A_LR": res.get("LR_vs_HR", {}).get("A", np.nan),
        "S_LR": res.get("LR_vs_HR", {}).get("S", np.nan),
        "L_LR": res.get("LR_vs_HR", {}).get("L", np.nan),
    }
    return res