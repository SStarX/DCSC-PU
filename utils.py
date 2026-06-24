"""Small helpers used by the Chainlink demo."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


ROOT = Path(__file__).resolve().parent
CHAINLINK_PATH = ROOT / "data" / "synthetic" / "chainlink.mat"
OUTPUT_DIR = ROOT / "outputs"


def load_mat_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    payload = loadmat(Path(path))
    data = payload.get("data", payload.get("X"))
    labels = payload.get("labels", payload.get("label"))
    if data is None or labels is None:
        raise KeyError(f"Unsupported MAT structure: {path}")
    data = np.asarray(data, dtype=float)
    minimum = data.min(axis=0)
    span = data.max(axis=0) - minimum
    span[span == 0] = 1.0
    return np.ascontiguousarray((data - minimum) / span), np.ravel(labels)


def clustering_scores(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    true_values, true_index = np.unique(truth, return_inverse=True)
    pred_values, pred_index = np.unique(prediction, return_inverse=True)
    matrix = np.zeros((pred_values.size, true_values.size), dtype=np.int64)
    np.add.at(matrix, (pred_index, true_index), 1)
    rows, columns = linear_sum_assignment(matrix.max(initial=0) - matrix)
    accuracy = matrix[rows, columns].sum() / len(truth)
    return {
        "ACC": 100.0 * accuracy,
        "NMI": 100.0 * normalized_mutual_info_score(truth, prediction),
        "ARI": 100.0 * adjusted_rand_score(truth, prediction),
    }
