"""DPC baseline used only by the noisy Chainlink demonstration.

The implementation follows the root-level ``00-inbox/DPC.py`` version while
omitting its interactive plotting entry point and large-scale helper.
"""

from collections import OrderedDict

import numpy as np
from scipy.spatial.distance import pdist, squareform


def get_distance_cut(distances: np.ndarray, distance_percent: float) -> float:
    order = np.argsort(distances)
    index = int(len(order) * distance_percent / 100)
    return float(distances[order[index]])


def get_density(sample_count: int, distance_matrix: np.ndarray, distance_cut: float) -> np.ndarray:
    weights = np.exp(-((distance_matrix / distance_cut) ** 2))
    np.fill_diagonal(weights, 0)
    return np.sum(weights, axis=1)


def _assign_clusters(
    sample_count: int,
    distance_matrix: np.ndarray,
    density: np.ndarray,
    cluster_count: int,
) -> tuple[OrderedDict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    density_order = np.flipud(np.argsort(density))
    delta = np.zeros(sample_count, dtype=float)
    leader = np.full(sample_count, -1, dtype=int)
    delta[density_order[0]] = np.max(distance_matrix[density_order[0], :])

    for position in range(1, sample_count):
        current = density_order[position]
        higher_density = density_order[:position]
        nearest_position = np.argmin(distance_matrix[current, higher_density])
        nearest = higher_density[nearest_position]
        delta[current] = distance_matrix[current, nearest]
        leader[current] = nearest

    delta[density_order[0]] = max(delta[1:])
    gamma_order = np.flipud(np.argsort(delta * density))
    cluster_index = np.full(sample_count, -1, dtype=int)
    selected_centers: list[int] = []
    center_indices: list[int] = []

    cursor = 0
    while len(center_indices) < cluster_count and cursor < sample_count:
        candidate = int(gamma_order[cursor])
        duplicate = any(distance_matrix[candidate, chosen] == 0 for chosen in selected_centers)
        if not duplicate:
            cluster_index[candidate] = len(center_indices)
            selected_centers.append(candidate)
            center_indices.append(candidate)
        cursor += 1

    center_mask = cluster_index.copy()
    for point in density_order:
        if cluster_index[point] == -1:
            cluster_index[point] = cluster_index[leader[point]]

    clusters = OrderedDict((cluster_id, []) for cluster_id in range(cluster_count))
    for point, cluster_id in enumerate(cluster_index):
        clusters[cluster_id].append(point)
    return clusters, delta, density, np.asarray(center_indices), leader


def DPC(
    data: np.ndarray,
    cluster_count: int,
    distance_percent: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run Density Peak Clustering with the paper's Gaussian density kernel."""
    distances = pdist(data, metric="euclidean")
    distance_matrix = squareform(distances)
    distance_cut = get_distance_cut(distances, distance_percent)
    density = get_density(len(data), distance_matrix, distance_cut)
    clusters, delta, density, centers, leader = _assign_clusters(
        len(data), distance_matrix, density, cluster_count
    )
    labels = np.zeros(len(data), dtype=int)
    for cluster_id, members in clusters.items():
        labels[members] = cluster_id
    return labels, centers, leader, density, delta
