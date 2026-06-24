"""
DCSC_PUparallel.py
-----------
完整的 DCSC-PU 算法实现，将第一阶段 T/P_k 的估计替换为并行版本
（calculate_peak_uncertainty_online_parallel）。
其余阶段（核心簇选择、族树构建、相似度计算、谱聚类）完全与原版 DCSC_PU.py 保持一致，
确保算法语义不变的前提下提升效率。
"""
import math
import warnings

import joblib
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import SpectralClustering
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

# 第一阶段：使用并行加速的 T/P_k 估计
from Src.peak_uncertainty import (
    calculate_peak_uncertainty_online_parallel,
    compute_bandwidth,
    _as_float_array,
    _fit_qspp,
    _group_core_indices,
    _compute_density_and_delta,
    _find_qspp_params_parallel,
    sampling,
    noising,
)
from QuickshiftPP import QuickshiftPP

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 核心区域选取（与原版一致）
# ---------------------------------------------------------------------------

def _select_cores(D, C, k, beta, width):
    """Run QSPP once and keep the top-C principal core regions."""
    D = _as_float_array(D)
    model = _fit_qspp(D, k, beta, width)
    core_groups = _group_core_indices(model.Mcore)
    if not core_groups:
        return []

    densities, deltas = _compute_density_and_delta(
        D, core_groups, np.asarray(model.den, dtype=float)
    )
    scores = densities * deltas
    top_idx = np.argsort(scores)[::-1][: min(C, len(core_groups))]
    return [core_groups[i] for i in top_idx]


def _find_qspp_params(D, C, width, n_jobs=-1):
    """Quick heuristic search for a QSPP parameter pair with enough cores."""
    return _find_qspp_params_parallel(D, C, width, n_jobs=n_jobs)


def _actual_jobs(n_jobs):
    cpu_cores = joblib.cpu_count()
    if n_jobs == -1:
        return cpu_cores
    return min(max(1, int(n_jobs)), cpu_cores)


def _adaptive_core_mask_worker(D, C, k, beta, base_width, perturb_type, sub_seed):
    if perturb_type == "base":
        regions = _select_cores(D, C, k, beta, base_width)
        return [np.asarray(group, dtype=int) for group in regions]

    if perturb_type == "sample":
        D_view, idx_map = sampling(D, sub_seed)
        regions = _select_cores(D_view, C, k, beta, base_width)
        return [idx_map[group] for group in regions]

    D_view = noising(D, sub_seed)
    regions = _select_cores(D_view, C, k, beta, base_width)
    return [np.asarray(group, dtype=int) for group in regions]


def _knn_density_from_distances(neighbor_dist):
    """Estimate local density from KNN distances already computed for tree building."""
    if neighbor_dist.size == 0:
        return np.ones(neighbor_dist.shape[0], dtype=float)
    eps = np.finfo(float).eps
    mean_dist = np.mean(neighbor_dist, axis=1)
    return 1.0 / np.maximum(mean_dist, eps)


# ---------------------------------------------------------------------------
# 第二阶段：T 自适应核心掩码构建（与原版一致）
# ---------------------------------------------------------------------------

def build_adaptive_core_mask(D, C, k, beta, T, seed=42, n_jobs=-1):
    """Build the T-adaptive core mask described in the paper."""
    D = _as_float_array(D)
    m_core = max(1, math.ceil(5 * T))
    rng = np.random.default_rng(seed)
    core_mask = np.zeros(D.shape[0], dtype=float)

    base_width = compute_bandwidth(D, random_state=seed)
    tasks = [("base", seed)]
    for _ in range(1, m_core):
        sub_seed = int(rng.integers(1, 100000))
        if rng.random() < 0.5:
            tasks.append(("sample", sub_seed))
        else:
            tasks.append(("noise", sub_seed))

    actual_jobs = min(_actual_jobs(n_jobs), len(tasks))
    if actual_jobs == 1:
        results = [
            _adaptive_core_mask_worker(D, C, k, beta, base_width, perturb_type, sub_seed)
            for perturb_type, sub_seed in tasks
        ]
    else:
        results = Parallel(n_jobs=actual_jobs, backend="loky")(
            delayed(_adaptive_core_mask_worker)(
                D, C, k, beta, base_width, perturb_type, sub_seed
            )
            for perturb_type, sub_seed in tasks
        )

    for regions in results:
        for group in regions:
            core_mask[group] = 1.0
    return core_mask


# ---------------------------------------------------------------------------
# 第三阶段：核心族树构建（与原版一致）
# ---------------------------------------------------------------------------

def build_corefamilytree(D, C, k, beta, T, P_k, alpha=1.0, seed=42, n_jobs=-1):
    """Construct local density family trees on the T-adaptive core set."""
    D = _as_float_array(D)
    n = D.shape[0]
    core_mask = build_adaptive_core_mask(D, C, k, beta, T, seed, n_jobs=n_jobs)
    valid_cores = np.flatnonzero(core_mask > 0)
    if valid_cores.size == 0:
        return {}

    nn = NearestNeighbors(n_neighbors=min(k + 1, n))
    nn.fit(D)
    knn_dist, knn_idx = nn.kneighbors(D)
    neighbor_k = knn_idx[:, 1:]
    neighbor_dist = knn_dist[:, 1:]
    raw_density = _knn_density_from_distances(neighbor_dist)
    density = raw_density * (1.0 + alpha * np.asarray(P_k, dtype=float) * core_mask)

    valid_neighbor_density = density[neighbor_k[valid_cores]]
    is_peak = np.all(density[valid_cores, None] > valid_neighbor_density, axis=1)
    peaks = valid_cores[is_peak].astype(int)

    labels = np.full(n, -1, dtype=int)
    labels[peaks] = peaks

    order = valid_cores[np.argsort(density[valid_cores])[::-1]]
    for point_idx in order:
        if labels[point_idx] != -1:
            continue
        nbrs = neighbor_k[point_idx]
        denser_mask = density[nbrs] >= density[point_idx]
        if not np.any(denser_mask):
            continue

        denser_nbrs = nbrs[denser_mask]
        denser_dists = neighbor_dist[point_idx][denser_mask]
        leader = denser_nbrs[np.argmin(denser_dists)]
        labels[point_idx] = labels[leader]

    return {int(peak): np.flatnonzero(labels == peak) for peak in peaks}


# ---------------------------------------------------------------------------
# 第四阶段：相似度计算（与原版一致）
# ---------------------------------------------------------------------------

def measure_similarity(D, core_tree, k, T=1.0):
    """Compute tree-level similarity using SNN overlap and graph distance."""
    D = _as_float_array(D)
    peaks = list(core_tree.keys())
    n_peaks = len(peaks)
    if n_peaks == 0:
        return np.zeros((0, 0), dtype=float)

    graph = kneighbors_graph(
        D,
        n_neighbors=min(max(k, 1), max(D.shape[0] - 1, 1)),
        mode="distance",
        include_self=False,
    )
    peak_dist = dijkstra(graph, directed=False, indices=peaks)[:, peaks]

    nbr_indices = NearestNeighbors(n_neighbors=min(max(k, 1), D.shape[0])).fit(D).kneighbors(
        D, return_distance=False
    )
    tree_neighbor_sets = [
        set(np.unique(nbr_indices[members].reshape(-1)).tolist()) for members in core_tree.values()
    ]

    sim_matrix = np.zeros((n_peaks, n_peaks), dtype=float)
    for i in range(n_peaks):
        set_i = tree_neighbor_sets[i]
        for j in range(i + 1, n_peaks):
            graph_dist = peak_dist[i, j]
            if not np.isfinite(graph_dist):
                continue
            overlap = len(set_i & tree_neighbor_sets[j])
            sim = overlap / (1.0 + graph_dist)
            sim_matrix[i, j] = sim_matrix[j, i] = sim

    s_max = sim_matrix.max(initial=0.0)
    if s_max > 0:
        sim_matrix = np.expm1(T * (sim_matrix / s_max))
    return sim_matrix


# ---------------------------------------------------------------------------
# 第五阶段：最终聚类（与原版一致）
# ---------------------------------------------------------------------------

def final_clustering(D, S, core_tree, C):
    """Merge family trees with spectral clustering and fill the remaining points."""
    D = _as_float_array(D)
    if len(core_tree) == 0:
        return np.full(D.shape[0], -1, dtype=int)

    tree_labels = SpectralClustering(
        n_clusters=C,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    ).fit_predict(S)

    final_labels = np.full(D.shape[0], -1, dtype=int)
    assigned_mask = np.zeros(D.shape[0], dtype=bool)
    peaks = list(core_tree.keys())

    for idx, cluster_id in enumerate(tree_labels):
        members = core_tree[peaks[idx]]
        final_labels[members] = int(cluster_id)
        assigned_mask[members] = True

    unassigned = np.flatnonzero(~assigned_mask)
    assigned = np.flatnonzero(assigned_mask)
    if unassigned.size > 0 and assigned.size > 0:
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(D[assigned])
        nearest = nn.kneighbors(D[unassigned], return_distance=False).reshape(-1)
        final_labels[unassigned] = final_labels[assigned[nearest]]
    return final_labels


# ---------------------------------------------------------------------------
# 并行版主接口
# ---------------------------------------------------------------------------

def DCSC_PU_parallel(D, C, k, beta, T, P_k, alpha=1.0, seed=42, n_jobs=-1):
    """Main DCSC-PU interface（与 DCSC_PU 等价，可直接替换调用）."""
    D = _as_float_array(D)
    core_tree = build_corefamilytree(D, C, k, beta, T, P_k, alpha, seed, n_jobs=n_jobs)
    similarity = measure_similarity(D, core_tree, k, T)
    labels = final_clustering(D, similarity, core_tree, C)
    return labels, core_tree


def DCSC_PU_full_parallel(D, C, k, beta, alpha=1.0, seed=42, n_jobs=-1, **uncertainty_kwargs):
    """End-to-end wrapper：使用并行 T/P_k 估计后运行 DCSC-PU。

    Parameters
    ----------
    D : array-like, shape (n_samples, n_features)
    C : int, 目标簇数
    k : int, QSPP 邻居数
    beta : float, QSPP beta 参数
    alpha : float, 密度加权系数
    seed : int, 随机种子
    n_jobs : int, 并行进程数，-1 表示使用所有 CPU 核心
    **uncertainty_kwargs : dict
        透传给 calculate_peak_uncertainty_online_parallel 的其余参数
        (M_T, M_p, K_max 等)
    """
    D = _as_float_array(D)
    T, P_k, converged, iters = calculate_peak_uncertainty_online_parallel(
        D, C, seed=seed, n_jobs=n_jobs, **uncertainty_kwargs
    )
    labels, core_tree = DCSC_PU_parallel(
        D, C, k, beta, T, P_k, alpha=alpha, seed=seed, n_jobs=n_jobs
    )
    return labels, core_tree, T, P_k, converged, iters


DCSC_PU = DCSC_PU_parallel
DCSC_PU_full = DCSC_PU_full_parallel
