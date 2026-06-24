import warnings
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
from sklearn.neighbors import NearestNeighbors
import joblib
from joblib import Parallel, delayed

# 样本量阈值：小于该值时重用进程池（减少建立开销），否则不重用（防止大内存残留 OOM）
_PROCESS_REUSE_THRESHOLD = 10000

from QuickshiftPP import QuickshiftPP

warnings.filterwarnings("ignore")

_BANDWIDTH_EXACT_LIMIT = 4096
_BANDWIDTH_SAMPLE_SIZE = 2048


def _as_float_array(D):
    return np.ascontiguousarray(np.asarray(D, dtype=float))


def compute_bandwidth(D, sample_size=_BANDWIDTH_SAMPLE_SIZE, random_state=42):
    """Estimate the kernel bandwidth used by the clustering pipeline."""
    D = _as_float_array(D)
    n, d = D.shape
    if n <= 1:
        return 0.001

    percentile = 2 if d <= 10 else 1
    if n <= _BANDWIDTH_EXACT_LIMIT:
        distances = pdist(D)
    else:
        rng = np.random.default_rng(random_state)
        subset_size = min(sample_size, n)
        subset_idx = rng.choice(n, size=subset_size, replace=False)
        distances = pdist(D[subset_idx])
    return max(float(np.percentile(distances, percentile)), 0.001)


def sampling(D, seed=42):
    """Randomly keep 90% of the samples."""
    D = _as_float_array(D)
    rng = np.random.default_rng(seed)
    idx = rng.choice(D.shape[0], size=max(1, int(D.shape[0] * 0.9)), replace=False)
    return D[idx], idx


def noising(D, seed=42):
    """Add Gaussian feature noise with the paper's default scale."""
    D = _as_float_array(D)
    rng = np.random.default_rng(seed)
    return D + rng.standard_normal(D.shape) * 0.05 * np.std(D, axis=0)


def _fit_qspp(D, k, beta, width):
    model = QuickshiftPP(k, beta, width)
    model.fit(_as_float_array(D))
    return model


def _group_core_indices(indicators):
    indicators = np.asarray(indicators)
    valid_mask = indicators != -1
    if not np.any(valid_mask):
        return []

    valid_labels = indicators[valid_mask]
    unique_labels = np.unique(valid_labels)
    return [np.flatnonzero(indicators == label) for label in unique_labels]


def _compute_density_and_delta(D, core_groups, point_density):
    num_cores = len(core_groups)
    if num_cores == 0:
        return np.empty(0), np.empty(0)

    densities = np.array([point_density[group].sum() for group in core_groups], dtype=float)
    centers = np.array([D[group].mean(axis=0) for group in core_groups], dtype=float)

    if num_cores == 1:
        return densities, np.zeros(1, dtype=float)

    dist_matrix = cdist(centers, centers)
    deltas = np.zeros(num_cores, dtype=float)
    for i in range(num_cores):
        higher = densities > densities[i]
        if np.any(higher):
            deltas[i] = dist_matrix[i, higher].min()
        else:
            other = np.arange(num_cores) != i
            deltas[i] = dist_matrix[i, other].max()
    return densities, deltas


def _select_cores(D, C, k, beta, width):
    D = _as_float_array(D)
    model = _fit_qspp(D, k, beta, width)
    core_groups = _group_core_indices(model.Mcore)
    if not core_groups:
        return []

    densities, deltas = _compute_density_and_delta(D, core_groups, np.asarray(model.den, dtype=float))
    scores = densities * deltas
    top_idx = np.argsort(scores)[::-1][: min(C, len(core_groups))]
    return [core_groups[i] for i in top_idx]


def _test_qspp_param(D, k_curr, beta_curr, width):
    """Helper for parallel QSPP parameter search."""
    model = _fit_qspp(D, k_curr, beta_curr, width)
    core_num = len(_group_core_indices(model.Mcore))
    return k_curr, beta_curr, core_num


def _find_qspp_params_parallel(D, C, width, n_jobs=-1):
    """Parallelized heuristic search for QSPP parameters."""
    n = D.shape[0]
    k_init = 10 if n <= 500 else max(2, int(0.02 * n))
    beta_init = 0.1

    search_space = [
        (k_init, beta_init),
        (k_init, 0.0),
        (max(k_init // 2, 2), 0.0),
        (max(k_init // 4, 2), 0.0),
        (2, 0.0),
    ]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_test_qspp_param)(D, k_curr, beta_curr, width)
        for k_curr, beta_curr in search_space
    )

    for k_curr, beta_curr, core_num in results:
        if core_num >= C:
            return k_curr, beta_curr
    return 2, 0.0


def _get_core_centers(D, core_groups):
    if len(core_groups) == 0:
        return np.empty((0, D.shape[1]), dtype=float)
    return np.array([D[group].mean(axis=0) for group in core_groups], dtype=float)


def _avg_internal_dist(centers):
    if len(centers) <= 1:
        return 1.0
    distances = pdist(centers)
    if distances.size == 0:
        return 1.0
    avg_dist = float(np.mean(distances))
    return avg_dist if avg_dist > 0 else 1.0


def _normalized_drift_distance(centers_1, centers_2):
    if len(centers_1) == 0 or len(centers_2) == 0:
        return 0.0

    cost_matrix = cdist(centers_1, centers_2)
    rows, cols = linear_sum_assignment(cost_matrix)
    d_om = float(cost_matrix[rows, cols].mean())

    scale = (_avg_internal_dist(centers_1) + _avg_internal_dist(centers_2)) / 2.0
    if scale == 0:
        return 0.0
    return d_om / scale


def _process_single_perturbation(D, perturb_type, sub_seed, C, k_qs, beta_qs, base_width, nbrs_5):
    """Worker task that computes a single perturbation's core clusters and centers."""
    n = D.shape[0]
    if perturb_type == "sample":
        D_k, idx_k = sampling(D, sub_seed)
        cores_k_local = _select_cores(D_k, C, k_qs, beta_qs, base_width)
        original_cores_k = [idx_k[group] for group in cores_k_local]
        centers_k = _get_core_centers(D_k, cores_k_local)

        core_mask = np.zeros(n, dtype=bool)
        for group in original_cores_k:
            core_mask[group] = True

        sampled_mask = np.zeros(n, dtype=bool)
        sampled_mask[idx_k] = True
        unsampled = np.flatnonzero(~sampled_mask)

        hitchhiked = []
        for point_idx in unsampled:
            sampled_nbrs = nbrs_5[point_idx][sampled_mask[nbrs_5[point_idx]]]
            if sampled_nbrs.size >= 2 and np.all(core_mask[sampled_nbrs]):
                hitchhiked.append(point_idx)
    else:
        D_k = noising(D, sub_seed)
        original_cores_k = _select_cores(D_k, C, k_qs, beta_qs, base_width)
        centers_k = _get_core_centers(D_k, original_cores_k)
        hitchhiked = []

    valid = len(original_cores_k) >= C
    return original_cores_k, centers_k, hitchhiked, valid


def _compute_nbrs_parallel(D, nbr_count, n_jobs):
    """并行构建 KNN 邻居索引。

    将数据集分块后各自在子进程中计算 KNN，然后在主进程拼合，
    相比在主进程中单线程计算 NearestNeighbors 可显著缩短等待时间。
    """
    n = D.shape[0]
    nbr_count = min(nbr_count, n)

    # 计算每个分块的查询点范围（数据集本身用于建树，只分块查询）
    cpu_cores = joblib.cpu_count()
    actual_jobs = cpu_cores if n_jobs == -1 else min(max(1, n_jobs), cpu_cores)
    chunk_size = max(1, n // actual_jobs)
    chunks = [
        (start, min(start + chunk_size, n))
        for start in range(0, n, chunk_size)
    ]

    def _query_chunk(D_full, query_start, query_end, nbr_count):
        nn = NearestNeighbors(n_neighbors=nbr_count)
        nn.fit(D_full)
        return nn.kneighbors(D_full[query_start:query_end], return_distance=False)

    results = Parallel(n_jobs=actual_jobs, backend="loky")(
        delayed(_query_chunk)(D, start, end, nbr_count)
        for start, end in chunks
    )
    return np.vstack(results)


def calculate_peak_uncertainty_online_parallel(
    D,
    C,
    M_T=0.01,
    M_p=0.05,
    K_max=50,
    seed=42,
    n_jobs=-1,
    batch_size=None,
    return_history=False,
):
    """Parallelized online estimation of Peak Uncertainty T and representative confidence.

    Preserves the exact sequential incremental update equations and early stopping criteria
    by pre-generating the perturbation stream and processing it in speculative parallel batches.

    Optimizations vs. the first version:
    1. KNN pre-computation is parallelized across CPU cores (_compute_nbrs_parallel).
    2. Process pool reuse strategy:
       - N < _PROCESS_REUSE_THRESHOLD: Parallel used as a context manager so the worker
         pool is kept alive across all batch iterations (saves process-spawn overhead for
         small datasets where memory accumulation is negligible).
       - N >= _PROCESS_REUSE_THRESHOLD: Each batch spawns and destroys its own worker
         pool so OS forcibly reclaims all memory after every batch (prevents OOM for
         large datasets where per-worker residual memory can accumulate to GBs).
    """
    D = _as_float_array(D)
    n = D.shape[0]

    # 1. Parallel parameter search
    base_width = compute_bandwidth(D, random_state=seed)
    k_qs, beta_qs = _find_qspp_params_parallel(D, C, base_width, n_jobs=n_jobs)

    T_k = 0.0
    P_k = np.zeros(n, dtype=float)
    prev_centers = None
    converged = False
    accepted_iters = 0
    T_history = []

    # 改进1：并行 KNN 预计算
    nbr_count = min(6, n)
    nbrs_5 = _compute_nbrs_parallel(D, nbr_count, n_jobs)[:, 1:]

    # 2. Pre-generate the perturbation schedule to ensure deterministic, mathematically identical results
    rng = np.random.default_rng(seed)
    tasks_info = []
    for idx in range(K_max):
        perturb_type = rng.choice(["sample", "noise"])
        sub_seed = int(rng.integers(1, 100000))
        tasks_info.append((perturb_type, sub_seed))

    # 3. Determine job count and batch size
    cpu_cores = joblib.cpu_count()
    actual_jobs = cpu_cores if n_jobs == -1 else min(max(1, n_jobs), cpu_cores)
    if batch_size is None:
        batch_size = max(2, actual_jobs)

    # 改进2：按样本量选择进程池重用策略
    reuse_pool = n < _PROCESS_REUSE_THRESHOLD

    def _run_batch(parallel_obj, batch_tasks):
        """将一批扰动任务分发给给定的 Parallel 对象执行。"""
        return parallel_obj(
            delayed(_process_single_perturbation)(
                D, p_type, s_seed, C, k_qs, beta_qs, base_width, nbrs_5
            )
            for p_type, s_seed in batch_tasks
        )

    def _process_batch_results(results):
        """顺序处理一批结果，执行增量更新和收敛判断。返回是否已收敛。"""
        nonlocal T_k, P_k, prev_centers, accepted_iters, converged
        for original_cores_k, centers_k, hitchhiked, valid in results:
            if not valid:
                continue

            accepted_iters += 1

            I_k = np.zeros(n, dtype=float)
            for group in original_cores_k:
                I_k[group] = 1.0
            if hitchhiked:
                I_k[hitchhiked] = 1.0

            P_prev = P_k.copy()
            P_k = P_prev + (I_k - P_prev) / accepted_iters

            if prev_centers is not None and len(prev_centers) > 0:
                d_k = _normalized_drift_distance(centers_k, prev_centers)
                T_prev = T_k
                T_k = T_prev + (d_k - T_prev) / max(accepted_iters - 1, 1)

                macro_diff = abs(T_k - T_prev)
                micro_diff = np.linalg.norm(P_k - P_prev, ord=np.inf)
                T_history.append(float(T_k))
                if macro_diff < M_T and micro_diff < M_p:
                    converged = True
                    return True

            prev_centers = centers_k
            if accepted_iters == 1:
                T_history.append(float(T_k))
        return False

    # 4. Process in speculative batches while maintaining sequential updating order
    task_idx = 0

    if reuse_pool:
        # 小样本：进程池常驻，跨批次重用，省去每批次的建立开销
        with Parallel(n_jobs=actual_jobs, backend="loky") as parallel:
            while task_idx < K_max and not converged:
                current_batch_size = min(batch_size, K_max - task_idx)
                batch_tasks = tasks_info[task_idx: task_idx + current_batch_size]
                results = _run_batch(parallel, batch_tasks)
                _process_batch_results(results)
                task_idx += current_batch_size
    else:
        # 大样本：每批次独立建立/销毁进程池，强制 OS 在批次间回收内存，防止 OOM
        while task_idx < K_max and not converged:
            current_batch_size = min(batch_size, K_max - task_idx)
            batch_tasks = tasks_info[task_idx: task_idx + current_batch_size]
            results = Parallel(n_jobs=actual_jobs, backend="loky")(
                delayed(_process_single_perturbation)(
                    D, p_type, s_seed, C, k_qs, beta_qs, base_width, nbrs_5
                )
                for p_type, s_seed in batch_tasks
            )
            _process_batch_results(results)
            task_idx += current_batch_size

    if not converged:
        print(
            f"[Warning] Dual criteria not met within K_max={K_max}. "
            "Data may be highly fragile."
        )
    result = (T_k, P_k, converged, accepted_iters)
    if return_history:
        return (*result, T_history)
    return result
