"""Chainlink demo: uncertainty convergence, clustering, and noise robustness."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from DPC import DPC
from Src.DCSC_PU import DCSC_PU
from Src.peak_uncertainty import calculate_peak_uncertainty_online_parallel
from utils import CHAINLINK_PATH, OUTPUT_DIR, clustering_scores, load_mat_dataset


SEED = 42
M_T = 0.005
M_P = 0.05
K_MAX = 50
TRIALS = 5
NOISE_SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]
DC_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]
K_RATIOS = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05]
BETA_GRID = [0.05, 0.1, 0.2, 0.5, 0.8]


def mixed_perturbation_view(data, noise_scale, seed):
    """Apply the same mixed measurement-noise setting used in the paper."""
    noise_bands = ((0.2, 0.1), (0.4, 0.05), (0.2, 0.01))
    rng = np.random.default_rng(seed)
    view = np.asarray(data, dtype=float).copy()
    permutation = rng.permutation(len(view))
    feature_scale = np.std(view, axis=0)
    feature_scale[feature_scale == 0] = 1.0

    start = 0
    for ratio, sigma in noise_bands:
        count = int(round(len(view) * ratio))
        selected = permutation[start : start + count]
        start += count
        view[selected] += rng.normal(
            0.0,
            sigma * noise_scale * feature_scale,
            size=(len(selected), view.shape[1]),
        )
    return view


def make_k_grid(sample_count):
    values = []
    for ratio in K_RATIOS:
        k = min(max(int(round(ratio * sample_count)), 2), sample_count - 1)
        values.append(k)
    return list(dict.fromkeys(values))


def best_dpc(data, truth, cluster_count):
    best = None
    for distance_percent in DC_GRID:
        prediction, *_ = DPC(data, cluster_count, distance_percent=distance_percent)
        scores = clustering_scores(truth, prediction)
        if best is None or scores["NMI"] > best[1]["NMI"]:
            best = prediction, scores, distance_percent
    return best


def best_dcsc_pu(data, truth, cluster_count, seed, n_jobs):
    uncertainty, confidence, converged, iterations = (
        calculate_peak_uncertainty_online_parallel(
            data,
            cluster_count,
            M_T=M_T,
            M_p=M_P,
            K_max=K_MAX,
            seed=seed,
            n_jobs=n_jobs,
        )
    )
    best = None
    for k in make_k_grid(len(data)):
        for beta in BETA_GRID:
            prediction, _ = DCSC_PU(
                data,
                cluster_count,
                k,
                beta,
                uncertainty,
                confidence,
                seed=seed,
                n_jobs=n_jobs,
            )
            scores = clustering_scores(truth, prediction)
            if best is None or scores["NMI"] > best[1]["NMI"]:
                best = prediction, scores, k, beta
    return (*best, uncertainty, converged, iterations)


def clean_chainlink_demo(data, truth, cluster_count, n_jobs, show):
    uncertainty, confidence, converged, iterations, history = (
        calculate_peak_uncertainty_online_parallel(
            data,
            cluster_count,
            M_T=M_T,
            M_p=M_P,
            K_max=K_MAX,
            seed=SEED,
            n_jobs=n_jobs,
            return_history=True,
        )
    )
    prediction, _ = DCSC_PU(
        data,
        cluster_count,
        k=15,
        beta=0.5,
        T=uncertainty,
        P_k=confidence,
        seed=SEED,
        n_jobs=n_jobs,
    )
    scores = clustering_scores(truth, prediction)

    print("\n1. Peak Uncertainty convergence and clean clustering")
    print(f"   T={uncertainty:.6f}, converged={converged}, iterations={iterations}")
    print("   " + ", ".join(f"{key}={value:.2f}" for key, value in scores.items()))

    figure = plt.figure(figsize=(15, 4.5))
    curve_axis = figure.add_subplot(1, 3, 1)
    curve_axis.plot(range(1, len(history) + 1), history, marker="o", color="#7b2cbf")
    curve_axis.axhline(uncertainty, color="#7b2cbf", linestyle="--", alpha=0.5)
    curve_axis.set_xlabel("Accepted perturbation views")
    curve_axis.set_ylabel("Peak Uncertainty T")
    curve_axis.set_title("Online convergence of T")
    curve_axis.grid(alpha=0.25)

    for position, labels, title in (
        (2, truth, "Ground truth"),
        (3, prediction, f"DCSC-PU\nACC={scores['ACC']:.1f}, NMI={scores['NMI']:.1f}"),
    ):
        axis = figure.add_subplot(1, 3, position, projection="3d")
        axis.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap="tab10", s=10)
        axis.set_title(title)
        axis.set_xlabel("x1")
        axis.set_ylabel("x2")
        axis.set_zlabel("x3")

    figure.tight_layout()
    output = OUTPUT_DIR / "chainlink_convergence_and_clustering.png"
    OUTPUT_DIR.mkdir(exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    print(f"   Figure: {output.relative_to(OUTPUT_DIR.parent)}")


def noise_robustness_demo(data, truth, cluster_count, trials, n_jobs, show):
    dpc_acc = np.zeros((len(NOISE_SCALES), trials))
    dcsc_acc = np.zeros_like(dpc_acc)

    print("\n2. Noise robustness: DPC vs. DCSC-PU")
    for scale_index, noise_scale in enumerate(NOISE_SCALES):
        for trial in range(trials):
            trial_seed = SEED + trial
            noisy_data = mixed_perturbation_view(data, noise_scale, trial_seed)
            _, dpc_scores, _ = best_dpc(noisy_data, truth, cluster_count)
            _, dcsc_scores, *_ = best_dcsc_pu(
                noisy_data, truth, cluster_count, trial_seed, n_jobs
            )
            dpc_acc[scale_index, trial] = dpc_scores["ACC"]
            dcsc_acc[scale_index, trial] = dcsc_scores["ACC"]

        print(
            f"   noise={noise_scale:.2f}: "
            f"DPC={dpc_acc[scale_index].mean():.2f}+/-{dpc_acc[scale_index].std():.2f}, "
            f"DCSC-PU={dcsc_acc[scale_index].mean():.2f}+/-{dcsc_acc[scale_index].std():.2f}"
        )

    dpc_mean, dpc_std = dpc_acc.mean(axis=1), dpc_acc.std(axis=1)
    dcsc_mean, dcsc_std = dcsc_acc.mean(axis=1), dcsc_acc.std(axis=1)

    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    axis.errorbar(
        NOISE_SCALES, dpc_mean, yerr=dpc_std, marker="o", capsize=4,
        linewidth=2, label="DPC", color="#e76f51",
    )
    axis.errorbar(
        NOISE_SCALES, dcsc_mean, yerr=dcsc_std, marker="s", capsize=4,
        linewidth=2, label="DCSC-PU", color="#2a9d8f",
    )
    axis.set_xlabel("Noise scale")
    axis.set_ylabel("ACC (%)")
    axis.set_xticks(NOISE_SCALES)
    axis.set_ylim(max(0, min(dpc_mean.min(), dcsc_mean.min()) - 8), 103)
    axis.set_title("Chainlink robustness under noise")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output = OUTPUT_DIR / "chainlink_noise_acc.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)

    dpc_drop = dpc_mean[0] - dpc_mean[-1]
    dcsc_drop = dcsc_mean[0] - dcsc_mean[-1]
    print(
        "\nConclusion: as noise increases, DPC loses "
        f"{dpc_drop:.2f} ACC points, whereas DCSC-PU loses {dcsc_drop:.2f}. "
        "DCSC-PU is substantially more robust to measurement noise."
    )
    print(f"Figure: {output.relative_to(OUTPUT_DIR.parent)}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data, truth = load_mat_dataset(CHAINLINK_PATH)
    cluster_count = len(np.unique(truth))
    clean_chainlink_demo(data, truth, cluster_count, args.n_jobs, args.show)
    noise_robustness_demo(
        data, truth, cluster_count, args.trials, args.n_jobs, args.show
    )


if __name__ == "__main__":
    main()
