# DCSC-PU

Code and data for the paper **"Beyond Observed Data: Towards Robust Density
Clustering under Peak Uncertainty."**

DCSC-PU estimates dataset-level Peak Uncertainty and point-level representative
confidence from perturbed views, then uses them to construct stable density-core
regions and produce a robust clustering result.

## Repository contents

- `Src/DCSC_PU.py`: main DCSC-PU implementation.
- `Src/peak_uncertainty.py`: estimation of Peak Uncertainty and representative confidence.
- `DPC.py`: Density Peak Clustering baseline used in the demo.
- `demo_chainlink.py`: convergence, clustering, and noise-robustness demonstration.
- `data/`: 71 synthetic and 17 original-feature real-world datasets used in the paper.
- `third_party/quickshiftpp/`: modified Quickshift++ source required by DCSC-PU.

## Installation

The experiments use Python 3.12. Create the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate dcsc-pu
```

The environment installs all Python dependencies and compiles the included
Quickshift++ extension.

## Chainlink demo

Run the complete demonstration with:

```bash
python demo_chainlink.py
```

The demo contains two parts:

1. **Peak Uncertainty and clean clustering.** It records the online estimate of
   Peak Uncertainty after every accepted perturbed view, plots its convergence,
   and displays the ground truth and DCSC-PU clustering result on Chainlink.
2. **Noise robustness.** Following the paper, it evaluates noise scales
   `0`, `0.25`, `0.5`, `0.75`, and `1.0`. Each setting is repeated five times,
   and the mean ACC with standard deviation is plotted for DPC and DCSC-PU.

Optional arguments:

```bash
python demo_chainlink.py --trials 5 --n-jobs 1 --show
```

- `--trials`: number of repeated noise trials; default is `5`.
- `--n-jobs`: number of parallel workers; use `-1` for all available cores.
- `--show`: display figures interactively in addition to saving them.

The generated figures are:

- `outputs/chainlink_convergence_and_clustering.png`
- `outputs/chainlink_noise_acc.png`

With the default seed and five trials, Peak Uncertainty converges to `0.392384`
after 22 accepted views, and DCSC-PU obtains 100% ACC/NMI/ARI on clean
Chainlink. As the noise scale increases from 0 to 1, DPC's mean ACC decreases
from 100.00% to 77.06%, while DCSC-PU remains at 100.00%. This result
demonstrates the stronger noise robustness of DCSC-PU.

## Data

MAT files store features under `data` or `X` and labels under `label` or
`labels`. The demo applies feature-wise min-max normalization at runtime. See
`data/README.md` for the real-world dataset inventory.

## License

The repository is released under the MIT License. The modified Quickshift++
source under `third_party/quickshiftpp/` retains its Apache-2.0 license.
