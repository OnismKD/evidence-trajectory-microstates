# Evidence-Trajectory Microstates

This repository contains the code accompanying our paper on evidence-aware
temporal descriptors for EEG topographic state models. It implements matched
Hard-label, Evidence-trajectory, and confidence-aware Lempel-Ziv complexity
(LZC) readouts for fixed-`K` models (KMeans, AAHC, and HMM) and adaptive
community models (Leiden and Infomap).

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/OnismKD/evidence-trajectory-microstates.git
cd evidence-trajectory-microstates
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[full,download,plot]'
```

For exact Linux x86-64 compatibility with the numerical environment used for
the paper results, create the locked conda environment instead:

```bash
conda env create -f environment-paper.yml
conda activate evidence-trajectory-paper
python -m pip install -e . --no-deps
```

`requirements-paper.txt` records the Python package versions for other
platforms, but their BLAS implementations may produce small differences in a
few near-degenerate clustering solutions.

## Quick test

Run the data-free demonstration:

```bash
evidence-microstates demo --output-dir outputs/demo
```

## Reproduce the ds004504 result

We recommend validating the installation with
[OpenNeuro ds004504, snapshot 1.0.9](https://openneuro.org/datasets/ds004504/versions/1.0.9),
because it can be downloaded directly without an access application.

```bash
python scripts/download_ds004504.py \
  --output data/ds004504 \
  --input derivatives

evidence-microstates extract --config configs/ds004504.yaml

python scripts/reproduce_paper_ds004504.py \
  --config configs/ds004504.yaml \
  --n-jobs 8
```

The full state-model fit is computationally intensive and resumes from completed
per-subject feature files if interrupted.

The verifier checks the selected parameters and the classification results
reported in the paper:

| Feature family | Balanced accuracy | ROC-AUC |
|---|---:|---:|
| Hard | 0.806 +/- 0.096 | 0.862 +/- 0.089 |
| Trajectory | 0.828 +/- 0.081 | 0.854 +/- 0.092 |
| Combined | 0.839 +/- 0.091 | 0.888 +/- 0.083 |

Results are written to `outputs/ds004504/paper_reproduction/`.

## Other paper datasets

The remaining datasets must be downloaded from their official sources:

| Dataset | Source |
|---|---|
| CAUEEG | [Official repository and access instructions](https://github.com/ipis-mjkim/caueeg-dataset) |
| ds005385 | [OpenNeuro ds005385](https://openneuro.org/datasets/ds005385) |
| TD-BRAIN V3.1 | [Official download and data-use agreement](https://brainclinics.com/resources/tdbrain-dataset/introduction) |

Place downloaded data under `data/`, or provide absolute EEG paths in a CSV
manifest. Copy the relevant example configuration and update its
`manifest_path` if necessary:

```bash
cp configs/caueeg.example.yaml configs/caueeg.yaml
cp configs/ds005385.example.yaml configs/ds005385.yaml
cp configs/tdbrain.example.yaml configs/tdbrain.yaml
```

Each manifest requires `subject_id`, `dataset`, and `eeg_path`; optional fields
include `group`, `age`, `sex`, and `include`. See
[`docs/datasets.md`](docs/datasets.md) for dataset-specific file selection and
[`docs/methods.md`](docs/methods.md) for the analysis definitions.

## General workflow

```bash
evidence-microstates extract --config configs/<dataset>.yaml
evidence-microstates analyze --config configs/<dataset>.yaml
```

Predictive validation can be run with:

```bash
python scripts/run_predictive_validation.py --help
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The code is
released under the BSD 3-Clause License. Dataset files are distributed
separately by their original providers and are not included in this repository.
