# Evidence-Trajectory Microstates

Research code for comparing conventional hard-label EEG topographic-state
descriptors with matched evidence-aware readouts. The same fitted state model
produces both representations:

- **Hard label:** the state with maximum template similarity at each GFP peak.
- **Evidence trajectory:** the full vector of template similarities, optionally
  sharpened and summarized as high-evidence temporal episodes.
- **Confidence-aware LZC:** a symbolic sequence with a null state when the
  selected state does not satisfy a predefined evidence or percentile rule.

The implementation supports the paper's two complementary state-model
families: classic fixed-`K` microstates (KMeans, AAHC, and HMM at `K=4` and
`K=7`) and subject-adaptive graph communities (Leiden and Infomap).

## Repository layout

```text
configs/                  Dataset and analysis settings
data/                     Local data only; raw EEG is git-ignored
docs/                     Method and dataset contracts
examples/                 Minimal runnable examples
scripts/                  Download and predictive-validation entry points
src/evidence_microstates/ Reusable implementation
tests/                    Numerical and workflow tests
```

The exploratory scripts and machine-specific output caches used during method
development are intentionally excluded. This repository contains the compact
analysis path needed to reproduce the method rather than a snapshot of a local
filesystem.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/OnismKD/evidence-trajectory-microstates.git
cd evidence-trajectory-microstates
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[full,download,plot]'
```

The base installation includes the KMeans workflow. The `full` extra adds
AAHC, HMM, Leiden, and Infomap.

## One-minute smoke test

This does not require EEG data:

```bash
evidence-microstates demo --output-dir outputs/demo
```

It generates polarity-varying synthetic peak topographies, fits a two-level
KMeans `K=4` model, backfits the group templates, and writes matched hard-label,
trajectory, and null-LZC descriptors.

## Public ds004504 example

The repository uses OpenNeuro
[`ds004504`](https://openneuro.org/datasets/ds004504) as the public example.
The downloader selects the 36 AD participants and 29 healthy controls and
retrieves their eyes-closed EEG plus BIDS metadata.

```bash
python scripts/download_ds004504.py --output data/ds004504
evidence-microstates extract --config configs/ds004504.yaml
evidence-microstates analyze --config configs/ds004504.yaml
```

For a quick test, download a few subjects and temporarily restrict
`fixed_algorithms` to `[kmeans]`:

```bash
python scripts/download_ds004504.py --output data/ds004504 --subjects 1-4,37-40
```

Main outputs are:

```text
outputs/ds004504/manifest_snapshot.csv
outputs/ds004504/peaks/*_peaks.npz
outputs/ds004504/models/*_templates.npy
outputs/ds004504/features_group_two_level.csv
outputs/ds004504/features_subject_level.csv
```

## Analysis contract

### Preprocessing and samples

Within each dataset, continuous EEG is resampled as configured, optionally
receives a broad-band filter, is filtered to 8-13 Hz, linearly detrended, and
common-average referenced. GFP is the across-channel standard deviation. All
local GFP maxima separated by at least 10 ms are retained. A peak's temporal
weight is the interval between adjacent peak midpoints, so duration summaries
remain in seconds even though the state sequence is indexed by peaks.

### State models

For fixed-`K` analyses, each subject is first fitted independently. The
resulting subject templates are pooled and clustered at a second level to form
group templates, which are then backfitted to all peak maps. HMM state maps are
aligned at the second level with polarity-invariant KMeans because unordered
topographies do not define a second-level temporal HMM.

Leiden and Infomap instead operate on a subject-specific polarity-invariant
k-nearest-neighbor graph. Their community templates are affinity-weighted
means and their number of states is not fixed.

### Matched readouts

For peak topography `x_t` and template `m_k`, evidence is

```text
s[t, k] = abs(corr_space(x_t, m_k))
p[t, k] = s[t, k]^gamma / sum_j s[t, j]^gamma
```

The hard label is `argmax_k s[t, k]`. For each state, a high-evidence episode
is a contiguous run where `p[t, k]` exceeds that state's weighted percentile
threshold. Mean duration and occurrence are computed from these runs after the
same minimum-duration rule used for hard labels. See
[`docs/methods.md`](docs/methods.md) for the exact definitions and null-LZC
variants.

## Statistical and predictive validation

Group-level effect analyses use the two-level group templates. Predictive
validation uses **subject-native global features** by default. Reusing a group
template fitted to the complete cohort inside cross-validation would expose
held-out subjects to template learning. An alternative is to refit the entire
two-level model within every outer training fold, which is substantially more
expensive and must also map test subjects without updating the templates.

Run paired nested CV on the subject-level output:

```bash
python scripts/run_predictive_validation.py \
  --features outputs/ds004504/features_subject_level.csv \
  --task ad_hc \
  --repeats 10 \
  --output-dir outputs/ds004504/prediction
```

For aging datasets, `--task young_old` defines young adults as `<35` years and
older adults as `>60` years. `--task age_regression` retains the continuous age
outcome. Scaling, median imputation, and regularization tuning occur inside the
inner CV loop. The same outer splits are used for Hard, Trajectory, and Combined
feature families.

Paired fold scores are useful sensitivity summaries, but repeated-CV folds are
not statistically independent. Confirmatory inference should therefore also
report subject-level permutation tests or a corrected resampled test.

## Other datasets

CAUEEG, ds005385, and TD-BRAIN are represented by placeholder YAML files. They
expect a local manifest with at least `subject_id`, `eeg_path`, and optionally
`group`, `age`, `sex`, and `include`. No controlled or locally licensed data are
redistributed. See [`docs/datasets.md`](docs/datasets.md).

## Reproducibility notes

- Raw EEG and generated outputs are excluded by `.gitignore`.
- Random seeds are explicit in every YAML file.
- Hard and evidence-aware descriptors are always derived from the same state
  model.
- Fixed-`K` state-specific features are valid after template alignment;
  subject-adaptive states should be summarized with label-invariant global
  descriptors unless an explicit alignment procedure is added.
- Parameter grids should be declared before outcome analysis. Dataset-specific
  optimization without nested selection changes the estimand and can inflate
  apparent performance.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The code is
released under the BSD 3-Clause License. The ds004504 data are distributed
separately by OpenNeuro under CC0; cite the dataset authors and descriptor when
using those recordings.
