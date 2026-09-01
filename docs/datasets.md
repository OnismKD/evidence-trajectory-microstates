# Dataset adapters

## ds004504

Download snapshot 1.0.9 from
[OpenNeuro](https://openneuro.org/datasets/ds004504/versions/1.0.9), preferably
with `scripts/download_ds004504.py`. `configs/ds004504.yaml` requests the
public denoised files under `derivatives/`, which are the inputs used for the
PDF result. The adapter reads `participants.tsv`, maps group `A` to AD and
group `C` to HC, excludes FTD, and locates each eyes-closed EEG file. Set
`eeg_variant: raw` only for a separately labelled sensitivity analysis.

## CAUEEG

Request academic access through the
[official CAUEEG repository](https://github.com/ipis-mjkim/caueeg-dataset).
Place the unmodified download under `data/caueeg-dataset/`, then use
`configs/caueeg.example.yaml` with a local manifest containing the strict AD
and HC cohort selected for the paper. Preserve the official `event/*.json`
boundaries when extracting eyes-closed blocks. Keep phenotype recoding in a
separate, versioned manifest-generation script; do not embed controlled
participant annotations in this repository.

## ds005385

Download [OpenNeuro ds005385](https://openneuro.org/datasets/ds005385) into
`data/ds005385/`. Use `configs/ds005385.example.yaml`. The manifest should
identify the pre-task (`acq-pre`), session-1 (`ses-1`) eyes-closed
(`task-EyesClosed`) recording and include continuous age. Avoid mixing pre-task
and post-task recordings in one row per subject unless acquisition is modeled.

## TD-BRAIN

Accept the official data-use agreement on the
[TD-BRAIN download page](https://brainclinics.com/resources/tdbrain-dataset/introduction)
and place V3.1 under `data/tdbrain/`. Use `configs/tdbrain.example.yaml` with
the BDF/BDF+ release and current participant metadata. The manifest should
select healthy adults and `task-restEC`. Do not point the V3.1 configuration at
the discontinued derivative CSV release.

## Generic manifest contract

Required columns:

```text
subject_id,dataset,eeg_path
```

Optional analysis columns:

```text
group,age,sex,include
```

All subjects in a group-template analysis must use the same ordered channel
list. The code intentionally fails on a mismatch instead of silently
interpolating or dropping channels. Cross-dataset montage harmonization is not
required because models are fitted independently within each dataset.

Relative `eeg_path` values are resolved against the manifest's directory. This
makes a manifest portable: moving the repository only requires preserving the
layout under `data/`, while data stored elsewhere can use absolute paths.
Dataset roots may also be symbolic links, which is the simplest way to retain
the checked-in YAML files when EEG is kept on an external or mounted disk.
