# Dataset adapters

## ds004504

`configs/ds004504.yaml` is runnable after the OpenNeuro download. The adapter
reads `participants.tsv`, maps group `A` to AD and group `C` to HC, excludes
the FTD group, and locates each eyes-closed EEG file from the BIDS hierarchy.
It does not infer diagnoses from file ordering.

## CAUEEG

Use `configs/caueeg.example.yaml` with a local manifest containing the strict
AD and HC cohort selected for the paper. Keep phenotype recoding in a separate,
versioned manifest-generation script; do not embed private phenotype values in
this repository.

## ds005385

Use `configs/ds005385.example.yaml`. The manifest should identify the selected
pre-task, session-1 eyes-closed recording and include continuous age. Avoid
mixing pre-task and post-task recordings in one row per subject unless the
session/run is explicitly modeled.

## TD-BRAIN

Use `configs/tdbrain.example.yaml` with the current BDF/BDF+ release and current
participant metadata. The manifest should select healthy adults and the
eyes-closed resting-state condition. If artifact correction yields multiple
valid continuous blocks, pass their sample boundaries to peak extraction so
midpoint intervals and symbolic parsing remain within blocks.

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
