# Frozen paper references

These small CSV files record the ds004504 feature choices and predictive
results shown in Table 3 of the manuscript PDF. They contain no participant
data. `scripts/reproduce_paper_ds004504.py` treats them as an executable
regression target and fails when the selected `g,p` features, cohort size,
feature counts, or three-decimal classification results change.

The values were traced to the original complete subject-level feature grid.
The PDF protocol uses 36 AD and 29 HC participants, 16 Hard predictors, 16
Trajectory predictors, and their 32-feature concatenation.
