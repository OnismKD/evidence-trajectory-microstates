# Method definitions

## Fixed-K two-level microstates

For KMeans, AAHC, and HMM, `K=4` and `K=7` are fitted separately. Each
subject's complete set of valid alpha-band GFP-peak topographies is first used
to estimate subject templates. Subject templates are then pooled and clustered
at a second level to obtain group templates. Every recording is backfitted to
those group templates using polarity-invariant absolute spatial correlation.

HMM uses a Gaussian emission model after deterministic polarity
canonicalization. Viterbi state maps are converted to subject templates. Since
these templates are unordered, their second-level alignment uses
polarity-invariant KMeans rather than a second temporal HMM.

## Subject-adaptive communities

Leiden and Infomap use a polarity-invariant k-nearest-neighbor graph built from
each subject's peak topographies. Community templates are polarity-aligned,
within-community affinity-weighted averages. The number and identity of
communities may vary between subjects, so only global label-invariant
descriptors are used without an additional template alignment step.

## Hard-label descriptors

Let `z_t` denote the row-wise maximum-evidence state at GFP peak `t`, and let
`w_t` be the peak's midpoint interval in seconds. Consecutive equal labels form
segments. Segment duration is the sum of `w_t` over the segment. Global mean
duration is the arithmetic mean across all segments; state-specific mean
duration is the same calculation restricted to one aligned state. Occurrence
is the number of segments per second, and coverage is the fraction of total
weight assigned to a state.

Runs shorter than the configured minimum duration (30 ms in the paper
configuration) are deterministically merged within a continuous recording.
No run can cross a declared block boundary.

## Evidence-trajectory descriptors

For absolute spatial similarity `s_tk`, the state evidence trajectory is

```text
p_tk = s_tk^gamma / sum_j s_tj^gamma.
```

For each state, a weighted percentile threshold is estimated from that state's
own trajectory. A high-evidence episode is a contiguous run above that
threshold. Mean duration and occurrence use the same midpoint weights and
minimum-duration rule as hard-label descriptors. States may overlap; points at
which no state exceeds its threshold are tracked as the trajectory null
fraction but are not inserted into duration calculations.

## Lempel-Ziv complexity

Classic LZC is calculated from the hard-label peak sequence using LZ76
exhaustive-history parsing and normalized by sequence length and observed
alphabet size:

```text
C_norm = C_LZ * log(n) / (n * log(alpha)).
```

The package exposes two confidence-aware null-sequence rules:

1. `percentile_max`: convert each state's evidence to a within-state weighted
   percentile, select the state with the highest percentile at each peak, and
   emit null if that maximum percentile does not exceed the threshold.
2. `evidence_quantile_self`: select the percentile winner, then retain it only
   if its sharpened evidence exceeds that state's weighted evidence quantile.
   This is the configurable implementation corresponding to the latest
   percentile-self sensitivity analysis.

Null is treated as an additional symbol before normalized LZC is calculated.
The null fraction must be reported because it changes the effective symbolic
alphabet and provides an interpretable check against a degenerate all-null or
never-null sequence.

The functions `scan_duration_grid` and `scan_null_lzc_grid` in
`evidence_microstates.sensitivity` evaluate predeclared gamma/percentile grids
without refitting the underlying state model. Duration and LZC grids are kept
separate because their optimal percentile ranges need not coincide.

## Inference

AD-HC comparisons report Welch-test p-values and Cohen's d signed as AD minus
HC. Lifespan analyses report Pearson's r with age. Comparisons between matched
descriptors use absolute effect magnitude, because a stronger negative and a
stronger positive association both indicate a larger statistical effect.

Predictive models use subject-native global descriptors. Outer repeated
cross-validation estimates held-out performance; imputation, scaling, and
regularization tuning are restricted to each outer training split. Young-old
classification includes subjects younger than 35 years or older than 60 years.
