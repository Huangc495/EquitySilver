# Phase 2 - Final sample counts

Samples surviving the complete-window rule, per station and input type. A sample is one acidity measurement; it is kept only if every day of its lookback window has both precipitation and mean temperature. Nothing is imputed.

Measurements available after cleaning: **BD 365**, **C7 384** (both exactly match Ma et al.).

## Default configuration (`common_sample_set: false`)

| station | type | n_features | candidates | surviving | dropped | pct_kept | first      | last       |
|---------|------|------------|------------|-----------|---------|----------|------------|------------|
| BD      | A    | 2          | 365        | 274       | 91      | 75.1     | 1998-12-04 | 2017-06-22 |
| BD      | B    | 2          | 365        | 185       | 180     | 50.7     | 1999-02-04 | 2016-09-28 |
| C7      | A    | 2          | 384        | 278       | 106     | 72.4     | 1998-04-09 | 2017-06-22 |
| C7      | B    | 2          | 384        | 186       | 198     | 48.4     | 1999-02-04 | 2016-09-28 |

This is the sample set the parametric study (Phase 6) will use.

## With the time tag (refined model, Phase 8)

| station | type | n_features | candidates | surviving | dropped | pct_kept | first      | last       |
|---------|------|------------|------------|-----------|---------|----------|------------|------------|
| BD      | A    | 3          | 365        | 274       | 91      | 75.1     | 1998-12-04 | 2017-06-22 |
| BD      | B    | 3          | 365        | 185       | 180     | 50.7     | 1999-02-04 | 2016-09-28 |
| C7      | A    | 3          | 384        | 278       | 106     | 72.4     | 1998-04-09 | 2017-06-22 |
| C7      | B    | 3          | 384        | 186       | 198     | 48.4     | 1999-02-04 | 2016-09-28 |

The time tag adds a third feature without changing which samples survive (verified), so tagged and untagged models are compared on identical samples.

## With `common_sample_set: true` (robustness check, Phase 6)

| station | type | n_features | candidates | surviving | dropped | pct_kept | first      | last       |
|---------|------|------------|------------|-----------|---------|----------|------------|------------|
| BD      | A    | 2          | 365        | 185       | 180     | 50.7     | 1999-02-04 | 2016-09-28 |
| BD      | B    | 2          | 365        | 185       | 180     | 50.7     | 1999-02-04 | 2016-09-28 |
| C7      | A    | 2          | 384        | 186       | 198     | 48.4     | 1999-02-04 | 2016-09-28 |
| C7      | B    | 2          | 384        | 186       | 198     | 48.4     | 1999-02-04 | 2016-09-28 |

Type A is restricted to the samples that also survive Type B, so the Type A vs Type B comparison isolates the input representation rather than confounding it with a change of sample set.

## Forecast and sensitivity periods (Phases 9-10)

| period              | n_samples |
|---------------------|-----------|
| train+val 1999-2014 | 161       |
| predict 2015-2016   | 24        |
| excluded            | 0         |

Station BD, Type B, with time tag. Both periods are usable, though the prediction period is small.

