# Source paper

This repository replicates:

> Ma, L., Huang, C., Liu, Z.-S., Morin, K.A., Aziz, M., Meints, C. (2021).
> *The correlation between drainage chemistry and weather for full-scale waste
> rock piles based on artificial neural network.*
> **Journal of Contaminant Hydrology** 239, 103793.
> https://doi.org/10.1016/j.jconhyd.2021.103793

## The PDF is not tracked

`CLAUDE.md` expects the paper at `docs/Ma_2021_JCH.pdf`, and the code and
reports refer to it by that path. The file is **deliberately not committed**:
it is a copyrighted Elsevier article and this repository is public.

`docs/*.pdf` is gitignored. To work with the paper locally, place your own
copy at:

```
docs/Ma_2021_JCH.pdf
```

Access it through the DOI above, via your institution's subscription, or from
ScienceDirect. Nothing in the pipeline reads the PDF at runtime — it is a
reference for interpreting the replication, so the code runs without it.

## What the paper provides

Every target this replication is measured against is reproduced as data in the
code rather than quoted from the paper:

| Constant | Module | Contents |
|---|---|---|
| `PAPER_TABLE1` | `experiments.py` | Table 1 best-of-5 MSE, both stations, Types A and B, H = 5/10/20 |
| `PAPER_FIG6_R` | `experiments.py` | Fig. 6 correlation at Type B, H = 10 |
| `PAPER_FC_R`, `PAPER_LSTM_R` | `experiments.py` | Section 3.4 fully connected baseline comparison |
| `PAPER_ORIGINAL`, `PAPER_REFINED` | `experiments.py` | Section 3.5 time-tag model |
| `PAPER_FORECAST` | `forecast.py` | Section 3.6 two-year forecast |

`tests/test_report_phase11.py` asserts these match the values recorded in
`CLAUDE.md`, so a typo in a target cannot silently flatter the replication.

See `reports/replication_report.md` for the results and `DEVIATIONS.md` for
every departure from the paper.
