# ManyEcoEvo data provenance (F-D1)

Fetched 2026-08-21 by shallow clone of https://github.com/egouldo/ManyEcoEvo
(GPL-3; Zenodo DOI 10.5281/zenodo.10046153), clone HEAD
7e2d6eaf24bd46413fa5a1d67eeb0bc61c335b57. Paper: Gould et al. 2025,
"Same data, different analysts: variation in effect sizes due to analytical
decisions in ecology and evolutionary biology," BMC Biology,
DOI 10.1186/s12915-024-02101-x.

`extracted/` (committed) holds the working copies:
- `blue_tit_data.csv` (3720×42) — from data/blue_tit_data.rda via pyreadr.
- `euc_data.csv` (351×43) — from data/euc_data.rda via pyreadr.
- `master_data.csv` (302 coded analyses × 155 cols) + `master_metadata.csv`
  — the per-analysis decision coding (model class, transformation, effect
  structure, exclusions, conclusions), from data-raw/anonymised_data/.
- `bt_reviews.csv`, `euc_reviews.csv` — peer ratings of each analysis.
- `predictions_validations_worksheet*.csv` — out-of-sample prediction checks.

The full `ManyEcoEvo/` and `ManyAnalysts/` clones were removed in the
2026-08-26 rescope cleanup (only `ManyAnalysts/index.qmd` is kept as
the cited primary source); re-clone from GitHub at the recorded SHA to
reproduce. The nested tibble objects
(ManyEcoEvo.rda etc.) are unreadable by pyreadr (list-columns) — the flat
data-raw CSVs above carry the same coding content.
