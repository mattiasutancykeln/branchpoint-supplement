# Branchpoint — supplementary code and data

Everything needed to reproduce the reported study, and nothing else.

## Contents

| Path | What |
|---|---|
| `branchpoint/` | The sampler: decision detection, forking, ban lists. |
| `harness/` | The analysis loop and the provider-native continuation recorder. |
| `experiments/` | Grid launcher, Layer 2, figures, and the per-dataset questions. |
| `scripts/` | Table generation. |
| `research/scratch/manyecoevo/` | The two analyst-facing datasets and the human comparator (ManyEcoEvo, Gould et al. 2025; GPL-3, Zenodo 10.5281/zenodo.10046153). |
| `results/final_v1/` | The 18 conditions as reported: numerics and figures. |
| `paper/` | The experiment-setup and results sections, and the generated table. |
| `runall.sh` | Replication entry point. |

## Reproduce

```sh
./runall.sh --report-only   # figures + table from banked Layer 2 — offline, deterministic
./runall.sh --dry-run       # print the 18-condition matrix, no API calls
./runall.sh                 # the full study from scratch (paid model APIs, ~$145)
```

**`--report-only` is the replication path for this package.** It needs no API
keys and no network, and regenerates all 15 figures and
`paper/table_coverage_spread.tex` byte-for-byte from the banked Layer-2 outputs
shipped here. Verified by running it in a clean copy with the provider keys
unset: 15/15 figures and the table identical, and all 18 `results_final.json`
left untouched.

`--layer2-only` re-derives Layer 2 from Layer 1 and is **not** usable from this
package. Its extraction and scope banks are digest-bound to Layer-1 *trace*
content, which is not included (see below). Against a trace-free root it
silently re-judges and produces different numbers — in our check
`bt_opus_naive` moved from 0.446 to 0.279. The mode is listed for completeness,
for use against a root that does have traces.

A full run needs `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY`
in the environment or in a `keys.txt` beside `runall.sh`. It is a fresh sample
and will not match the banked results analysis-for-analysis: model outputs are
not deterministic at these settings. It is also not a no-op over the banked
root — `bt_glm_naive` and `eu_glm_naive` each hold 50 analyses of which one
failed to report, so the grid counts 49 and tops each up by one.

Python 3.11+ with `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`, and
the provider SDKs.

## What is not here

Layer-1 traces, forks and snapshots — roughly 930 MB of working state. The
study's convention is that results are numerics; the traces are the
intermediate transcripts that produced them. Every reported number is
re-derivable from what is shipped via `--report-only`, but re-running Layer 2
itself is not, for the digest reason above.

The MARP (religiosity) dataset is configured in `experiments/config.toml` but is
excluded from this study throughout, and its data file is not included.

## Reported numbers

18 conditions: 2 datasets × 3 models × {naive, Branchpoint, BpDynamic}, n=50
each, 896 analyses, $145 total.
