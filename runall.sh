#!/usr/bin/env bash
# Replicate results/final_v1 exactly: 18 conditions, 896 analyses, ~$145.
#
#   ./runall.sh                 # the whole study, Layer 1 -> Layer 2 -> figures -> table
#   ./runall.sh --dry-run       # print the 18-condition matrix and stop
#   ./runall.sh --layer2-only   # re-derive Layer 2 from banked Layer 1 (needs traces + keys)
#   ./runall.sh --report-only   # regenerate figures and the table from banked Layer 2
#
# Layer 1 calls paid model APIs. Layer 2 is digest-bound, so re-running it over
# an unchanged root makes no API calls and reproduces byte-identical output.
#
# Exactness, and which mode reproduces what.
#
# --report-only is deterministic and offline. It reads the banked Layer-2
# outputs (results_final.json) and regenerates every figure and the paper
# table byte-for-byte. No model calls, no keys.
#
# --layer2-only re-derives Layer 2 from Layer 1. Its extraction and scope
# banks are digest-bound to the *trace* content, so it needs the Layer-1
# traces present and API keys set. Run against a root whose traces are absent
# it will silently re-judge and produce different numbers -- it is not a
# trace-free replication path.
#
# A full run is a fresh sample and will not match analysis-for-analysis: model
# outputs are not deterministic at these settings. It is also not a no-op over
# the banked root -- bt_glm_naive and eu_glm_naive each hold 50 analyses of
# which one failed to report, so the grid counts 49 and tops each up by one.
#
# Keys: put OPENAI_API_KEY / ANTHROPIC_API_KEY / OPENROUTER_API_KEY in the
# environment, or NAME=value lines in a gitignored keys.txt beside this file.
set -Eeuo pipefail

readonly ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PYTHON="${BRANCHPOINT_PYTHON:-python3}"
readonly KEYS_FILE="${BRANCHPOINT_KEYS_FILE:-$ROOT/keys.txt}"
readonly OUT="${BRANCHPOINT_OUTPUT_ROOT:-results/final_v1}"

# The exact grid. n=50 and seed=7 throughout; these are the values the banked
# results were produced under and changing either makes the run a different
# study. Naive is the control; Branchpoint is uniform allocation over open
# decisions; BpDynamic ("spread") draws the least-tried decision first.
readonly N=50
readonly SEED=7
readonly GUARD=400
readonly DATASETS=(bt eu)
readonly MODELS=(sol opus glm)
readonly ARMS=(naive continuation spread)

trap 'rc=$?; printf "\nrunall stopped (exit %d). Banked work resumes on the next run.\n" "$rc" >&2; exit "$rc"' ERR
trap 'printf "\nInterrupted. Banked work resumes on the next run.\n" >&2; exit 130' INT TERM

load_keys() {
  [[ -f "$KEYS_FILE" ]] || return 0
  local line name value
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
    name=${line%%=*}; value=${line#*=}
    case "$name" in
      OPENAI_API_KEY|ANTHROPIC_API_KEY|ANTHROPIC_WORKSPACE_ID|OPENROUTER_API_KEY)
        [[ -z "${!name:-}" ]] && export "$name=$value" ;;
    esac
  done < "$KEYS_FILE"
}

conditions() {
  local ds model arm
  for ds in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for arm in "${ARMS[@]}"; do
        printf -- '--condition\n%s_%s_%s:%s:%s:%s:%s:%s\n' \
          "$ds" "$model" "$arm" "$ds" "$model" "$arm" "$N" "$SEED"
      done
    done
  done
}

cd -- "$ROOT"
mode=full
case "${1:-}" in
  --dry-run)     mode=dry ;;
  --layer2-only) mode=layer2 ;;
  --report-only) mode=report ;;
  -h|--help)     sed -n '2,12p' "$0"; exit 0 ;;
  "")            ;;
  *)             printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
esac

mapfile -t CONDITIONS < <(conditions)

if [[ "$mode" == full || "$mode" == dry ]]; then
  load_keys
  # Layer 1 plus its own Layer 2 pass. Resumable: a condition already at its
  # target is skipped, so an interrupted run is restarted by re-invoking this.
  "$PYTHON" -m experiments.launch_grid "${CONDITIONS[@]}" \
    --output-root "$OUT" --concurrency 4 --workers 6 --guard "$GUARD" \
    --postprocess $([[ "$mode" == dry ]] && printf -- --dry-run)
  [[ "$mode" == dry ]] && exit 0
fi

# Layer 2 over the whole root, then the reportable artifacts. Safe to repeat.
if [[ "$mode" != report ]]; then
  "$PYTHON" -m experiments.layer2 --root "$OUT" --guard "$GUARD" --workers 6
fi
"$PYTHON" -m experiments.figs --root "$OUT"
"$PYTHON" -m scripts.make_coverage_table --root "$OUT" --out paper/table_coverage_spread.tex

printf '\nDone. %s/ holds the 18 conditions; figures in %s/figures; table in paper/.\n' "$OUT" "$OUT"
