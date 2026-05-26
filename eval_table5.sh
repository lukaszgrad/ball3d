#!/usr/bin/env bash
# eval_table5.sh — reproduce one camera-group column of paper Table 5.
# Each row in the output CSV is a model; values are macro-averaged 3D mean error (m)
# for Full / Straight / Arc frame groups.
#
# Assumes ball_3d.<version>.csv already exists under each ROOT's track/ folder.
# Runs evaluate_trajectory.py per (root, model) and writes a macro-averaged CSV.
#
# Paper Table 5 reports two camera groups for LP (static and broadcast) side by side.
# Run this script twice (once per group) and combine the two CSVs to get the full table.
#
# Env vars:
#   OUTPUT_CSV     — output path (default: logs/table5.csv)
#   SKIP_EXISTING  — set to 1 to skip eval if gt_metrics-<ver>.json already exists
set -euo pipefail

# Paper Table 5 reports only the two best-performing models. Add more rows if needed.
MODELS=(
    "basic_kinetic_fitg:basic_fitg"
    "basic_angular_velocity:basic_angular"
)

# One ROOT per camera-half within the camera group being reproduced.
# LP-broadcast (default): 1 camera × 2 halves.
# ROOTS=(
#     "data/legia_warszawa-piast_gliwice-20251214/camera00/half_1"
#     "data/legia_warszawa-piast_gliwice-20251214/camera00/half_2"
# )
# LP-static: 5 cameras × 2 halves — uncomment to use:
ROOTS=(
    "data/legia_warszawa-piast_gliwice-20251214/camera01/half_1"
    "data/legia_warszawa-piast_gliwice-20251214/camera01/half_2"
    "data/legia_warszawa-piast_gliwice-20251214/camera02/half_1"
    "data/legia_warszawa-piast_gliwice-20251214/camera02/half_2"
    "data/legia_warszawa-piast_gliwice-20251214/camera03/half_1"
    "data/legia_warszawa-piast_gliwice-20251214/camera03/half_2"
    "data/legia_warszawa-piast_gliwice-20251214/camera04/half_1"
    "data/legia_warszawa-piast_gliwice-20251214/camera04/half_2"
    "data/legia_warszawa-piast_gliwice-20251214/camera05/half_1"
    "data/legia_warszawa-piast_gliwice-20251214/camera05/half_2"
)

OUTPUT_CSV="${OUTPUT_CSV:-logs/table5.csv}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# --- 1. Per-(root, model) evaluation ---
for root in "${ROOTS[@]}"; do
    if [[ ! -d "$root" ]]; then
        echo "WARN: $root does not exist, skipping" >&2
        continue
    fi
    for pair in "${MODELS[@]}"; do
        traj="${pair%:*}"
        ver="${pair#*:}"
        out="$root/eval/gt_metrics-${ver}.json"
        if [[ "$SKIP_EXISTING" == "1" && -f "$out" ]]; then
            echo "[skip] $out already exists"
            continue
        fi
        echo "[eval] $root × $ver"
        uv run python evaluate_trajectory.py \
            root="$root" trajectory="$traj" version="$ver"
    done
done

# --- 2. Macro-average across roots, write Table 5 CSV ---
mkdir -p "$(dirname "$OUTPUT_CSV")"

ROOTS_STR=$(printf '%s\n' "${ROOTS[@]}")
MODELS_STR=$(printf '%s\n' "${MODELS[@]}")

OUTPUT_CSV="$OUTPUT_CSV" ROOTS_STR="$ROOTS_STR" MODELS_STR="$MODELS_STR" \
    uv run python - <<'PYEOF'
import csv, json, os
from pathlib import Path

output_csv = os.environ['OUTPUT_CSV']
roots = [r for r in os.environ['ROOTS_STR'].splitlines() if r]
models = [m for m in os.environ['MODELS_STR'].splitlines() if m]

FIELDS = ('full_err', 'str_err', 'arc_err')

rows = []
for pair in models:
    traj, ver = pair.split(':', 1)
    acc = {k: [] for k in FIELDS}
    for root in roots:
        p = Path(root) / 'eval' / f'gt_metrics-{ver}.json'
        if not p.exists():
            print(f'WARN: missing {p}')
            continue
        d = json.loads(p.read_text())
        for k in FIELDS:
            acc[k].append(d[k])
    if not acc[FIELDS[0]]:
        continue
    rows.append((ver,) + tuple(round(sum(acc[k])/len(acc[k]), 4) for k in FIELDS))

with open(output_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['model', 'full_err_m', 'str_err_m', 'arc_err_m'])
    for row in rows:
        w.writerow([row[0]] + [f'{v:.3f}' for v in row[1:]])

print(f'\nWrote {output_csv}')
print(f'{"model":30}  {"Full":>7}  {"Str":>7}  {"Arc":>7}')
for ver, full, str_, arc in rows:
    print(f'{ver:30}  {full:>7.3f}  {str_:>7.3f}  {arc:>7.3f}')
PYEOF
