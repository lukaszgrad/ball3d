#!/usr/bin/env bash
# eval_table3.sh — reproduce one dataset column of paper Table 3.
# Each row in the output CSV is a model; the values are macro-averaged across ROOTS.
#
# Assumes ball_3d.<version>.csv already exists under each ROOT's track/ folder.
# Runs evaluate_trajectory.py per (root, model) and writes a macro-averaged CSV.
#
# Edit MODELS and ROOTS below, then run from the ball3d repo root:
#     ./eval_table3.sh
#
# Env vars:
#   OUTPUT_CSV      — output path (default: logs/table3.csv)
#   SKIP_EXISTING   — set to 1 to skip per-(root, model) eval when gt_metrics-<ver>.json exists
set -euo pipefail

# 7 paper models, in Table 3 row order. Format: trajectory_config:version
MODELS=(
    "basic_kinetic_estimator_parabola:basic_parabola"
    "basic_kinetic_estimator:basic_kinetic"
    "basic_kinetic_fitg:basic_fitg"
    "basic_angular_velocity:basic_angular"
    "mujoco_kinetic_estimator:mujoco_kinetic"
    "mujoco_angular_velocity:mujoco_angular"
    "mujoco_ellipsoid_angular_velocity:mujoco_ellipsoid"
)

# One ROOT per camera-half. The final macro-average is taken across these.
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
    "data/legia_warszawa-piast_gliwice-20251214/camera06/half_1"
    "data/legia_warszawa-piast_gliwice-20251214/camera06/half_2"
)

OUTPUT_CSV="${OUTPUT_CSV:-logs/table3.csv}"
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
            root="$root" trajectory="$traj" version="$ver" \
            || echo "[skip] eval failed for $root × $ver (likely missing ball_3d.${ver}.csv)" >&2
    done
done

# --- 2. Macro-average across roots, write Table 3 CSV ---
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

rows = []
for pair in models:
    traj, ver = pair.split(':', 1)
    bals, arcs = [], []
    for root in roots:
        p = Path(root) / 'eval' / f'gt_metrics-{ver}.json'
        if not p.exists():
            print(f'WARN: missing {p}')
            continue
        d = json.loads(p.read_text())
        bals.append(d['mAP_balanced'])
        arcs.append(d['arc_mAP'])
    if not bals:
        continue
    rows.append((ver, round(sum(bals)/len(bals), 4), round(sum(arcs)/len(arcs), 4)))

with open(output_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['model', 'mAP_balanced', 'mAP_arc'])
    for ver, b, a in rows:
        w.writerow([ver, f'{b:.4f}', f'{a:.4f}'])

print(f'\nWrote {output_csv}')
print(f'{"model":30}  {"mAPbal":>8}  {"mAParc":>8}')
for ver, b, a in rows:
    print(f'{ver:30}  {b:>8.4f}  {a:>8.4f}')
PYEOF
