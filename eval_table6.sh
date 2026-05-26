#!/usr/bin/env bash
# eval_table6.sh — reproduce paper Table 6 (arc-loss ablation, supplement §8.1).
# 8 rows: basic_parabola × {full, no_end, no_z, traj_only}
#       + basic_fitg     × {full, no_end, no_z, traj_only}.
# 2 columns: mAParc and full-trajectory vertical error z (m).
#
# Per paper §8.1, Table 6 is on LP-static only.
#
# Assumes ball_3d.<version>.csv already exists under each ROOT's track/ folder.
# Runs evaluate_trajectory.py per (root, model_variant) and writes a macro-averaged CSV.
#
# Env vars:
#   OUTPUT_CSV     — output path (default: logs/table6.csv)
#   SKIP_EXISTING  — set to 1 to skip eval if gt_metrics-<ver>.json already exists
set -euo pipefail

# 8 ablation rows in Table 6 order: trajectory_config:version
MODELS=(
    "basic_kinetic_estimator_parabola:basic_parabola"
    "basic_kinetic_estimator_parabola_no_end:basic_parabola_no_end"
    "basic_kinetic_estimator_parabola_no_z:basic_parabola_no_z"
    "basic_kinetic_estimator_parabola_traj_only:basic_parabola_traj_only"
    "basic_kinetic_fitg:basic_fitg"
    "basic_kinetic_fitg_no_end:basic_fitg_no_end"
    "basic_kinetic_fitg_no_z:basic_fitg_no_z"
    "basic_kinetic_fitg_traj_only:basic_fitg_traj_only"
)

# LP-static (paper Table 6 dataset): 5 cameras × 2 halves.
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

OUTPUT_CSV="${OUTPUT_CSV:-logs/table6.csv}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# --- 1. Per-(root, version) evaluation ---
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

# --- 2. Macro-average across roots, write Table 6 CSV ---
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
    arcs, zerrs = [], []
    for root in roots:
        p = Path(root) / 'eval' / f'gt_metrics-{ver}.json'
        if not p.exists():
            print(f'WARN: missing {p}')
            continue
        d = json.loads(p.read_text())
        arcs.append(d['arc_mAP'])
        zerrs.append(d['full_z_err'])
    if not arcs:
        continue
    rows.append((ver, round(sum(arcs)/len(arcs), 4), round(sum(zerrs)/len(zerrs), 4)))

with open(output_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['model', 'mAP_arc', 'z_m'])
    for ver, a, z in rows:
        w.writerow([ver, f'{a:.3f}', f'{z:.3f}'])

print(f'\nWrote {output_csv}')
print(f'{"model":35}  {"mAParc":>7}  {"z (m)":>7}')
for ver, a, z in rows:
    print(f'{ver:35}  {a:>7.3f}  {z:>7.3f}')
PYEOF
