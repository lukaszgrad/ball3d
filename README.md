# ball3d

3D ball trajectory estimation and analysis for football/soccer video footage.

## Installation

```bash
uv sync
```

## Data

See [DATA.md](DATA.md) for datasets, download link, per-clip layout, conventions, and file-by-file schemas.

## Running Experiments

All scripts use [Hydra](https://hydra.cc/) for configuration. **Always run via `uv run`.**

See `conf/base.yaml` for global parameters and `conf/trajectory/` for estimator-specific configs.

### Trajectory Estimation

```bash
uv run python estimate_trajectory.py \
    trajectory=basic_angular_velocity \
    root=data/stalowa_wola_5/pano/clip \
    version=basic_angular
```

Writes `<root>/track/ball_3d.<version>.csv`. The `<root>` path depends on the dataset (see the [Data structure](#data-structure) examples).

### Visualization

Render a model's 3D trajectory overlaid on the source video. Requires `input.mkv` in `<root>` (LP and SW pano only).

```bash
uv run python visualize_trajectory.py \
    root=data/stalowa_wola_5/pano/clip \
    version=basic_angular
```

Add `visualisation.show_ground_truth=true` to also overlay the ground-truth trajectory:

```bash
uv run python visualize_trajectory.py \
    root=data/stalowa_wola_5/pano/clip \
    version=basic_angular \
    visualisation.show_ground_truth=true
```

### Evaluation

Compute per-clip metrics (mAP at multiple distance thresholds, 3D errors broken down by full / arc / straight, coverage, etc.) for a model's predictions:

```bash
uv run python evaluate_trajectory.py \
    root=data/stalowa_wola_5/pano/clip \
    trajectory=basic_angular_velocity version=basic_angular
```

Writes `<root>/eval/gt_metrics-<version>.json`, `<root>/eval/errors-<version>.csv`, and diagnostic plots. The JSON contains every metric reported in paper Tables 3 and 5.

### Available Trajectory Estimators

The seven physics models from the paper (Table 1):

| Config | Paper name | Description |
|--------|------------|-------------|
| `basic_kinetic_estimator_parabola` | basic parabola | Pure ballistic motion under gravity; no drag, no spin |
| `basic_kinetic_estimator` | basic kinetic | Gravity + quadratic drag (`k3`) |
| `basic_kinetic_fitg` | basic fitg | Drag (`k3`) + fitted gravitational constant `g` (proxy for unmodelled effects) |
| `basic_angular_velocity` | basic angular | Drag + Magnus force decomposed into lift (`kl`, topspin/backspin) and sidespin (`ks`) |
| `mujoco_kinetic_estimator` | MuJoCo kinetic | MuJoCo simulation with fixed inertia-based fluid drag (no fitted aero params) |
| `mujoco_angular_velocity` | MuJoCo angular | MuJoCo with fitted initial angular velocity `ω₀` |
| `mujoco_ellipsoid_angular_velocity` | MuJoCo ellipsoid | MuJoCo ellipsoid fluid model: blunt drag, angular drag, Magnus, with fitted `ω₀` |

For the arc-loss ablation (paper Table 6, supplement §8.1), six additional configs match `basic_parabola` and `basic_fitg` but zero out individual loss-objective terms (`L_end`, `L_z`, or both):

- `basic_kinetic_estimator_parabola_{no_end,no_z,traj_only}.yaml`
- `basic_kinetic_fitg_{no_end,no_z,traj_only}.yaml`

### Common Overrides

```bash
# Parallel jobs
n_jobs=8

# Frame stepping
step_frame=1

# Time range
start_sec=0 end_sec=-1

# Use ground truth pivot points
use_gt_pivots=true
```

## Reproducing paper tables

Three shell scripts run evaluation across a dataset and macro-average the per-view results into a CSV matching the paper layout. Each assumes `ball_3d.<version>.csv` predictions already exist under each ROOT.

| Script | Reproduces | Default ROOTS |
|---|---|---|
| `./eval_table3.sh` | Table 3 (mAPbal, mAParc per model) | LP-static (cameras 01–05 × 2 halves) |
| `./eval_table5.sh` | Table 5 (Full / Str / Arc mean 3D error, m, for the two best models) | LP-static (cameras 01–05 × 2 halves) |
| `./eval_table6.sh` | Table 6 (arc-loss ablation, mAParc and vertical error, 8 model × objective variants) | LP-static (cameras 01–05 × 2 halves) |

What each script does:

1. Calls `evaluate_trajectory.py` for every (root, model) pair to produce per-half `eval/gt_metrics-<version>.json`.
2. Macro-averages the relevant fields across roots.
3. Writes `logs/table<N>.csv`.

Edit the `ROOTS=(...)` array at the top of a script to switch dataset columns. `eval_table3.sh` and `eval_table5.sh` ship with a commented-out LP-broadcast block alongside the active LP-static one; `eval_table6.sh` is LP-static only (per paper §8.1).

Env vars:

```bash
OUTPUT_CSV=logs/table3-lp-static.csv  # override output path
SKIP_EXISTING=1                       # reuse existing eval/gt_metrics-<ver>.json
```
