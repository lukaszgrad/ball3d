# ball3d

3D ball trajectory estimation and analysis for football/soccer video footage.

## Installation

```bash
uv sync
```

## Data

### GCloud

Install `gcloud` CLI: https://cloud.google.com/sdk/docs/install

Then run in the root directory:

```bash
gsutil -m cp -r gs://3dball-research/data .
```

### Data structure

```
data/<clip_name>/
├── clip/
│   ├── detection/
│   │   └── ball_detection.csv        # Ball bounding boxes per frame
│   ├── dev/
│   │   └── df_merged_ball_player.csv # Ball + player detections
│   ├── track/
│   │   ├── ball_pivot_point.csv      # Pivot point annotations
│   │   ├── ball_3d.csv               # Estimated 3D trajectory
│   │   └── ball_3d-gt.csv           # Ground truth trajectory
│   ├── camera_smooth.csv             # Camera parameters per frame
│   ├── hom_smooth.csv                # Homography matrices
│   ├── frame.csv                     # Frame metadata
│   └── sequence_metadata.json        # Video properties (fps, width, height)
```

## Running Experiments

All scripts use [Hydra](https://hydra.cc/) for configuration. **Always run via `uv run`.**

See `conf/base.yaml` for global parameters and `conf/trajectory/` for estimator-specific configs.

### Trajectory Estimation

```bash
# Default estimator
uv run python estimate_trajectory.py \
    root=data/1080-txm/clip version=base

# Custom estimator
uv run python estimate_trajectory.py \
    trajectory=basic_angular_velocity \
    root=data/1080-txm/clip version=base
```

### Visualization

```bash
uv run python visualize_trajectory.py \
    root=data/1080-txm/clip version=base
```

### Evaluation

Compute per-clip metrics (mAP at multiple distance thresholds, 3D errors broken down by full / arc / straight, coverage, etc.) for a model's predictions:

```bash
uv run python evaluate_trajectory.py \
    root=data/<clip>/clip \
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
| `./eval_table3.sh` | Table 3 (mAPbal, mAParc per model) | LP-broadcast (camera00 × 2 halves) |
| `./eval_table5.sh` | Table 5 (Full / Str / Arc mean 3D error, m, for the two best models) | LP-broadcast |
| `./eval_table6.sh` | Table 6 (arc-loss ablation, mAParc and vertical error, 8 model × objective variants) | LP-static (camera01–05 × 2 halves) |

What each script does:

1. Calls `evaluate_trajectory.py` for every (root, model) pair to produce per-half `eval/gt_metrics-<version>.json`.
2. Macro-averages the relevant fields across roots.
3. Writes `logs/table<N>.csv`.

Edit the `ROOTS=(...)` array at the top of a script to switch dataset columns (e.g. LP-static vs LP-broadcast — both blocks are present in each script, one commented out).

Env vars:

```bash
OUTPUT_CSV=logs/table3-lp-static.csv  # override output path
SKIP_EXISTING=1                       # reuse existing eval/gt_metrics-<ver>.json
```
