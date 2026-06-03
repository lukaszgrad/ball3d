# ball3d

3D ball trajectory estimation and analysis for football/soccer video footage.

## Installation

```bash
uv sync
```

## Data

The full dataset is hosted on Google Drive:

**https://drive.google.com/drive/folders/1QCrmzZYZt9_tIBqY6AmP7DjDyPXe3AuI**

Download the contents into a local `data/` directory at the repo root. You can browse and download via the GDrive web UI, or use [rclone](https://rclone.org/drive/) for scripted bulk download.

### Datasets included

| Dataset | Folder(s) | View-sequences | Has `input.mkv`? |
|---|---|---|---|
| **LP** (Legia–Piast) | `legia_warszawa-piast_gliwice-20251214/` | 1 broadcast + 6 static, × 2 halves | yes |
| **SW** (Stalowa Wola) | `stalowa_wola_{5,6,39,42}/` | 1 panoramic per sequence | yes |
| **EB** (Ekstraklasa Broadcast) | 5 game folders (`piast_gliwice-rakow_…`, etc.) | 1 broadcast each | no |
| **ISSIA-3D** | `ISSIA/` | 6 cameras × `half_1` | no |

### Data structure

Each view-sequence (one monocular clip used as input to the pipeline) is stored under a `<root>/` directory with this layout:

```
<root>/
├── input.mkv                       # source video (LP and SW pano only)
├── camera_smooth.csv               # smoothed per-frame camera parameters
├── sequence_metadata.json          # fps, width, height
├── detection/
│   └── ball_detection.csv          # ball bounding boxes per frame
├── dev/
│   ├── df_merged_ball_player.csv   # ball + player detections
│   └── pauses.csv                  # ball-out-of-play intervals
└── track/
    ├── ball_3d-gt.csv              # ground-truth 3D trajectory
    ├── ball_pivot_point-gt.csv     # pivot-point annotations
    ├── split.csv                   # per-frame arc/straight labels
    └── ball_3d.<version>.csv       # estimated 3D trajectory (one per model profile)

<root>/../pitch_geom/                # one level above <root>
└── calibrate_camera_dict.pickle    # camera calibration
```

Concrete `<root>` examples for each dataset:

| Dataset | Example `<root>` |
|---|---|
| LP broadcast | `data/legia_warszawa-piast_gliwice-20251214/broadcast/half_1/` |
| LP static | `data/legia_warszawa-piast_gliwice-20251214/camera03/half_1/` |
| SW pano | `data/stalowa_wola_5/pano/clip/` |
| EB game | `data/piast_gliwice-rakow_czestochowa-20230811/clip/` |
| ISSIA-3D | `data/ISSIA/camera01/half_1/` |

Per-profile predictions (`ball_3d.<version>.csv`) are shipped for the seven paper models: `basic_parabola`, `basic_kinetic`, `basic_fitg`, `basic_angular`, `mujoco_kinetic`, `mujoco_angular`, `mujoco_ellipsoid`.

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

Render a model's 3D trajectory overlaid on the source video. Requires `input.mkv` in the clip directory.

```bash
uv run python visualize_trajectory.py \
    root=data/<clip>/clip \
    version=basic_angular
```

Add `visualisation.show_ground_truth=true` to also overlay the ground-truth trajectory:

```bash
uv run python visualize_trajectory.py \
    root=data/<clip>/clip \
    version=basic_angular \
    visualisation.show_ground_truth=true
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
