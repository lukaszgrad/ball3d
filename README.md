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

### Preprocessing

Preprocess raw ball detections into smoothed trajectories and pivot points. This replaces the external respo Docker pipeline.

```bash
# Basic usage (reads unversioned files, writes unversioned output)
uv run python preprocess_trajectory.py \
    root=data/<clip>/clip

# Separate input/output versions (read "dl", write "dl_test")
uv run python preprocess_trajectory.py \
    root=data/<clip>/clip version=dl output_version=dl_test
```

The pipeline runs 5 steps: ball tracking, track merging, Kalman smoothing, ball-player contact detection, and pivot point prediction. Config: `conf/preprocessing/base.yaml`.

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

```bash
uv run python evaluate_trajectory.py \
    root=data/1080-txm/clip version=base
```

### Statistics

```bash
# Estimated trajectory statistics
uv run python statistics_trajectory.py \
    root=data/1080-txm/clip version=base

# Ground truth statistics
uv run python statistics_gt_trajectory.py \
    root=data/1080-txm/clip
```

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

### Batch Evaluation

See [EVALUATE.md](EVALUATE.md) for per-dataset evaluation commands, HTML report generation, and `run_all.sh` / `generate_eval_report.py` option reference.

## Testing

```bash
# Run all tests
uv run pytest tests/

# Estimator tests
uv run pytest tests/estimators/test_estimator.py -v

# Physics simulation tests
uv run pytest tests/physics/test_mujoco.py -v
```
