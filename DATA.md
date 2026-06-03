# Data Reference

Companion reference for the datasets shipped with the ball3d release. The dataset itself is hosted on Google Drive:

**https://drive.google.com/drive/folders/1QCrmzZYZt9_tIBqY6AmP7DjDyPXe3AuI**

Download the contents into a `data/` directory at the repo root.

## Datasets

The release groups the data into four datasets corresponding to the evaluations reported in the paper. The two primary datasets with full video are **LP** and **SW**.

| Dataset | Folder(s) | View-sequences | Source video | Has player detections |
|---|---|---|---|---|
| **LP** — Legia × Piast (94-min match) | `legia_warszawa-piast_gliwice-20251214/` | 1 broadcast + 6 static, × 2 halves = 14 | yes (LP broadcast + LP static) | yes |
| **SW** — Stalowa Wola drills | `stalowa_wola_{5,6,39,42}/` | 1 panoramic per sequence = 4 | yes (pano only) | no |
| **EB** — Ekstraklasa broadcast | 5 game folders | 1 broadcast each = 5 | no (licensing) | no |
| **ISSIA-3D** | `ISSIA/` | 6 cameras × `half_1` = 6 | no | yes |

EB folders: `piast_gliwice-rakow_czestochowa-20230811`, `radomiak_radom-piast_gliwice-20230902`, `slask_wroclaw-zaglebie_lubin-20230729`, `zaglebie_lubin-lech_poznan-20230806`, `legia_warszawa-widzew_lodz-20230903`.

## Per-clip layout

Each view-sequence is one monocular clip. It lives in a `<root>/` directory with the layout below. Concrete `<root>` per dataset:

| Dataset | Example `<root>` |
|---|---|
| LP broadcast | `data/legia_warszawa-piast_gliwice-20251214/broadcast/half_1/` |
| LP static | `data/legia_warszawa-piast_gliwice-20251214/camera03/half_1/` |
| SW pano | `data/stalowa_wola_5/pano/clip/` |
| EB game | `data/piast_gliwice-rakow_czestochowa-20230811/clip/` |
| ISSIA-3D camera | `data/ISSIA/camera01/half_1/` |

```
<root>/
├── input.mkv                          # source video (LP and SW pano only)
├── sequence_metadata.json             # video metadata: fps, width, height
├── camera_smooth.csv                  # smoothed per-frame camera parameters
├── hom_smooth.csv                     # per-frame image→pitch homography
├── detection/
│   ├── ball_detection.csv             # ball bounding boxes per frame
│   └── detection.feather              # player bounding boxes (LP + ISSIA only)
├── dev/
│   ├── df_merged_ball_player.csv      # preprocessing output (smoothed ball + contacts)
│   └── pauses.csv                     # ball-out-of-play intervals
└── track/
    ├── ball_3d-gt.csv                 # ground-truth 3D trajectory
    ├── ball_pivot_point-gt.csv        # ground-truth pivot-point annotations
    ├── split.csv                      # per-frame arc/straight labels
    └── ball_3d.<version>.csv          # estimated 3D trajectory, one per model profile

<root>/../pitch_geom/                  # one level above <root>, shared across halves
└── calibrate_camera_dict.pickle       # camera calibration object
```

Per-profile predictions ship for the seven paper models: `basic_parabola`, `basic_kinetic`, `basic_fitg`, `basic_angular`, `mujoco_kinetic`, `mujoco_angular`, `mujoco_ellipsoid`.

## Conventions

- **Length units:** decimeters (dm). 3D ball positions (`x`, `y`, `z`) and pitch-plane coordinates (`x_pitch2D`, `y_pitch2D`) are in dm. Velocities are dm/s, accelerations dm/s². Camera translation (`tx`, `ty`, `tz`) is in dm.
- **Image coordinates** (`x0`, `y0`, `x1`, `y1`, `xk`, `yk`, `princ_x`, `princ_y`): pixels, origin at image top-left.
- **Frame indexing:** the column is named `file_name` in most ball-side CSVs (and `frame_index` in camera/hom CSVs). Both are 0-based integer frame indices.
- **Camera convention:** pin-hole model. Rotation is stored as a Rodrigues vector (`rot_x`, `rot_y`, `rot_z`). Reconstruct the 3×3 intrinsics from `fx, fy, princ_x, princ_y` and the projection matrix as `K [R | t]`.
- **Homography:** `h0..h8` are the flattened **inverse** homography (pitch ← image), i.e. apply it to an image-space point to get the pitch-plane point. Reshape as `H.reshape(3, 3)` and use as `p_pitch = H @ [x_img, y_img, 1]` (then divide by the homogeneous component).

## File schemas

### `sequence_metadata.json`

Video metadata produced by ffprobe at clip-cut time. The fields used by the pipeline are `fps`, `width`, `height`. Other ffprobe fields (codec, profile, color space, etc.) are passed through unmodified and ignored by the pipeline.

### `camera_smooth.csv` — smoothed camera parameters per frame

Column list matches `Camera.columns_pandas()` (see `src/ball_estimation/camera/__init__.py`), plus two bookkeeping columns:

| Column | Type | Description |
|---|---|---|
| `rot_x`, `rot_y`, `rot_z` | float | Rodrigues rotation vector |
| `tx`, `ty`, `tz` | float | Camera translation (dm) |
| `fx`, `fy` | float | Focal lengths (pixels) |
| `princ_x`, `princ_y` | float | Principal point (pixels) |
| `error` | float | Reprojection error for the fitted calibration |
| `frame_index` | int | Frame index (0-based) |

### `hom_smooth.csv` — per-frame homography

| Column | Type | Description |
|---|---|---|
| `h0`..`h8` | float | Flattened 3×3 inverse homography (image → pitch) |
| `error` | float | Reprojection error |
| `frame_index` | int | Frame index |

### `detection/ball_detection.csv` — ball detector output

| Column | Type | Description |
|---|---|---|
| `file_name` | int | Frame index |
| `x0`, `y0`, `x1`, `y1` | float | Bounding box (pixels) |
| `detection_id` | int | ID within the frame |
| `category`, `category_id` | str / int | Class label and ID |
| `score` | float | Detector confidence |
| `segmentation` | str | RLE-encoded segmentation mask |

### `detection/detection.feather` — player detector output (LP, ISSIA only)

Same column conventions as `ball_detection.csv` but covering all player and goalkeeper bounding boxes (`category` ∈ {`player`, `goalkeeper`, …}). Stored as Apache Arrow Feather for fast load.

### `dev/df_merged_ball_player.csv` — preprocessing output

Output of the preprocessing pipeline (ball tracking → smoothing → contact detection). Combines smoothed ball trajectory, the per-frame homography, and ball–player contact flags.

| Group | Columns |
|---|---|
| Identity / smoothing | `file_name`, `track_id`, `xk`, `yk`, `xvar`, `yvar`, `score`, `x0`, `y0`, `x1`, `y1`, `a`, `b` |
| Homography | `h0`..`h8` |
| Pitch projection | `x_pitch2D`, `y_pitch2D`, `x_pano_real`, `y_pano_real` |
| Contact flags | `p_detection_id`, `common_points`, `distance`, `ball_height_rel`, `out`, `far_contact`, `close_contact`, `high_contact` |
| Height correction | `yk_corr`, `yk_original` |

### `dev/pauses.csv` — ball-out-of-play intervals

| Column | Type | Description |
|---|---|---|
| `start_pause`, `end_pause` | int | Frame range (inclusive start, exclusive end) where the ball is out of play |

### `track/ball_3d-gt.csv` — ground-truth 3D trajectory

| Column | Type | Description |
|---|---|---|
| `file_name` | int | Frame index |
| `track_id` | int | Track-segment ID |
| `x`, `y`, `z` | float | 3D position (dm) |
| `x_velocity`, `y_velocity`, `z_velocity` | float | Velocity (dm/s) |
| `x_acceleration`, `y_acceleration`, `z_acceleration` | float | Acceleration (dm/s²) |

### `track/ball_pivot_point-gt.csv` — pivot-point annotations

| Column | Type | Description |
|---|---|---|
| `file_name` | int | Frame index |
| `track_id` | int | Track-segment ID |
| `pivot_probability` | float | Confidence in `[0, 1]` |
| `pivot_point` | str | Token (`pivot_point`, `high_pivot_point`, `additional_pivot_point`, or empty) |

### `track/split.csv` — per-frame arc/straight labels

| Column | Type | Description |
|---|---|---|
| `frame_index` | int | Frame index |
| `error_arc`, `error_straight` | float | Fit error for arc / straight model on the segment containing this frame |
| `is_arc` | bool | True if the segment containing this frame is classified as an arc |
| `is_arc_long` | bool | True if the segment is a long arc (used by some evaluations) |

### `track/ball_3d.<version>.csv` — estimated 3D trajectory (one per profile)

Wide denormalized output written by `estimate_trajectory.py`. The downstream tooling reads it by named column, so the full column set is not load-bearing; key column groups are:

| Group | Columns |
|---|---|
| Identity | `file_name`, `is_ball_detected`, `is_camera_detected`, `track_id` |
| Smoothed 2D ball + homography | `xk`, `yk`, `xvar`, `yvar`, `score`, `h0`..`h8` |
| Pitch projection | `x_pitch2D`, `y_pitch2D`, `x_pano_real`, `y_pano_real` |
| Predicted 3D | `x_predicted`, `y_predicted`, `z_predicted`, `z_predicted_raw` |
| Derived kinematics | `vx_predicted`, `vy_predicted`, `vz_predicted`, `ax_predicted`, `ay_predicted`, `az_predicted`, `v_abs_predicted`, `a_abs_predicted` |
| Segment / physics fit | `type`, `g`, `k`, `k3`, `kl`, `ks`, `vx`, `vy`, `vz` (varies by model) |
| Contact flags | `far_contact`, `close_contact`, `high_contact`, `out` |
| Repair flags | `is_repaired`, `is_imputed` |

### `<root>/../pitch_geom/calibrate_camera_dict.pickle`

Single Python pickle containing a `dict` with the static (i.e. constant across the half) camera-calibration prior used during pitch-geometry fitting. Plain dict, no custom classes — `pickle.load` works on it from a fresh Python process.
