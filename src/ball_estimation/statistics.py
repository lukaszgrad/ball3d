import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict
from pydantic import BaseModel, NonNegativeFloat

from ball_estimation.evaluate import _impute_fitting_errors, _add_pauses_data

_logger = logging.getLogger(__name__)


def plot_bins(
    data: List[float],
    min_val: float,
    max_val: float,
    num_bins: int,
    title: str,
    plt_idx: tuple = None,
):
    """Plot a histogram of the data. Don't forget to do plt.show() after calling this function!
    Parameters
    ----------
    data : List[float]
        List of data points.
    min_val : float
        Minimum value of the histogram.
    max_val : float
        Maximum value of the histogram.
    num_bins : int
        Number of bins.
    title : str
        Title of the plot.
    plt_idx : tuple
        Indeces of the subplot. E.g. (2, 2, 1) for a 2x2 grid and the first plot.
    """
    if plt_idx is None:
        plt.hist(data, bins=np.linspace(min_val, max_val, num_bins))
    else:
        plt.subplot(*plt_idx)
        plt.hist(data, bins=np.linspace(min_val, max_val, num_bins))
        plt.title(title)


def build_segment_statistics(
    dev_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    root: str,
    version: str | None,
    pivots: np.ndarray | None = None,
) -> dict:
    """Build per-segment statistics JSON from dev and gt dataframes.

    Parameters
    ----------
    dev_df : pd.DataFrame
        Predicted/dev ball_3d dataframe.
    gt_df : pd.DataFrame
        Ground truth ball_3d dataframe.
    root : str
        Dataset clip root (used for pivot CSV fallback resolution).
    version : str | None
        Version suffix.
    pivots : np.ndarray | None
        Optional list of pivot frame indices. If None, will infer from pivot CSV or in-CSV tokens.

    Returns
    -------
    dict
        Mapping of segment keys to statistics payloads.
    """
    from typing import Any

    def _detect_pivot_column(df: pd.DataFrame) -> str | None:
        candidate_cols: List[str] = [c for c in df.columns if df[c].dtype == object]
        lowered_tokens = {"pivot_point", "high_pivot_point", "additional_pivot_point"}
        for col in candidate_cols:
            s = df[col].dropna().astype(str).str.lower()
            if (
                s.isin(lowered_tokens).any()
                or s.str.contains("pivot_point", na=False).any()
            ):
                return col
        return None

    def _load_pivot_frames(root: str) -> np.ndarray | None:
        for name in ["ball_pivot_point-gt.csv", "ball_pivot_point.csv"]:
            pivot_path = os.path.join(root, "track", name)
            if os.path.exists(pivot_path):
                piv = pd.read_csv(pivot_path)
                if {"file_name", "pivot_point"}.issubset(piv.columns):
                    arr = (
                        piv.loc[piv["pivot_point"] == 1, "file_name"]
                        .astype(int)
                        .to_numpy()
                    )
                    return np.sort(np.unique(arr))
        return None

    def _collect_params(
        segment_df: pd.DataFrame, segment_type: str
    ) -> Dict[str, float]:
        params: Dict[str, float] = {}
        if segment_type == "straight":
            keys = ["k", "x_start", "y_start", "x_end", "y_end"]
        else:
            keys = [
                "g",
                "k3",
                "kl",
                "ks",
                "vx0",
                "vy0",
                "vz0",
                "x0",
                "y0",
                "z0",
                "blunt_drag",
                "slender_drag",
                "angular_drag",
                "kutta_lift",
                "magnus_lift",
            ]
        for key in keys:
            if key not in segment_df.columns:
                continue
            series = segment_df[key]
            try:
                if key.endswith("_start"):
                    val = series.dropna().astype(float).iloc[0]
                elif key.endswith("_end"):
                    val = series.dropna().astype(float).iloc[-1]
                else:
                    val = series.dropna().astype(float).iloc[0]
                params[key] = float(val)
            except Exception:
                vals = series.to_numpy(dtype=float)
                vals = vals[~np.isnan(vals)]
                if vals.size > 0:
                    params[key] = float(np.mean(vals))
        return params

    def _compute_segment_error(
        segment_df: pd.DataFrame, gt_df: pd.DataFrame
    ) -> float | None:
        cols_dev = ["file_name", "x_predicted", "y_predicted", "z_predicted"]
        cols_gt = ["file_name", "x", "y", "z"]
        if not set(cols_dev).issubset(segment_df.columns):
            return None
        if not set(cols_gt).issubset(gt_df.columns):
            return None
        left = segment_df[cols_dev]
        right = gt_df[cols_gt]
        merged = left.merge(right, on="file_name", how="inner")
        if merged.empty:
            return None
        coords = merged[
            ["x_predicted", "y_predicted", "z_predicted", "x", "y", "z"]
        ].to_numpy(dtype=float)
        mask = ~np.isnan(coords).any(axis=1)
        coords = coords[mask]
        if coords.size == 0:
            return None
        pred = coords[:, 0:3]
        gt = coords[:, 3:6]
        err = np.linalg.norm(pred - gt, axis=1)
        return float(err.mean())

    # resolve pivots
    if pivots is None:
        pivots = _load_pivot_frames(root)
        if pivots is None:
            pivot_col = _detect_pivot_column(dev_df)
            if pivot_col is None:
                raise RuntimeError(
                    "Could not detect pivot column containing 'pivot_point' tokens."
                )
            tokens = {"pivot_point", "high_pivot_point", "additional_pivot_point"}
            mask = dev_df[pivot_col].astype(str).str.lower().isin(tokens)
            pivots = dev_df.loc[mask, "file_name"].astype(int).to_numpy()
            pivots = np.sort(np.unique(pivots))

    if pivots.size < 2:
        _logger.warning("Less than 2 pivot frames detected; nothing to segment.")
        return {}

    result: Dict[str, Any] = {}
    for start, next_pivot in zip(pivots[:-1], pivots[1:]):
        end_exclusive = int(next_pivot) + 1
        seg_df = dev_df[
            (dev_df["file_name"] >= int(start)) & (dev_df["file_name"] < end_exclusive)
        ]
        if seg_df.empty:
            continue
        if "type" in seg_df.columns:
            straight_count = (
                seg_df["type"].astype(str).str.lower() == "straight"
            ).sum()
            arc_count = (seg_df["type"].astype(str).str.lower() == "arc").sum()
            segment_type = "straight" if straight_count >= arc_count else "arc"
        else:
            segment_type = "straight"

        fitted_params = _collect_params(seg_df, segment_type)
        if segment_type == "arc":
            pivot_row = dev_df.loc[dev_df["file_name"] == int(start)]
            if not pivot_row.empty:
                pr = pivot_row.iloc[0]
                for coord in ("x0", "y0", "z0"):
                    if coord in dev_df.columns:
                        try:
                            fitted_params[coord] = float(pr.get(coord))
                        except Exception:
                            pass
        err = _compute_segment_error(seg_df, gt_df)
        if err is None:
            continue
        key = f"trajectory_PRED_segment_{int(start)}"
        result[key] = {
            "trajectory_id": "PRED",
            "segment_type": segment_type,
            "start_frame": int(start),
            "end_frame": int(end_exclusive),
            "fitting_error": float(err),
            "fitted_params": fitted_params,
        }

    return result


def statistics_ball_trajectory(
    ball_3d_dev: pd.DataFrame,
    dev__pauses: pd.DataFrame | None = None,
    split_df: pd.DataFrame | None = None,
    path: str | None = None,
    version: str = "",
) -> dict:
    dev__ball_3d = ball_3d_dev

    _impute_fitting_errors(dev__ball_3d)
    _add_pauses_data(dev__ball_3d, dev__pauses, split_df)
    pauses_rate = dev__ball_3d["pause"].mean()
    dev__ball_3d = dev__ball_3d.loc[~dev__ball_3d["pause"]]

    if "is_ball_detected" not in dev__ball_3d:
        dev__ball_3d["is_ball_detected"] = dev__ball_3d["x0"].notna()
    if "is_camera_detected" not in dev__ball_3d:
        dev__ball_3d["is_camera_detected"] = dev__ball_3d["h0"].notna()

    ball_det_coverage = dev__ball_3d["is_ball_detected"].mean()
    ball_smooth_coverage = dev__ball_3d["xk"].notna().mean()
    cam_coverage = dev__ball_3d["is_camera_detected"].mean()
    cam_ball_coverage = (
        dev__ball_3d["is_camera_detected"] * dev__ball_3d["is_ball_detected"]
    ).mean()
    cam_ball_smooth_coverage = (
        dev__ball_3d["is_camera_detected"] * (dev__ball_3d["xk"].notna())
    ).mean()
    trajectory_coverage = dev__ball_3d["x_predicted"].notna().mean()
    _logger.info(
        f"\n\nPauses_rate = {pauses_rate:.4f} of all frames;\n"
        + f"Coverage:  ball_detected={ball_det_coverage:5.4f};"
        f"  ball_smoothed={ball_smooth_coverage:5.4f}; "
        + f"camera={cam_coverage:5.4f}; camera+ball={cam_ball_coverage:5.4f}; camera+ball_smoothed={cam_ball_smooth_coverage:5.4f};\n"  # noqa: B950
        + f"trajectory_coverage = {trajectory_coverage:.4f}"
        f" ({trajectory_coverage/cam_ball_coverage:.4f} of max possible)\n"
    )

    df = dev__ball_3d

    pivot_indices = df["file_name"][df["pivot_probability"] > 0.15].to_numpy()

    fit_keys_general = [
        "vx",
        "vy",
        "vz",
        "x_start",
        "y_start",
        "x_end",
        "y_end",
        "wx0",
        "wy0",
        "wz0",
        "kl",
        "ks",
    ]  # They are not directly used, but only used to compute another variable
    fit_keys_arc = [
        "g",
        "k3",
        "blunt_drag",
        "slender_drag",
        "angular_drag",
        "kutta_lift",
        "magnus_lift",
    ]
    fit_keys_straight = ["k"]
    inferrred_keys_arc = [
        "v",
        "klabs",
        "ksabs",
        "w",
    ]  # These variables are inferred from the previous ones
    inferrred_keys_straight = [
        "v",
        "distance",
    ]  # These variables are inferred from the previous ones
    fit_keys = fit_keys_general + fit_keys_arc + fit_keys_straight
    fit_dicts = (
        {}
    )  # {key: [mean value for first trajectory, mean value for second trajectory, ...]}
    num_arc, num_straight = 0, 0
    # gs_arc = []
    xy_errors_arc, xy_errors_straight = [], []
    # ks_arc, ks_straight = [], []
    prev_index = int(pivot_indices[0])
    for index in pivot_indices[1:]:
        index = int(index)
        # is it straight or arc trajectory?
        type = df.loc[prev_index : index - 1]["type"]
        frames_straight = type[type == "straight"].size
        frames_arc = type[type == "arc"].size

        z, z_std = (
            df.loc[prev_index : index - 1]["z_predicted"],
            df.loc[prev_index : index - 1]["z_predicted"],
        )
        fit_dict = {}
        for key in fit_keys:
            if key in df.keys():
                fit_dict[key] = df.loc[prev_index : index - 1][key]
        # g = df.loc[prev_index : index - 1]["g"]
        # k = df.loc[prev_index : index - 1]["k"]
        xy_error = np.sqrt(
            (
                df.loc[prev_index : index - 1]["x_pitch2D"]
                - df.loc[prev_index : index - 1]["x_predicted"]
            )
            ** 2
            + (
                df.loc[prev_index : index - 1]["y_pitch2D"]
                - df.loc[prev_index : index - 1]["y_predicted"]
            )
            ** 2
        )

        fit_dict["v"] = np.sqrt(
            fit_dict["vx"] ** 2 + fit_dict["vy"] ** 2 + fit_dict["vz"] ** 2
        )
        if "wx0" in fit_dict and "wy0" in fit_dict and "wz0" in fit_dict:
            fit_dict["w"] = np.sqrt(
                fit_dict["wx0"] ** 2 + fit_dict["wy0"] ** 2 + fit_dict["wz0"] ** 2
            )
        # since kl and ks can be positive and negative, we take the absolute value
        if "kl" in fit_dict:
            fit_dict["klabs"] = np.abs(fit_dict["kl"])
        if "ks" in fit_dict:
            fit_dict["ksabs"] = np.abs(fit_dict["ks"])
        fit_dict["distance"] = np.sqrt(
            (fit_dict["x_end"] - fit_dict["x_start"]) ** 2
            + (fit_dict["y_end"] - fit_dict["y_start"]) ** 2
        )
        # remove nans
        for key, value in fit_dict.items():
            fit_dict[key] = value[~np.isnan(fit_dict[key])]
        xy_error = xy_error[~np.isnan(xy_error)]

        if frames_arc > frames_straight:  # it is an arc trajectory
            num_arc += 1
            for key in fit_keys_arc:
                if key in fit_dict and fit_dict[key].size > 0:
                    if f"{key}_arc" not in fit_dicts:
                        fit_dicts[f"{key}_arc"] = []
                        fit_dicts[f"frames_{key}_arc"] = []
                    fit_dicts[f"{key}_arc"].append(fit_dict[key].mean())
                    fit_dicts[f"frames_{key}_arc"].append((prev_index, index - 1))
            for key in inferrred_keys_arc:
                if f"{key}_arc" not in fit_dicts and key in fit_dict:
                    fit_dicts[f"{key}_arc"] = []
                    fit_dicts[f"frames_{key}_arc"] = []
                if key in fit_dict and fit_dict[key].size > 0:
                    fit_dicts[f"{key}_arc"].append(fit_dict[key].mean())
                    fit_dicts[f"frames_{key}_arc"].append((prev_index, index - 1))

            if xy_error.size > 0:
                xy_errors_arc.append(xy_error.mean())
        else:  # it is a straight trajectory
            num_straight += 1
            for key in fit_keys_straight:
                if key in fit_dict and fit_dict[key].size > 0:
                    if f"{key}_straight" not in fit_dicts:
                        fit_dicts[f"{key}_straight"] = []
                        fit_dicts[f"frames_{key}_straight"] = []
                    fit_dicts[f"{key}_straight"].append(fit_dict[key].mean())
                    fit_dicts[f"frames_{key}_straight"].append((prev_index, index - 1))
            for key in inferrred_keys_straight:
                if f"{key}_straight" not in fit_dicts and key in fit_dict:
                    fit_dicts[f"{key}_straight"] = []
                    fit_dicts[f"frames_{key}_straight"] = []
                if key in fit_dict and fit_dict[key].size > 0:
                    fit_dicts[f"{key}_straight"].append(fit_dict[key].mean())
                    fit_dicts[f"frames_{key}_straight"].append((prev_index, index - 1))
            if xy_error.size > 0:
                xy_errors_straight.append(xy_error.mean())

        prev_index = index

    # # do some plots
    # if path is not None:
    #     for ind, (key, value) in enumerate(fit_dicts.items()):
    #         plot_bins(value, min(value), max(value), 10, key, (len(fit_dicts.keys())+2, 1, ind+1))
    #     plot_bins(
    #         xy_errors_arc,
    #         0.0,
    #         max(xy_errors_arc),
    #         20,
    #         "xy error arc",
    #         (len(fit_dicts.keys())+2, 1, (len(fit_dicts.keys())+1)),
    #     )
    #     plot_bins(
    #         xy_errors_straight,
    #         0.0,
    #         max(xy_errors_straight),
    #         20,
    #         "xy error straight",
    #         (len(fit_dicts.keys())+2, 1, (len(fit_dicts.keys())+2)),
    #     )
    #     plt.tight_layout()
    #     savepath = os.path.join(path, "eval")
    #     os.makedirs(savepath, exist_ok=True)
    #     savepath = (
    #         os.path.join(savepath, f"statistics{'-'+version if version is not None else ''}.png")
    #         if version != ""
    #         else os.path.join(savepath, "statistics.png")
    #     )
    #     plt.savefig(savepath)

    for key, value in fit_dicts.items():
        print(f"Mean {key}:", np.mean(value), ", Std {key}:", np.std(value))

    print(
        "Mean arc xy error:",
        np.mean(xy_errors_arc),
        ", Std arc xy error:",
        np.std(xy_errors_arc),
    )
    print(
        "Mean straight xy error:",
        np.mean(xy_errors_straight),
        ", Std straight xy error:",
        np.std(xy_errors_straight),
    )

    # print("# arcs:", len(gs_arc), ", # straights:", len(ks_straight))
    print(
        "# arcs:",
        len(fit_dicts["k3_arc"]),
        ", # straights:",
        len(fit_dicts["k_straight"]),
    )

    save_dict = {}
    for key, value in fit_dicts.items():
        if "frames" not in key:
            save_dict[f"mean_{key}"] = np.mean(value)
            save_dict[f"std_{key}"] = np.std(value)
        save_dict[key] = value
    save_dict["mean_arc_xy_error"] = np.mean(xy_errors_arc)
    save_dict["std_arc_xy_error"] = np.std(xy_errors_arc)
    save_dict["arc_xy_error"] = xy_errors_arc
    save_dict["mean_straight_xy_error"] = np.mean(xy_errors_straight)
    save_dict["std_straight_xy_error"] = np.std(xy_errors_straight)
    save_dict["straight_xy_error"] = xy_errors_straight
    save_dict["num_arcs"] = len(xy_errors_arc)
    save_dict["num_straights"] = len(xy_errors_straight)

    return save_dict
