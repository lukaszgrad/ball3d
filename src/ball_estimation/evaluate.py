import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, NonNegativeFloat

_logger = logging.getLogger(__name__)

_WRONG_BALL_ANNOTATION_TYPES = ("wrong_ball_trajectory", "wrong_ball_point")
_PIVOT_POINT_TYPES = ["pivot_point", "high_pivot_point", "wrong_pivot_point"]
_ACC_THRESHOLDS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
_GT_Z_ARC_THRESHOLD_DM = 3.0  # 0.3 m — frames with GT z above this are arcs


def _resolve_is_arc(
    df: pd.DataFrame,
    split_df: pd.DataFrame | None,
) -> pd.Series:
    """Derive per-frame is_arc from split_df or GT z-values.

    Priority:
    1. split_df["is_arc"] mapped by frame_index → file_name
    2. GT z column (df["z"] > 3 dm)
    3. NaN (unclassified)
    """
    is_arc = pd.Series(pd.NA, index=df.index, dtype="boolean")

    if split_df is not None and "is_arc" in split_df.columns:
        sdf = split_df.copy()
        if "frame_index" in sdf.columns and "file_name" not in sdf.columns:
            sdf = sdf.rename(columns={"frame_index": "file_name"})
        if "file_name" in sdf.columns:
            sdf["file_name"] = pd.to_numeric(sdf["file_name"], errors="coerce").astype("Int64")
            lookup = sdf.drop_duplicates("file_name").set_index("file_name")["is_arc"]
            mapped = df["file_name"].map(lookup)
            is_arc = mapped.astype("boolean")

    # Fallback: use GT z for frames still unclassified
    needs_fallback = is_arc.isna()
    if needs_fallback.any() and "z" in df.columns:
        gt_z = df["z"]
        has_z = gt_z.notna() & needs_fallback
        if has_z.any():
            is_arc = is_arc.copy()
            is_arc.loc[has_z] = (gt_z.loc[has_z] > _GT_Z_ARC_THRESHOLD_DM).astype("boolean")

    return is_arc


class Ball3dMetrics(BaseModel):
    ball_detected_coverage: NonNegativeFloat
    ball_smoothed_coverage: NonNegativeFloat
    camera_coverage: NonNegativeFloat
    camera_ball_coverage: NonNegativeFloat


class Ball3dGTMetrics(BaseModel):
    class Config:
        extra = 'allow'
    pauses_rate: float
    full_err: float
    full_xy_err: float
    full_z_err: float
    str_err: float
    str_xy_err: float
    str_z_err: float
    arc_err: float
    arc_xy_err: float
    arc_z_err: float


def _compute_ap_at_threshold(
    errors: np.ndarray,
    fitting_errors: np.ndarray,
    total_gt_frames: int,
    distance_threshold: float,
) -> float:
    """Compute Average Precision at a single distance threshold.

    Adapts the AP metric from object detection: each predicted frame is a
    "detection", fitting error is the confidence score, and a prediction is
    a true positive if its 3D error < distance_threshold. Unpredicted GT
    frames are automatic false negatives.

    Parameters
    ----------
    errors : array of 3D errors (meters) for predicted frames with GT
    fitting_errors : fitting error (confidence proxy) for the same frames
    total_gt_frames : |T_s| — total GT playtime frames (denominator for recall)
    distance_threshold : X meters
    """
    if total_gt_frames == 0 or len(errors) == 0:
        return 0.0

    order = np.argsort(fitting_errors)
    errors_sorted = errors[order]

    tp = (errors_sorted < distance_threshold).astype(np.float64)
    tp_cumsum = np.cumsum(tp)

    precision = tp_cumsum / np.arange(1, len(tp) + 1)
    recall = tp_cumsum / total_gt_frames

    # Prepend sentinel (recall=0, precision=1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])

    # Monotone decreasing envelope (VOC2010+ interpolation)
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    ap = float(np.sum(np.diff(recall) * precision[1:]))
    return ap


def _compute_new_metrics(
    df: pd.DataFrame,
    has_gt: pd.Series,
    split_mask: pd.Series | None = None,
) -> dict[str, float]:
    """Compute acc@Xm and AP@Xm metrics for a given split.

    Parameters
    ----------
    df : merged DataFrame (dev + GT) after pause removal
    has_gt : boolean mask for frames with GT annotations
    split_mask : optional boolean mask to restrict to str/arc frames.
        If None, uses all frames (overall).
    """
    gt_frames = has_gt if split_mask is None else has_gt & split_mask
    total_gt = int(gt_frames.sum())

    predicted = df["x_predicted"].notna()
    pred_with_gt = gt_frames & predicted
    errors_valid = df.loc[pred_with_gt, "xyz_error"].values
    fitting_valid = df.loc[pred_with_gt, "mean_error"].values

    # Handle NaN fitting errors: assign worst (highest) so they sort last
    nan_fit = np.isnan(fitting_valid)
    if nan_fit.any():
        max_fit = np.nanmax(fitting_valid) if not nan_fit.all() else 1.0
        fitting_valid = np.where(nan_fit, max_fit + 1.0, fitting_valid)

    metrics = {}
    ap_values = []
    for thr in _ACC_THRESHOLDS:
        acc = (errors_valid < thr).sum() / total_gt if total_gt > 0 else 0.0
        ap = _compute_ap_at_threshold(errors_valid, fitting_valid, total_gt, thr)
        metrics[f"acc_{thr}m"] = round(acc, 4)
        metrics[f"AP_{thr}m"] = round(ap, 4)
        ap_values.append(ap)

    metrics["mAP"] = round(float(np.mean(ap_values)), 4)
    metrics["coverage"] = round(pred_with_gt.sum() / total_gt, 4) if total_gt > 0 else 0.0
    if len(errors_valid) > 0 and not np.all(np.isnan(errors_valid)):
        metrics["mean_err"] = round(float(np.nanmean(errors_valid)), 4)
    else:
        metrics["mean_err"] = 0.0
    return metrics


def evaluate_ball_trajectory(
    ball_3d_gt: pd.DataFrame,
    ball_3d_dev: pd.DataFrame,
    dev__pauses: pd.DataFrame | None = None,
    split_df: pd.DataFrame | None = None,
):
    dev__ball_3d = ball_3d_dev
    df_gt = ball_3d_gt

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
    _logger.info(
        f"\n\nPauses_rate = {pauses_rate:.4f} of all frames;\n"
        + f"Coverage:  ball_detected={ball_det_coverage:5.4f};"
        f"  ball_smoothed={ball_smooth_coverage:5.4f}; "
        + f"camera={cam_coverage:5.4f}; camera+ball={cam_ball_coverage:5.4f}; camera+ball_smoothed={cam_ball_smooth_coverage:5.4f};\n"  # noqa: B950
    )

    df = dev__ball_3d.merge(df_gt, how="left", on="file_name")

    df["xy_error"] = np.sqrt(
        (df["x_predicted"] - df["x"]) ** 2 + (df["y_predicted"] - df["y"]) ** 2
    )
    df["z_error"] = np.abs(df["z_predicted"] - df["z"])
    if df["z"].mean() == 0:  # if we only have 2D GT data
        df["z_error"] *= 0
    df["xyz_error"] = np.linalg.norm(df[["xy_error", "z_error"]], axis=1)

    # to_meters
    df[["xy_error", "z_error", "xyz_error"]] /= 10

    is_arc = _resolve_is_arc(df, split_df)
    is_str = is_arc == False  # noqa: E712
    is_arc_mask = is_arc == True  # noqa: E712
    has_pred = df["x_predicted"].notna()
    predicted_str = is_str & has_pred
    predicted_arc = is_arc_mask & has_pred

    errors_df = pd.DataFrame(
        {
            "file_name": df["file_name"],
            "xyz_error": df["xyz_error"],
            "xy_error": df["xy_error"],
            "z_error": df["z_error"],
            "predicted_str": predicted_str,
            "predicted_arc": predicted_arc,
        }
    )

    error, xy_error, z_error = np.round(
        df[["xyz_error", "xy_error", "z_error"]].mean(), 3
    )

    str_error, str_xy_error, str_z_error = (
        np.round(df[predicted_str][["xyz_error", "xy_error", "z_error"]].mean(), 3)
        if len(df[predicted_str]) > 0
        else (0, 0, 0)
    )
    arc_error, arc_xy_error, arc_z_error = (
        np.round(df[predicted_arc][["xyz_error", "xy_error", "z_error"]].mean(), 3)
        if len(df[predicted_arc]) > 0
        else (0, 0, 0)
    )
    det_xy_error = (
        np.sqrt(
            (df[predicted_str]["x_pitch2D"] - df[predicted_str]["x"]) ** 2
            + (df[predicted_str]["y_pitch2D"] - df[predicted_str]["y"]) ** 2
        ).mean()
        / 10
    )

    _logger.info(
        "\n\nTrajectory_errors:\n"
        + f"Full: err={error:.3f}; xy_err={xy_error:.3f}; z_err={z_error:.3f}\n"
        + f"Str:  err={str_error:.3f}; xy_err={str_xy_error:.3f}; "
        f"z_err={str_z_error:.3f}\n"
        + f"Arc:  err={arc_error:.3f}; xy_err={arc_xy_error:.3f};"
        f" z_err={arc_z_error:.3f}\n"
        + f"Detection:       xy_err={det_xy_error:.3f}\n"
    )

    # --- New metrics: acc@Xm, AP@Xm, mAP ---
    has_gt = df["x"].notna()
    # is_str / is_arc_mask already derived from split_df / GT z above

    new_overall = _compute_new_metrics(df, has_gt)
    new_str = _compute_new_metrics(df, has_gt, split_mask=is_str)
    new_arc = _compute_new_metrics(df, has_gt, split_mask=is_arc_mask)

    new_metrics = {
        **new_overall,
        **{f"str_{k}": v for k, v in new_str.items()},
        **{f"arc_{k}": v for k, v in new_arc.items()},
    }

    # Balanced mAP: equal weight to str and arc (skip absent classes)
    str_mAP = new_str.get("mAP", 0.0)
    arc_mAP = new_arc.get("mAP", 0.0)
    str_gt = int((has_gt & is_str).sum())
    arc_gt = int((has_gt & is_arc_mask).sum())
    new_metrics["str_gt_frames"] = str_gt
    new_metrics["arc_gt_frames"] = arc_gt
    if str_gt > 0 and arc_gt > 0:
        new_metrics["mAP_balanced"] = round((str_mAP + arc_mAP) / 2, 4)
    elif str_gt > 0:
        new_metrics["mAP_balanced"] = round(str_mAP, 4)
    elif arc_gt > 0:
        new_metrics["mAP_balanced"] = round(arc_mAP, 4)
    else:
        new_metrics["mAP_balanced"] = 0.0

    # --- 3-way split: arc_short / arc_long ---
    has_3way = False
    if split_df is not None and "is_arc_long" in split_df.columns:
        split_lookup = split_df.set_index("frame_index")["is_arc_long"]
        mapped = df["file_name"].map(split_lookup)
        df["is_arc_long"] = mapped.where(mapped.notna(), False).astype(bool)
        is_arc_short = is_arc_mask & ~df["is_arc_long"]
        is_arc_long_mask = is_arc_mask & df["is_arc_long"]

        new_arc_short = _compute_new_metrics(df, has_gt, split_mask=is_arc_short)
        new_arc_long = _compute_new_metrics(df, has_gt, split_mask=is_arc_long_mask)

        new_metrics.update({f"arc_short_{k}": v for k, v in new_arc_short.items()})
        new_metrics.update({f"arc_long_{k}": v for k, v in new_arc_long.items()})

        arc_short_gt = (has_gt & is_arc_short).sum()
        arc_long_gt = (has_gt & is_arc_long_mask).sum()

        if str_gt > 0 and arc_short_gt > 0 and arc_long_gt > 0:
            new_metrics["mAP_balanced_3way"] = round(
                (str_mAP + new_arc_short["mAP"] + new_arc_long["mAP"]) / 3, 4
            )
            has_3way = True

    log_msg = (
        "\n\nNew metrics:\n"
        f"  mAP_balanced={new_metrics['mAP_balanced']:.4f}; "
        f"mAP_overall={new_overall['mAP']:.4f}; "
        f"mAP_str={new_str['mAP']:.4f}; mAP_arc={new_arc['mAP']:.4f}\n"
        f"  acc@1.0m={new_overall['acc_1.0m']:.4f}; "
        f"acc@1.0m_str={new_str['acc_1.0m']:.4f}; "
        f"acc@1.0m_arc={new_arc['acc_1.0m']:.4f}\n"
    )
    if has_3way:
        log_msg += (
            f"  mAP_balanced_3way={new_metrics['mAP_balanced_3way']:.4f}; "
            f"mAP_arc_short={new_arc_short['mAP']:.4f}; "
            f"mAP_arc_long={new_arc_long['mAP']:.4f}\n"
        )
    _logger.info(log_msg)

    coverage_mean_err_plot = _get_coverage_mean_err_plot(df)
    coverage_max_err_plot = _get_coverage_max_err_plot(df)

    ball3d_metrics = Ball3dMetrics(
        ball_detected_coverage=ball_det_coverage,
        ball_smoothed_coverage=ball_smooth_coverage,
        camera_coverage=cam_coverage,
        camera_ball_coverage=cam_ball_coverage,
    )
    ball3d_gt_metrics = Ball3dGTMetrics(
        pauses_rate=pauses_rate,
        full_err=error,
        full_xy_err=xy_error,
        full_z_err=z_error,
        str_err=str_error,
        str_xy_err=str_xy_error,
        str_z_err=str_z_error,
        arc_err=arc_error,
        arc_xy_err=arc_xy_error,
        arc_z_err=arc_z_error,
        **new_metrics,
    )

    return (
        coverage_mean_err_plot,
        coverage_max_err_plot,
        ball3d_metrics,
        ball3d_gt_metrics,
        errors_df,
    )


def _impute_fitting_errors(df):
    mask_pp = df["type"].isin(_PIVOT_POINT_TYPES)
    shift_m1_na = df["mean_error"].shift(-1).isna()
    shift_p1_na = df["mean_error"].shift(1).isna()
    shift_m1_lower = df["mean_error"].shift(-1) < df["mean_error"].shift(1)

    mask_m1 = mask_pp & ~shift_m1_na & (shift_m1_lower | shift_p1_na)
    mask_p1 = mask_pp & ~shift_p1_na & (~shift_m1_lower | shift_m1_na)

    df.loc[mask_m1, ["type", "mean_error"]] = df[["type", "mean_error"]].shift(-1)
    df.loc[mask_p1, ["type", "mean_error"]] = df[["type", "mean_error"]].shift(1)


def _add_pauses_data(df, df_pauses, split_df=None):
    # load pauses
    # directly annotated pauses
    if df_pauses is not None:
        starts = df_pauses["start_pause"].values
        ends = df_pauses["end_pause"].values
        pauses = [(int(start), int(end)) for start, end in zip(starts, ends)]
    else:
        pauses = []
    # pauses inferred from the splits.csv file
    if split_df is not None:
        frames = split_df["frame_index"].values
        max_frame = frames[-1]
        new_pauses = []
        start, end = None, None
        for f in range(max_frame + 1):
            is_pause = False
            for pause in pauses:
                s, e = pause
                if s <= f <= e:
                    is_pause = True
            if f not in frames:
                is_pause = True
            if is_pause:
                if start is None:
                    start = f
                end = f
            else:
                if end is not None:
                    new_pauses.append((start, end))
                    start, end = None, None
        if start is not None and end is not None:  # if the last frame is a pause
            new_pauses.append((start, end))
    else:
        new_pauses = []
    # Update the pauses list with the new pauses
    pauses = pauses + new_pauses

    # add pause column to the dataframe
    df["pause"] = False
    for i, f in enumerate(df['file_name']):
        for start, end in pauses:
            if start <= f <= end:
                df.loc[df["file_name"] == f, "pause"] = True
                break


def _get_coverage_mean_err_plot(df_ball_3d):
    df_plot = df_ball_3d[
        ["mean_error", "x_predicted", "xy_error", "xyz_error"]
    ].sort_values(by=["mean_error"])
    df_plot["xy_error_cumsum"] = df_plot["xy_error"].fillna(0).cumsum()
    df_plot["xyz_error_cumsum"] = df_plot["xyz_error"].fillna(0).cumsum()
    df_plot["frames_predicted_cumsum"] = df_plot["xyz_error"].notna().cumsum()

    df_plot = df_plot.groupby(["mean_error"], as_index=False).agg(
        {
            "frames_predicted_cumsum": "max",
            "xy_error_cumsum": "max",
            "xyz_error_cumsum": "max",
        }
    )
    df_plot["xyz_mean_error"] = (
        df_plot["xyz_error_cumsum"] / df_plot["frames_predicted_cumsum"]
    )
    df_plot["xy_mean_error"] = (
        df_plot["xy_error_cumsum"] / df_plot["frames_predicted_cumsum"]
    )
    df_plot["coverage"] = df_plot["frames_predicted_cumsum"] / len(df_ball_3d)
    df_plot = df_plot[df_plot.coverage > 0.1][
        ["coverage", "xy_mean_error", "xyz_mean_error"]
    ].copy()

    fig, ax = plt.subplots(figsize=(12, 8))
    cvr_mask = df_plot.coverage > 0.1
    ax.plot(
        df_plot[cvr_mask]["coverage"], df_plot[cvr_mask]["xy_mean_error"], linewidth=2.0
    )
    ax.plot(
        df_plot[cvr_mask]["coverage"],
        df_plot[cvr_mask]["xyz_mean_error"],
        linewidth=2.0,
    )
    ax.set(
        xlabel="Ball trajectory coverage",
        ylabel="Trajectory XY-error and XYZ-error (m)",
    )
    ax.grid()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    try:
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    except Exception:
        image = image.reshape((1600,2400,3))
    plt.close()

    return image


def _get_coverage_max_err_plot(df_ball_3d):
    df_plot = df_ball_3d[["xyz_error"]].sort_values(by=["xyz_error"])
    df_plot["coverage"] = df_plot["xyz_error"].notna().cumsum() / len(df_ball_3d)

    fig, ax = plt.subplots(figsize=(12, 8))
    cvr_mask = df_plot.xyz_error < 10
    ax.plot(
        df_plot[cvr_mask]["coverage"], df_plot[cvr_mask]["xyz_error"], linewidth=2.0
    )
    ax.set(
        xlabel="Ball trajectory coverage",
        ylabel="Trajectory Max XYZ-error (m)",
    )
    ax.grid(which="major", linewidth=1.5)
    plt.minorticks_on()
    ax.grid(which="minor", linewidth=0.5)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    try:
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    except Exception:
        image = image.reshape((1600,2400,3))
    plt.close()

    return image
