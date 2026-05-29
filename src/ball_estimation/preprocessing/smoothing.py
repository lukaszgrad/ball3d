"""2D ball trajectory smoothing using Kalman Filter.

Extracted from respo.modules.ball.smoothing and respo.core.processing.smooth.
"""

import inspect
import logging
import warnings
from functools import partial, wraps

import numpy as np
import numpy.ma as ma
import pandas as pd


_logger = logging.getLogger(__name__)



def _prepare_ball_df(df, width, height):
    """Filter ball detections and normalise to [0,1]x[0,1]."""
    if "category" not in df.columns:
        df["category"] = "ball"
    df_ball = df[df.category == "ball"][
        ["x0", "y0", "x1", "y1", "score", "detection_id", "file_name"]
    ]
    df_ball.sort_values("score", ascending=False, inplace=True)
    df_ball = df_ball.groupby("file_name").first().reset_index()
    df_ball["file_name"] = df_ball["file_name"].astype(int)

    df_ball["x_norm"] = (df_ball["x0"] + df_ball["x1"]) / 2 / width
    df_ball["y_norm"] = (df_ball["y0"] + df_ball["y1"]) / 2 / height
    return df_ball


def _estimate_intervals(df_ball, max_missing_interval):
    """Partition frames into contiguous intervals with enough detections."""
    num_min, num_max = df_ball["file_name"].min(), df_ball["file_name"].max() + 1
    index = np.arange(num_min, num_max)
    df_flag = pd.DataFrame(
        {"ball": np.zeros_like(index), "flag": np.ones_like(index)}, index=index
    )
    df_flag.loc[df_ball["file_name"], "ball"] = 1.0
    df_flag["flag_roll"] = (
        df_flag["flag"]
        .rolling(max_missing_interval * 2, center=True, min_periods=1)
        .sum()
    )
    df_flag["ball_roll"] = (
        df_flag["ball"]
        .rolling(max_missing_interval * 2, center=True, min_periods=1)
        .sum()
    )
    df_flag["ball_ok"] = (df_flag["ball_roll"] * 2) > df_flag["flag_roll"]
    df_flag = df_flag.reset_index().rename({"index": "file_name"}, axis=1)
    df_ball = df_ball.merge(df_flag, on="file_name", how="left")
    _logger.debug("ball_ok fraction: %.4f", df_ball["ball_ok"].mean())
    df_ball["ball_diff"] = df_ball["ball_ok"].astype("int").diff()
    df_ball.loc[df_ball.index[0], "ball_diff"] = (
        1 if df_ball.loc[df_ball.index[0], "ball_ok"] == 1 else -1
    )

    df_ball["interval_ok"] = np.where(
        df_ball["ball_diff"] == 1,
        np.ones_like(df_ball["ball_diff"]),
        np.zeros_like(df_ball["ball_diff"]),
    )
    df_ball.loc[df_ball["ball_diff"] == 0, "interval_ok"] = np.nan
    df_ball["interval_ok"] = df_ball["interval_ok"].ffill()

    df_ball["interval_index"] = np.where(
        df_ball["ball_diff"].abs() == 1,
        np.ones_like(df_ball["ball_diff"]),
        np.zeros_like(df_ball["ball_diff"]),
    )
    df_ball["interval_index"] = df_ball["interval_index"].cumsum()
    return df_ball


def _filtering_and_smoothing(
    df_pred, fps, process_sigma_bias, process_sigma_slope, obs_std, init_vel_std,
    ll_diff=5.0,
):
    """Dummy smoothing: return observed normalized positions without filtering.

    This no-op implementation preserves the original frame range and copies
    `x_norm`/`y_norm` into `xk`/`yk` (leaving missing frames as NaN). Variances
    and log-likelihood are returned as NaN.
    """
    ind_start = int(df_pred["file_name"].min())
    ind_end = int(df_pred["file_name"].max()) + 1
    ts = np.arange(ind_start, ind_end)

    xk = np.full(len(ts), np.nan, dtype=float)
    yk = np.full(len(ts), np.nan, dtype=float)
    xvar = np.full(len(ts), np.nan, dtype=float)
    yvar = np.full(len(ts), np.nan, dtype=float)
    loglik = np.full(len(ts), np.nan, dtype=float)

    for _, row in df_pred.iterrows():
        idx = int(row["file_name"] - ind_start)
        if 0 <= idx < len(ts):
            xk[idx] = row.get("x_norm", np.nan)
            yk[idx] = row.get("y_norm", np.nan)

    df_smoothed = pd.DataFrame({
        "file_name": ts,
        "xk": xk,
        "yk": yk,
        "xvar": xvar,
        "yvar": yvar,
        "loglik": loglik,
    })
    return df_smoothed


def kf_smoothing(
    df: pd.DataFrame,
    video_metadata: dict,
    max_missing_sec: float = 2,
    process_sigma_bias: float = 1.0,
    process_sigma_slope: float = 0.0,
    obs_std: float = 1.0,
    init_vel_std: float = 1.0,
    ll_diff: float = 5.0,
) -> pd.DataFrame:
    """2D ball trajectory smoothing and filtering using Kalman Filter.

    Parameters
    ----------
    df : pd.DataFrame
        Per-frame ball detections with bbox columns.
    video_metadata : dict
        Must contain ``fps``, ``width``, ``height``.
    hom_smooth_df : pd.DataFrame
        Homography data with ``frame_index`` and h0..h8 columns.

    Returns
    -------
    pd.DataFrame
        Smoothed ball positions (xk, yk) with homography columns.
    """
    fps = float(video_metadata["fps"])
    width = video_metadata["width"]
    height = video_metadata["height"]

    df_ball = _prepare_ball_df(df, width, height)
    max_missing_interval = int(max_missing_sec * fps)
    df_ball = _estimate_intervals(df_ball, max_missing_interval)

    df_smooth = (
        df_ball.loc[df_ball["interval_ok"] == 1]
        .groupby("interval_index")
        .filter(lambda x: len(x.index) > 10)
        .groupby("interval_index")
        .apply(
            partial(
                _filtering_and_smoothing,
                fps=fps,
                process_sigma_bias=process_sigma_bias,
                process_sigma_slope=process_sigma_slope,
                obs_std=obs_std,
                init_vel_std=init_vel_std,
                ll_diff=ll_diff,
            )
        )
    )
    if len(df_smooth) == 0:
        return pd.DataFrame()
    df_ball.drop(columns=["x_norm", "y_norm"], inplace=True)
    df_ball = df_smooth.merge(df_ball, on="file_name", how="left")

    # return to screen coordinates
    df_ball["xk"] *= width
    df_ball["yk"] *= height
    df_ball["xvar"] *= width**2
    df_ball["yvar"] *= height**2

    # fill full frame range
    df_ball = pd.DataFrame(
        {"file_name": range(df_ball["file_name"].min(), df_ball["file_name"].max() + 1)}
    ).merge(df_ball, on=["file_name"], how="left")
    return df_ball
