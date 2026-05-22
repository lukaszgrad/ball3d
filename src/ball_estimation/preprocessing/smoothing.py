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
from pykalman import KalmanFilter
from scipy.stats import iqr, norm

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# pykalman compat shim — older versions call inspect.getargspec (removed 3.11)
# ---------------------------------------------------------------------------

def _monkeypatches_inspect(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        inspect.getargspec = inspect.getfullargspec
        try:
            return fn(*args, **kwargs)
        finally:
            del inspect.getargspec
    return wrapper


# ---------------------------------------------------------------------------
# Core Kalman primitives (from respo.core.processing.smooth)
# ---------------------------------------------------------------------------

@_monkeypatches_inspect
def smooth_location(loc, ind, dt, process_sigma=1.0, loc_std=1.0, init_vel_std=1.0):
    """First-order Kalman Filter for 1D location smoothing.

    Parameters
    ----------
    loc : np.ndarray
        Location observations, array of floats.
    ind : np.ndarray
        Observation timestamps (sorted, increasing ints).
    dt : float
        Time delta between timestamps, in seconds.
    process_sigma : float or np.ndarray
        Square root of state process covariance.
    loc_std : float
        Observation standard deviation.
    init_vel_std : float
        Initial velocity standard deviation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (mean, covariance) arrays of shapes [T, 2] and [T, 2, 2],
        where T = max(ind) - min(ind) + 1.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        if len(loc) < 2:
            return (
                np.stack([loc, np.zeros_like(loc)], axis=1),
                np.zeros((len(loc), 2, 2)),
            )

        tm = np.array([[1, dt], [0, 1]])
        cm_base = np.array(
            [[1 / 4 * dt**4, 1 / 2 * dt**3], [1 / 2 * dt**3, dt**2]]
        )
        obs_matrix = np.array([[1, 0]])
        ind_start, ind_end = ind[0], ind[-1]
        total_len = ind_end - ind_start + 1

        if isinstance(process_sigma, np.ndarray):
            # Expand per-observation sigma to full timespan via interpolation
            sigma_full = np.full(total_len, np.median(process_sigma))
            sigma_full[ind - ind_start] = process_sigma
            cm = cm_base.reshape((1, 2, 2)) * sigma_full.reshape((-1, 1, 1))
        else:
            cm = cm_base * process_sigma**2

        obs = np.zeros((total_len,))
        obs[ind - ind_start] = loc
        mask = ~np.isin(np.arange(total_len), ind - ind_start)
        obs_masked = ma.masked_array(obs, mask)
        kf = KalmanFilter(
            transition_matrices=tm,
            transition_covariance=cm,
            observation_matrices=obs_matrix,
            observation_covariance=np.array([[loc_std]]) ** 2,
            initial_state_covariance=np.array([[loc_std, 0], [0, init_vel_std]]) ** 2,
            initial_state_mean=np.array([loc[0], 0]),
        )
        return kf.smooth(obs_masked)


def smooth_location2d(loc, ind, dt, process_sigma=1.0, loc_std=1.0, init_vel_std=1.0):
    """First-order Kalman Filter for 2D location smoothing.

    Each dimension is smoothed independently.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (mean, covariance) of shapes [T, 2, 2] and [T, 2, 2, 2].
    """
    x_mean, x_var = smooth_location(
        loc[:, 0], ind, dt,
        process_sigma=process_sigma, loc_std=loc_std, init_vel_std=init_vel_std,
    )
    y_mean, y_var = smooth_location(
        loc[:, 1], ind, dt,
        process_sigma=process_sigma, loc_std=loc_std, init_vel_std=init_vel_std,
    )
    return np.stack([x_mean, y_mean], axis=1), np.stack([x_var, y_var], axis=1)


# ---------------------------------------------------------------------------
# Higher-level smoothing functions (from respo.modules.ball.smoothing)
# ---------------------------------------------------------------------------

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


def _seq_inter(ind, max_gap=10):
    ind_seq = []
    for i, j in zip(ind[:-1], ind[1:]):
        if (j - i) < max_gap:
            ind_seq.extend(np.arange(i, j))
    if len(ind):
        ind_seq.append(ind[-1])
    return np.array(ind_seq, dtype=np.int64)


def _filtering_and_smoothing(
    df_pred, fps, process_sigma_bias, process_sigma_slope, obs_std, init_vel_std,
    ll_diff=5.0,
):
    """Run two-pass Kalman filter: first pass detects outliers, second pass smooths."""
    ind_start = df_pred["file_name"].min()
    ind_end = df_pred["file_name"].max() + 1
    ind_span = ind_end - ind_start
    _logger.debug("Start: %d, end: %d, missing: %.4f",
                  ind_start, ind_end, 1 - df_pred.shape[0] / ind_span)
    if process_sigma_slope == 0:
        process_sigma = process_sigma_bias
    else:
        process_sigma = process_sigma_bias + process_sigma_slope * df_pred["y_norm"].values

    # first pass — detect outliers
    ind = df_pred["file_name"].values
    ind0 = ind - ind[0]
    sigma_pass1 = process_sigma / 2 if np.isscalar(process_sigma) else process_sigma / 2
    mean, var = smooth_location2d(
        df_pred[["x_norm", "y_norm"]].values, ind,
        1.0 / fps, sigma_pass1, obs_std, init_vel_std,
    )
    ts = np.arange(ind_start, ind_end)
    loglik = np.zeros((len(ts), 2), dtype=np.float32)
    loglik[ind0] = norm(
        loc=mean[..., 0][ind0], scale=np.sqrt(var[..., 0, 0][ind0])
    ).logpdf(df_pred[["x_norm", "y_norm"]].values)
    ll = loglik[ind0].sum(1)
    ll_full = loglik.sum(1)
    ll_med, ll_iqr = np.median(ll), iqr(ll)

    mask = np.ones_like(ind0).astype("bool")
    _logger.debug("IQR: %.4f, median: %.4f", ll_iqr, ll_med)
    mask[_seq_inter(np.where(ll < (ll_med - ll_diff))[0])] = False
    if sum(mask) == 0:
        mask[0] = True

    # second pass — smooth without outliers
    start_shift = df_pred.loc[mask, "file_name"].min() - ind_start
    end_shift = ind_end - (df_pred.loc[mask, "file_name"].max() + 1)
    if end_shift == 0:
        ll_full = ll_full[start_shift:]
    else:
        ll_full = ll_full[start_shift:-end_shift]
    ind_start = df_pred.loc[mask, "file_name"].min()
    ind_end = df_pred.loc[mask, "file_name"].max() + 1
    ind = df_pred.loc[mask, "file_name"].values
    sigma_pass2 = process_sigma if np.isscalar(process_sigma) else process_sigma[mask]
    mean, var = smooth_location2d(
        df_pred.loc[mask, ["x_norm", "y_norm"]].values, ind,
        1.0 / fps, sigma_pass2, obs_std, init_vel_std,
    )
    ts = np.arange(ind_start, ind_end)
    loglik = np.zeros((len(ts), 2), dtype=np.float32)
    loglik[ind - ind_start] = norm(
        loc=mean[..., 0][ind - ind_start],
        scale=np.sqrt(var[..., 0, 0][ind - ind_start]),
    ).logpdf(df_pred.loc[mask, ["x_norm", "y_norm"]].values)
    df_smoothed = pd.DataFrame({
        "file_name": ts,
        "xk": mean[:, 0, 0],
        "yk": mean[:, 1, 0],
        "xvar": var[:, 0, 0, 0],
        "yvar": var[:, 1, 0, 0],
        "loglik": ll_full,
    })
    return df_smoothed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def kf_smoothing(
    df: pd.DataFrame,
    video_metadata: dict,
    hom_smooth_df: pd.DataFrame,
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

    # fill full frame range and merge homography
    df_ball = pd.DataFrame(
        {"file_name": range(df_ball["file_name"].min(), df_ball["file_name"].max() + 1)}
    ).merge(df_ball, on=["file_name"], how="left")
    df_homography = hom_smooth_df[
        ["frame_index", "h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8"]
    ].rename(columns={"frame_index": "file_name"})
    df_homography["file_name"] = df_homography["file_name"].astype(int)
    df_ball = df_ball.merge(df_homography, on=["file_name"], how="left")
    return df_ball
