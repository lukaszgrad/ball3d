import logging
import numpy as np
import pandas as pd
from ball_estimation.camera import Camera
from joblib import Parallel, delayed
from typing import Any, Dict, Tuple
from hydra.utils import instantiate
logger = logging.getLogger(__name__)

from ball_estimation.trajectory.estimator import TrajectoryEstimator

PIVOT_POINT_TYPES = ["pivot_point", "high_pivot_point", "wrong_pivot_point"]
DM2M = 0.1


def get_leap_pivots(
    df: pd.DataFrame,
    med_dist_threshold: float = 4.0,
    high_dist_threshold: float = 7.5,
    dist_frac_threshold: float = 1.5,
    angle_threshold: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts leap pivots from a trajectory DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing predicted ball trajectory.
    med_dist_threshold : float, optional
        Threshold for medium distance. Defaults to 4.0 dm.
    high_dist_threshold : float, optional
        Threshold for high distance. Defaults to 7.5 dm.
    dist_frac_threshold : float, optional
        Threshold for distance fraction. Defaults to 1.5.
    angle_threshold : float, optional
        Threshold for cosine angle. Defaults to 0.7.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two arrays:
        - pivots_list: Array of pivot filenames.
        - leap_pivots_list: Array of leap pivot filenames.
    """
    pivot_type_mask = df["type"].isin(PIVOT_POINT_TYPES)
    pivots_list = df[pivot_type_mask]["file_name"].values

    df["cos"] = calculate_cosine(df)
    df["leap_dist"] = calculate_leap_distance(df)

    big_xyz_leap = calculate_big_xyz_leap(df, high_dist_threshold, dist_frac_threshold)
    big_angle_leap = calculate_big_angle_leap(df, med_dist_threshold, angle_threshold)

    leap_pivots_list = df[
        (pivot_type_mask | pivot_type_mask.shift()) & (big_xyz_leap | big_angle_leap)
    ]["file_name"].values

    df["cos_shift"] = df["cos"].shift()
    save_leap_data(df)

    logger.info(f"Leap pivots list: {leap_pivots_list}")

    return pivots_list, leap_pivots_list


def calculate_cosine(df: pd.DataFrame) -> pd.Series:
    return -(
        df["x_predicted"].diff() * df["x_predicted"].diff(-1)
        + df["y_predicted"].diff() * df["y_predicted"].diff(-1)
    ) / (
        np.linalg.norm(df[["x_predicted", "y_predicted"]].diff(), axis=1)
        * np.linalg.norm(df[["x_predicted", "y_predicted"]].diff(-1), axis=1)
    )


def calculate_leap_distance(df: pd.DataFrame) -> pd.Series:
    return np.linalg.norm(
        df[["x_predicted", "y_predicted", "z_predicted"]].diff(), axis=1
    )


def calculate_big_xyz_leap(
    df: pd.DataFrame, high_dist_threshold: float, dist_frac_threshold: float
) -> pd.Series:
    return (
        (df["leap_dist"] > high_dist_threshold)
        & (df["leap_dist"] > dist_frac_threshold * df["leap_dist"].shift())
        & (df["leap_dist"] > dist_frac_threshold * df["leap_dist"].shift(-1))
    )


def calculate_big_angle_leap(
    df: pd.DataFrame, med_dist_threshold: float, angle_threshold: float
) -> pd.Series:
    return (
        (df["leap_dist"] > med_dist_threshold)
        & (df["cos"] < angle_threshold)
        & (df["cos"].shift() < angle_threshold)
    )


def save_leap_data(df: pd.DataFrame) -> None:
    df[
        [
            "file_name",
            "type",
            "x_predicted",
            "y_predicted",
            "z_predicted",
            "leap_dist",
            "cos",
            "cos_shift",
        ]
    ].to_csv("leaps.csv", index=False)


def remove_leap(df: pd.DataFrame, start: int, end: int) -> None:
    """
    Applies linear transformation to the trajectory segment to reduce the leap
    betweeen the ending point of the current trajectory segment
    and the starting point of the next trajectory segment.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing predicted ball trajectory.
    start : int
        Start frame of transformed trajectory segment.
    end : int
        End frame index of transformed trajectory segment.

    Returns
    -------
    None
        The function modifies the input DataFrame in-place.
    """
    for c in ["x_predicted", "y_predicted", "z_predicted"]:
        c0, c1, c2 = df.loc[df.file_name.isin([start, end - 1, end]), c].values
        delta = (c2 - c1 - (c2 - c0) / (end - start)) / (end - start - 1)
        df.loc[(df.file_name > start) & (df.file_name < end), c] = (
            df[c] + (df["file_name"] - start) * delta
        ).fillna(df[c])


def remove_leaps(df: pd.DataFrame, params: Dict[str, Any]) -> None:
    """
    Removes leaps from the ball trajectory based on the provided parameters.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing predicted ball trajectory.
    params : Dict[str, Any]
        A dictionary containing the following keys:
        - "med_dist_threshold" (float): The threshold for medium distance to identify leaps.
        - "high_dist_threshold" (float): The threshold for high distance to identify leaps.
        - "dist_frac_threshold" (float): The fractional threshold for distance
        to identify leaps.
        - "angle_threshold" (float): The threshold for the angle to identify leaps.

    Returns
    -------
    None
        This function does not return anything but modifies the dataframe in place.
    """
    pivots_list, leap_pivots_list = get_leap_pivots(
        df,
        params["med_dist_threshold"],
        params["high_dist_threshold"],
        params["dist_frac_threshold"],
        params["angle_threshold"],
    )

    for i, pivot in enumerate(pivots_list[1:], start=1):
        if pivot in leap_pivots_list or pivot + 1 in leap_pivots_list:
            start = pivots_list[i - 1] + 1
            end = pivot if pivot in leap_pivots_list else pivot + 1
            if end - start <= 1:
                logger.warning("Skipping leap removal at [%d, %d]: segment too short", start, end)
                continue
            matching = df.file_name.isin([start, end - 1, end]).sum()
            if matching < 3:
                logger.warning("Skipping leap removal at [%d, %d]: only %d/3 boundary frames found", start, end, matching)
                continue
            remove_leap(df, start, end)


def estimate_trajectory(
    ball_detection: pd.DataFrame,
    df: pd.DataFrame,
    ball_pivot_point: pd.DataFrame,
    camera: Camera,
    camera_smooth_df: pd.DataFrame,
    video_metadata: Dict[str, Any],
    estimator_name: str,
    estimator_parameters: Dict[str, Any],
    n_jobs: int = 1,
    step_frame: int = 1,
    start_sec: int = 0,
    end_sec: int = -1,
) -> pd.DataFrame:
    # TODO: consider exports for num threads limiting

    # Filter input data to the requested time range before estimation
    fps = float(video_metadata["fps"])
    start_frame = int(fps * start_sec)
    end_frame = int(fps * end_sec) if end_sec >= 0 else ball_detection["file_name"].max() + 1
    if start_frame > 0 or end_frame < ball_detection["file_name"].max() + 1:
        logger.info(f"Filtering to frame range [{start_frame}, {end_frame})")
        df = df[(df["file_name"] >= start_frame) & (df["file_name"] < end_frame)].copy()
        ball_pivot_point = ball_pivot_point[
            (ball_pivot_point["file_name"] >= start_frame) & (ball_pivot_point["file_name"] < end_frame)
        ].copy()

    camera_smooth_df["frame_index"] //= step_frame

    estimator = instantiate({"_target_":f"ball_estimation.trajectory.estimator.{estimator_name}",
                 **estimator_parameters,
                 "camera":camera, "camera_smooth_df": camera_smooth_df,
                 "video_metadata":video_metadata})

    if not estimator_parameters["use_multiple_balls"]:
        df = create_fake_track_ids(df, estimator_parameters)

    if not estimator_parameters["debug"]:
        dfs = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(predict_one_track)(
                id, df, ball_pivot_point, camera_smooth_df, estimator, estimator_parameters
            )
            for id in set(df.track_id.dropna())
        )
    else:
        dfs = [predict_one_track(id, df, ball_pivot_point, camera_smooth_df, estimator, estimator_parameters) for id in set(df.track_id.dropna())]
    df_ = pd.concat(dfs, ignore_index=True).sort_values(
        ["file_name", "track_id"]
    )


    df_ = fill_missing_predictions(df_)
    remove_leaps(df_, estimator_parameters)
    df_["z_predicted_raw"] = df_["z_predicted"]

    df_["file_name"] *= step_frame

    df_ = process_trajectory_data(
        df_,
        ball_detection,
        camera_smooth_df,
        video_metadata,
        start_sec,
        end_sec,
        step_frame,
    )

    df_ = calculate_velocities_and_accelerations(
        df_, step_frame, float(video_metadata["fps"])
    )

    df_ = remove_extreme_values_at_pivot_points(df_)

    # TODO: looks like stuff from respo that is not necessary
    df_["is_repaired"] = False
    df_["is_imputed"] = False

    return df_


def create_fake_track_ids(
    df: pd.DataFrame, estimator_parameters: Dict[str, Any]
) -> pd.DataFrame:
    df["predictable"] = df["xk"].notna() & df["h0"].notna()
    df["misses"] = df["predictable"].groupby(df["predictable"].cumsum()).cumcount()
    misses_allowed = (
        int(
            estimator_parameters["too_long_track"]
            * estimator_parameters["nans_allowed"]
        )
        + 1
    )
    df["track_id"] = (df["misses"] == misses_allowed).cumsum()
    return df


def predict_one_track(
    id: int,
    df: pd.DataFrame,
    ball_pivot_point: pd.DataFrame,
    camera_smooth_df: pd.DataFrame,
    estimator: TrajectoryEstimator,
    estimator_parameters: Dict[str, Any],
) -> pd.DataFrame:
    logger.info(f"Predict for track_id={id}")
    df_ = df[df.track_id == id].copy()
    if estimator_parameters["use_multiple_balls"]:
        ball_pivot_point_ = ball_pivot_point[ball_pivot_point.track_id == id].copy()
    else:
        ball_pivot_point_ = ball_pivot_point.copy()

    df_ = pd.DataFrame(
        {"file_name": range(df_["file_name"].min(), df_["file_name"].max() + 1)}
    ).merge(df_, on=["file_name"], how="left")
    df_ = df_.merge(
        camera_smooth_df[["frame_index", "rot_x", "fx"]].rename(
            columns={"frame_index": "file_name"}
        ),
        on="file_name",
        how="left",
    )
    return estimator.estimate_trajectory(df_, ball_pivot_point_)


def fill_missing_predictions(df: pd.DataFrame) -> pd.DataFrame:
    pivot_types = df["type"].isin(PIVOT_POINT_TYPES)
    preds = ["x_predicted", "y_predicted", "z_predicted", "z_predicted_raw"]
    df[preds] = df[preds].astype(float)
    df.loc[pivot_types, preds] = (
        df.loc[pivot_types, preds]
        .fillna(df[preds].shift() + df[preds].diff().shift())
        .fillna(df[preds].shift(-1) + df[preds].diff(-1).shift(-1))
    )
    return df


def process_trajectory_data(
    df: pd.DataFrame,
    ball_detection: pd.DataFrame,
    camera_smooth_df: pd.DataFrame,
    video_metadata: Dict[str, Any],
    start_sec: int,
    end_sec: int,
    step_frame: int,
) -> pd.DataFrame:
    FPS = float(video_metadata["fps"])
    start_frame = int(FPS * start_sec)
    end_frame = (
        int(FPS * end_sec) if end_sec >= 0 else ball_detection["file_name"].max() + 1
    )

    ball_detected = create_ball_detected_df(start_frame, end_frame, ball_detection)
    camera_detected = create_camera_detected_df(camera_smooth_df)

    df_ = (
        ball_detected.merge(camera_detected, on=["file_name"], how="left")
        .merge(df, on=["file_name"], how="left")
        .fillna(
            {
                "track_id": -1,
                "is_ball_detected": False,
                "is_camera_detected": False,
            }
        )
    )
    df_["track_id"] = df_["track_id"].astype(int)
    return df_


def create_ball_detected_df(
    start_frame: int, end_frame: int, ball_detection: pd.DataFrame
) -> pd.DataFrame:
    return pd.DataFrame({"file_name": range(start_frame, end_frame)}).merge(
        pd.DataFrame(
            {
                "file_name": list(set(ball_detection["file_name"])),
                "is_ball_detected": True,
            }
        ),
        on=["file_name"],
        how="left",
    )


def create_camera_detected_df(camera_smooth_df: pd.DataFrame) -> pd.DataFrame:
    camera_detected = camera_smooth_df[camera_smooth_df["rot_x"].notna()][
        ["frame_index"]
    ].rename(columns={"frame_index": "file_name"})
    camera_detected["is_camera_detected"] = True
    return camera_detected


def calculate_velocities_and_accelerations(
    df: pd.DataFrame, step_frame: int, fps: float
) -> pd.DataFrame:
    time_difference_per_row = step_frame / fps

    for axis in ["x", "y", "z"]:
        df[f"v{axis}_predicted"] = (
            df.groupby("track_id")[f"{axis}_predicted"].diff()
            * DM2M
            / time_difference_per_row
        )  # m/s
        df[f"a{axis}_predicted"] = (
            df.groupby("track_id")[f"v{axis}_predicted"].diff()
            / time_difference_per_row
        )  # m/s^2

    for i in ["v", "a"]:
        df[f"{i}_abs_predicted"] = np.linalg.norm(
            df[[f"{i}x_predicted", f"{i}y_predicted", f"{i}z_predicted"]].astype(float),
            axis=1,
        )

    return df


def remove_extreme_values_at_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    pp_frames = ["pivot_point", "high_pivot_point"]
    v_columns = ["vx_predicted", "vy_predicted", "vz_predicted", "v_abs_predicted"]
    a_columns = ["ax_predicted", "ay_predicted", "az_predicted", "a_abs_predicted"]
    df.loc[
        df["type"].isin(pp_frames) | df["type"].shift().isin(pp_frames),
        v_columns + a_columns,
    ] = np.nan

    for col in v_columns + a_columns:
        df[col] = df.groupby("track_id")[col].transform(
            lambda x: x.interpolate(limit=2)
        )

    return df
