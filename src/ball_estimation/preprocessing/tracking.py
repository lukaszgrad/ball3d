"""Ball tracking and track merging.

Step 1: Pick highest-scoring detection per frame (track_ball_inplay).
Step 2: Merge nearby tracks across small frame gaps (merge_ball_tracks).
"""

import logging

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


def track_ball_inplay(df_ball_detection: pd.DataFrame) -> pd.DataFrame:
    """Pick highest-scoring ball detection per frame, assign track_id=0.

    Parameters
    ----------
    df_ball_detection : pd.DataFrame
        Raw ball detections with at least columns: file_name, score.

    Returns
    -------
    pd.DataFrame
        One detection per frame with ``track_id`` column added.
    """
    df = (
        df_ball_detection.sort_values(by="score", ascending=False)
        .drop_duplicates("file_name")
        .sort_values(by="file_name")
    )
    df["track_id"] = 0
    return df


def merge_ball_tracks(
    df_ball_tracking: pd.DataFrame,
    gap_allowed: int = 5,
    max_d: float = 50.0,
) -> pd.DataFrame:
    """Merge nearby ball tracks using transitive closure.

    Tracks whose endpoints are within ``max_d`` pixels and separated by at
    most ``gap_allowed`` frames are merged into a single track.

    Parameters
    ----------
    df_ball_tracking : pd.DataFrame
        Ball track data with ``track_id``, ``file_name``, bbox columns.
    gap_allowed : int
        Maximum frame gap between track end and next track start.
    max_d : float
        Maximum Euclidean distance between track endpoint centres.

    Returns
    -------
    pd.DataFrame
        Same structure with merged ``track_id`` values.
    """
    if len(set(df_ball_tracking["track_id"])) == 1:
        return df_ball_tracking

    df_tracks = (
        df_ball_tracking.groupby(["track_id"])
        .agg({"file_name": [min, max]})
        .reset_index()
    )
    df_tracks.columns = ["track_id", "start", "end"]

    # duplicate tracks dataframe to get merge candidates
    df_tracks_1 = df_tracks.copy()
    df_tracks_1.columns = ["track_id_1", "start_1", "end_1"]

    # get all possible tracks continuations start frames
    df_shift = pd.DataFrame({"shift": range(1, gap_allowed + 1)})
    df_shift["i"] = 0
    df_tracks["i"] = 0
    df_tracks = df_tracks.merge(df_shift, on=["i"], how="outer").drop(columns="i")

    # get all possible pairs of tracks and their continuations
    df_tracks["start_1"] = df_tracks["end"] + df_tracks["shift"]
    df_tracks = df_tracks.merge(df_tracks_1, on=["start_1"], how="inner")

    # add ball detections BBoxes
    df_coords = df_ball_tracking[
        ["track_id", "file_name", "x0", "y0", "x1", "y1"]
    ].rename(columns={"file_name": "end"})
    df_tracks = df_tracks.merge(df_coords, on=["track_id", "end"], how="left")
    df_coords_1 = df_ball_tracking[
        ["track_id", "file_name", "x0", "y0", "x1", "y1"]
    ].rename(columns={"track_id": "track_id_1", "file_name": "start_1"})
    df_tracks = df_tracks.merge(
        df_coords_1, on=["track_id_1", "start_1"], how="left", suffixes=("", "_1")
    )

    # calculate distances between candidates' BBoxes and filter
    df_tracks["xk"] = (df_tracks["x0"] + df_tracks["x1"]) / 2
    df_tracks["yk"] = (df_tracks["y0"] + df_tracks["y1"]) / 2
    df_tracks["xk_1"] = (df_tracks["x0_1"] + df_tracks["x1_1"]) / 2
    df_tracks["yk_1"] = (df_tracks["y0_1"] + df_tracks["y1_1"]) / 2
    df_tracks["d"] = np.sqrt(
        (df_tracks["xk"] - df_tracks["xk_1"]) ** 2
        + (df_tracks["yk"] - df_tracks["yk_1"]) ** 2
    )
    df_tracks = df_tracks[df_tracks["d"] <= max_d].copy()

    # leave only 1 nearest continuation for each track
    df_tracks = df_tracks.sort_values(
        ["track_id", "d"], ascending=[True, True]
    ).drop_duplicates(subset=["track_id"], keep="first")

    # leave only 1 nearest previous track (if still multiple)
    df_tracks = df_tracks.sort_values(
        ["track_id_1", "d"], ascending=[True, True]
    ).drop_duplicates(subset=["track_id_1"], keep="first")

    df_tracks = df_tracks.sort_values("track_id", ascending=True).reset_index(drop=True)

    # get final tracks mapping (transitive closure)
    mapping = dict(zip(df_tracks["track_id_1"], df_tracks["track_id"]))
    mapping_full = mapping.copy()
    for i in range(
        df_ball_tracking["track_id"].min(), df_ball_tracking["track_id"].max() + 1
    ):
        if i not in mapping.keys():
            mapping_full[i] = i

    df_ball_tracking_merged = df_ball_tracking.copy()
    max_iterations = len(mapping) + 1
    for _ in range(max_iterations):
        if len(set(df_ball_tracking_merged["track_id"]) & set(mapping.keys())) == 0:
            break
        df_ball_tracking_merged["track_id"] = df_ball_tracking_merged["track_id"].map(
            mapping_full
        )
    else:
        _logger.warning("Transitive closure did not converge in %d iterations", max_iterations)

    return df_ball_tracking_merged
