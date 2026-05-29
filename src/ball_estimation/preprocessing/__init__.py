"""Ball preprocessing pipeline.

Transforms raw ball detections into the ``df_merged_ball_player`` and
``ball_pivot_point`` DataFrames consumed by ``estimate_trajectory.py``.
"""

import logging
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from ball_estimation.preprocessing.contact import detect_ball_player_contacts
from ball_estimation.preprocessing.smoothing import kf_smoothing
from ball_estimation.preprocessing.tracking import merge_ball_tracks, track_ball_inplay

_logger = logging.getLogger(__name__)


def preprocess_ball(
    ball_detection_df: pd.DataFrame,
    player_detection_df: pd.DataFrame,
    video_metadata: dict[str, Any],
    preprocessing_cfg: DictConfig,
    step_frame: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full ball preprocessing pipeline.

    Parameters
    ----------
    ball_detection_df : pd.DataFrame
        Raw ball detections (x0, y0, x1, y1, score, file_name, ...).
    player_detection_df : pd.DataFrame
        Player/goalkeeper detections with category, bbox, segmentation.
    video_metadata : dict
        Must contain fps, width, height.
    image_stitcher
        Panorama stitcher for coordinate transforms.
    preprocessing_cfg : DictConfig
        Hydra preprocessing config group.
    step_frame : int
        Frame subsampling rate.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_merged_ball_player, ball_pivot_point)
    """
    cfg = preprocessing_cfg

    # Step 1: Ball tracking
    _logger.info("Step 1/5: Ball tracking ...")
    if "score" in ball_detection_df.columns:
        df_tracked = track_ball_inplay(ball_detection_df)
    else:
        _logger.warning("No 'score' column in ball detections — skipping track_ball_inplay")
        df_tracked = ball_detection_df.copy()

    # Step 2: Track merging
    _logger.info("Step 2/5: Track merging ...")
    df_merged = merge_ball_tracks(
        df_tracked,
        gap_allowed=cfg.merge.gap_allowed,
        max_d=cfg.merge.max_d,
    )

    # Step 3: 2D Kalman smoothing
    _logger.info("Step 3/5: 2D Kalman smoothing ...")
    df_ball = kf_smoothing(
        df_merged,
        video_metadata=video_metadata,
        max_missing_sec=cfg.smoothing.max_missing_sec,
        process_sigma_bias=cfg.smoothing.process_sigma_bias,
        process_sigma_slope=cfg.smoothing.process_sigma_slope,
        obs_std=cfg.smoothing.obs_std,
        init_vel_std=cfg.smoothing.init_vel_std,
        ll_diff=cfg.smoothing.ll_diff,
    )
    if len(df_ball) == 0:
        _logger.warning("Filtering produced empty DataFrame — no valid intervals found")
        empty_merged = pd.DataFrame()
        empty_pivot = pd.DataFrame(
            columns=["file_name", "track_id", "pivot_probability", "pivot_point"]
        )
        return empty_merged, empty_pivot

    # restore track id
    if "track_id" not in df_ball.columns:
        df_ball["track_id"] = 0

    # Step 4: Ball-player contact detection
    _logger.info("Step 4/5: Ball-player contact detection ...")
    df_merged_ball_player = detect_ball_player_contacts(
        df_main=player_detection_df,
        df_ball=df_ball,
        video_metadata=video_metadata,
        epsilon_frac=cfg.contact.epsilon_frac,
        out_margin=cfg.contact.out_margin,
        max_distance_for_contact_approval=cfg.contact.max_distance_for_contact_approval,
        min_ball_height_to_detect_high_pivot=cfg.contact.min_ball_height_to_detect_high_pivot,
    )


    # Drop intermediate columns not needed downstream
    _intermediate_cols = [
        "ball", "ball_diff", "ball_ok", "ball_roll", "detection_id",
        "epsilon", "flag", "flag_roll", "interval_index", "interval_ok",
        "loglik", "xc", "yc",
    ]
    df_merged_ball_player.drop(
        columns=[c for c in _intermediate_cols if c in df_merged_ball_player.columns],
        inplace=True,
    )

    _logger.info("Preprocessing complete.")
    return df_merged_ball_player
