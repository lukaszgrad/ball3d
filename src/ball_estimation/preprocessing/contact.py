"""Ball-player contact detection.

Detects spatial overlap between ball and player segmentation masks,
adds pitch/panorama coordinates and contact type flags.
"""

import ast
import logging
from typing import Any

import cv2 as cv
import numpy as np
import pandas as pd
from scipy.spatial import distance
from tqdm import tqdm

from ball_estimation.trajectory.data_processing import (
    add_pitch_pano_coords,
    get_ellipse_mask,
    outside_pitch,
)
from ball_estimation.processors import rle_to_bitmask

_logger = logging.getLogger(__name__)

INF = 1e6


def get_ball_bounding_boxes(df_ball: pd.DataFrame, shift: bool = True) -> pd.DataFrame:
    """Compute ball bounding boxes with ellipse axes, shifted to smoothed position."""
    if "segmentation" in df_ball.columns:
        segmentation = ["segmentation"]
    else:
        segmentation = []

    df_ball["a"] = (df_ball.x1 - df_ball.x0) / 2
    df_ball["b"] = (df_ball.y1 - df_ball.y0) / 2

    for col in ["a", "b"] + segmentation:
        df_ball.loc[df_ball["x0"].notna() | df_ball["xk"].notna(), col] = (
            df_ball[df_ball["x0"].notna() | df_ball["xk"].notna()]
            .groupby(["track_id"])[col]
            .transform(lambda x: x.ffill().bfill())
        )

    columns = ["file_name", "track_id", "x0", "x1", "y0", "y1", "xk", "yk", "a", "b"]
    df = df_ball[columns + segmentation].copy()
    df = df[df["x0"].notna() | df["xk"].notna()]

    if shift:
        df["dx"] = np.round(df.xk - df.x0 / 2 - df.x1 / 2).fillna(0)
        df["dy"] = np.round(df.yk - df.y0 / 2 - df.y1 / 2).fillna(0)
        df["x0"] = df.x0 + df.dx
        df["x1"] = df.x1 + df.dx
        df["y0"] = df.y0 + df.dy
        df["y1"] = df.y1 + df.dy

    df["x0"] = df["x0"].fillna(np.round(df.xk - df.a))
    df["y0"] = df["y0"].fillna(np.round(df.yk - df.b))
    df["x1"] = df["x1"].fillna(df.x0 + 2 * df.a)
    df["y1"] = df["y1"].fillna(df.y0 + 2 * df.b)
    return df


def get_players_bounding_boxes(df_main: pd.DataFrame) -> pd.DataFrame:
    """Select player/goalkeeper bounding boxes, prefix columns with ``p_``."""
    usecols = ["file_name", "detection_id", "x0", "x1", "y0", "y1", "segmentation"]

    if len(df_main) == 0:
        df_main = pd.DataFrame(columns=usecols + ["category"])
    df_main = df_main[df_main["category"].isin(["goalkeeper", "player"])][usecols].copy()

    for col in df_main.columns[1:]:
        df_main.rename(columns={col: "p_" + col}, inplace=True)
    return df_main


def _get_mask_contours(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv.findContours(
        (mask * 255).astype(np.uint8),
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_NONE,
    )
    if len(contours) == 0:
        return np.array([])
    contours = np.squeeze(np.vstack(contours), axis=1)
    return contours


def _add_distances(df: pd.DataFrame) -> None:
    """Compute pixel overlap and contour distance between ball and player masks."""
    df["common_points"] = 0
    df["distance"] = 1e3

    df["xk"] = df["xk"].fillna(0.5 * df["x0"] + 0.5 * df["x1"])
    df["yk"] = df["yk"].fillna(0.5 * df["y0"] + 0.5 * df["y1"])
    df[["x0", "y0", "x1", "y1", "p_x0", "p_y0", "p_x1", "p_y1"]] = df[
        ["x0", "y0", "x1", "y1", "p_x0", "p_y0", "p_x1", "p_y1"]
    ].astype(int)

    is_ball_segmentation = "segmentation" in df.columns

    for i in tqdm(range(len(df)), desc="Computing ball-player distances"):
        height, width = df.height.iloc[i], df.width.iloc[i]
        mask_start_x, mask_start_y = df.mask_start_x.iloc[i], df.mask_start_y.iloc[i]
        x0, y0, x1, y1 = df.x0.iloc[i], df.y0.iloc[i], df.x1.iloc[i], df.y1.iloc[i]
        p_x0, p_y0, p_x1, p_y1 = (
            df.p_x0.iloc[i], df.p_y0.iloc[i], df.p_x1.iloc[i], df.p_y1.iloc[i]
        )
        xk, yk = df.xk.iloc[i], df.yk.iloc[i]

        if is_ball_segmentation:
            segm_ball = df["segmentation"].iloc[i]
            if isinstance(segm_ball, str):
                segm_ball = ast.literal_eval(segm_ball)
            ball_area = rle_to_bitmask(np.array(segm_ball), y1 - y0, x1 - x0)
            ball = np.zeros((height, width)).astype(bool)
            ball[
                y0 - mask_start_y : y1 - mask_start_y,
                x0 - mask_start_x : x1 - mask_start_x,
            ] = ball_area
        else:
            ball = get_ellipse_mask(
                h=height, w=width,
                center=(xk - mask_start_x, yk - mask_start_y),
                axes=(df.a.iloc[i], df.b.iloc[i]),
            )

        segm_player = df["p_segmentation"].iloc[i]
        if isinstance(segm_player, str):
            segm_player = ast.literal_eval(segm_player)
        player_area = rle_to_bitmask(np.array(segm_player), p_y1 - p_y0, p_x1 - p_x0)
        player = np.zeros((height, width)).astype(bool)
        player[
            p_y0 - mask_start_y : p_y1 - mask_start_y,
            p_x0 - mask_start_x : p_x1 - mask_start_x,
        ] = player_area

        common_points = np.bitwise_and(ball, player).sum()
        if common_points > 0:
            dist = 0
        else:
            ball_contours = _get_mask_contours(ball)
            player_contours = _get_mask_contours(player)
            if len(ball_contours) and len(player_contours):
                dist = distance.cdist(ball_contours, player_contours).min()
            else:
                dist = INF

        df.iloc[i, df.columns.get_loc("common_points")] = common_points
        df.iloc[i, df.columns.get_loc("distance")] = dist

    df.drop(["mask_start_x", "mask_start_y", "width", "height"], axis=1, inplace=True)


def _add_contact_columns(
    df_ball: pd.DataFrame,
    out_margin: float,
    max_distance_for_contact_approval: float,
    min_ball_height_to_detect_high_pivot: float,
) -> None:
    """Add boolean contact flags: out, far_contact, close_contact, high_contact."""
    df_ball["out"] = outside_pitch(df_ball, out_margin)
    df_ball.loc[df_ball["x_pitch2D"].isna(), "out"] = False
    df_ball["far_contact"] = df_ball.distance < df_ball["epsilon"]
    df_ball["close_contact"] = (
        df_ball.distance < max_distance_for_contact_approval * df_ball["epsilon"]
    )
    df_ball["high_contact"] = (
        df_ball.ball_height_rel > min_ball_height_to_detect_high_pivot
    )


def detect_ball_player_contacts(
    df_main: pd.DataFrame,
    df_ball: pd.DataFrame,
    video_metadata: dict[str, Any],
    image_stitcher,
    epsilon_frac: float = 0.66,
    out_margin: float = 5.0,
    max_distance_for_contact_approval: float = 0.55,
    min_ball_height_to_detect_high_pivot: float = 0.17,
) -> pd.DataFrame:
    """Detect ball-player contacts and add contact columns to ball DataFrame.

    Parameters
    ----------
    df_main : pd.DataFrame
        Player/goalkeeper detections with category, bbox, segmentation columns.
    df_ball : pd.DataFrame
        Smoothed ball DataFrame with xk, yk, track_id, bbox columns.
    video_metadata : dict
        Must contain ``width``, ``height``.
    image_stitcher
        Image stitcher for panorama coordinate transforms.
    epsilon_frac : float
        Fraction of player height used as spatial tolerance.
    out_margin : float
        Pitch boundary margin (dm) for out-of-bounds detection.
    max_distance_for_contact_approval : float
        Distance threshold multiplier for close contact approval.
    min_ball_height_to_detect_high_pivot : float
        Relative ball height threshold for high contact detection.

    Returns
    -------
    pd.DataFrame
        ``df_ball`` augmented with contact columns.
    """
    _logger.info("Analyzing ball-players contacts ...")

    df = get_ball_bounding_boxes(df_ball)
    df_players = get_players_bounding_boxes(df_main)
    df = df.merge(df_players, on=["file_name"], how="left")

    # leave only frames with ball-player bounding boxes overlap
    df["epsilon"] = epsilon_frac * (df["p_y1"] - df["p_y0"])
    df = (
        df[
            (df.x1 > df.p_x0 - df.epsilon)
            & (df.x0 < df.p_x1 + df.epsilon)
            & (df.y1 > df.p_y0 - df.epsilon)
            & (df.y0 < df.p_y1 + df.epsilon)
        ]
        .copy()
        .reset_index(drop=True)
    )

    # coordinate for ball at player's feet level
    df["yk_corr"] = df.p_y1
    df["mask_start_x"] = df[["x0", "p_x0"]].min(axis=1).astype(int) - 1
    df["mask_start_y"] = df[["y0", "p_y0"]].min(axis=1).astype(int) - 1
    df["width"] = df[["x1", "p_x1"]].max(axis=1).astype(int) - df["mask_start_x"] + 2
    df["height"] = df[["y1", "p_y1"]].max(axis=1).astype(int) - df["mask_start_y"] + 2

    _add_distances(df)

    # leave one nearest player for each ball track
    df = (
        df.sort_values(
            ["file_name", "common_points", "distance"],
            ascending=(True, False, True),
        )
        .groupby(["file_name", "track_id"])
        .first()
        .reset_index()
    )
    df["ball_height_rel"] = (df.yk - df.p_y1) / (df.p_y0 - df.p_y1)

    df_ball = df_ball.merge(
        df[
            [
                "file_name",
                "track_id",
                "p_detection_id",
                "common_points",
                "distance",
                "ball_height_rel",
                "yk_corr",
                "epsilon",
            ]
        ],
        on=["file_name", "track_id"],
        how="left",
    ).fillna({"common_points": 0, "distance": 1e3, "ball_height_rel": -1})

    df_ball["yk_original"] = df_ball["yk"]
    df_ball["yk"] = df_ball["yk"] + df_ball["b"]

    add_pitch_pano_coords(df_ball, video_metadata, image_stitcher)
    _add_contact_columns(
        df_ball, out_margin, max_distance_for_contact_approval,
        min_ball_height_to_detect_high_pivot,
    )

    _logger.info("Analyzing ball-players contacts ... done")
    return df_ball
