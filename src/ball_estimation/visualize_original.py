import numpy as np
import cv2 as cv
from tqdm import tqdm
import pandas as pd
from typing import Dict, Any
import warnings

from ball_estimation.functions import PIVOT_POINT_TYPES
from ball_estimation.io.video import (
    Video,
    VideoMetadata,
    FrameGenerator,
)
from ball_estimation.camera import (
    Camera,
    scale_matrix,
    project_points,
)


def visualize_trajectory(
    video: Video,
    df_trajectory: pd.DataFrame,
    camera_smooth_df: pd.DataFrame,
    video_metadata: VideoMetadata,
    traj_params: Dict[str, Any],
    step_frame: int = 1,
) -> FrameGenerator:
    # fill nan values in predicted columns at trajectory edges
    if "is_repaired" not in df_trajectory.columns:
        df_trajectory["is_repaired"] = False
    pivot_mask = df_trajectory["type"].isin(PIVOT_POINT_TYPES + ["manual_annotation"])
    ball_cols = ["x_predicted", "y_predicted", "z_predicted", "z_predicted_raw"]
    df_trajectory.loc[pivot_mask, ball_cols] = (
        df_trajectory.loc[:, ball_cols].fillna(method="ffill").fillna(method="bfill")
    )

    df_trajectory = df_trajectory.set_index("file_name")
    camera_smooth_df = camera_smooth_df.set_index("frame_index")

    traj_trace_frames = (
        int(traj_params["trajectory_trace_seconds"] * video_metadata.fps) - 1
    )
    fading_speed = traj_params["fading_speed"]

    def out_gen() -> FrameGenerator:
        for image, ind in tqdm(video(), desc="visualize_trajectory"):
            ball_size = image.shape[0] // 360

            cam_numpy = camera_smooth_df.loc[
                ind,
                Camera.columns_pandas(),
            ].values

            if not np.isnan(cam_numpy[0]):
                cam = Camera.from_numpy(cam_numpy)
                proj_matrix = cam.to_projection_matrix()
                proj_matrix = (
                    scale_matrix(
                        video_metadata.width / 640, video_metadata.height / 360
                    )
                    @ proj_matrix
                )

                arc_points = df_trajectory.loc[
                    ind - traj_trace_frames : ind,
                    ["x_predicted", "y_predicted", "z_predicted_raw"],
                ].values
                arc_points[:, 2] *= -1
                ground_points = arc_points.copy()
                ground_points[:, 2] = 0

                pivot_points = (
                    df_trajectory.loc[ind - traj_trace_frames : ind, "type"]
                    .isin(
                        [
                            "pivot_point",
                            "high_pivot_point",
                            "wrong_pivot_point",
                            "manual_annotation",
                        ]
                    )
                    .values
                )
                pivot_points = pivot_points * traj_params["show_pivot_points"]

                repaired_points = df_trajectory.loc[
                    ind - traj_trace_frames : ind, "is_repaired"
                ].values
                repaired_points = (
                    repaired_points * traj_params["highlight_repaired_segments"]
                )

                manual_annotation = (
                    df_trajectory.loc[ind - traj_trace_frames : ind, "type"]
                    .isin(["manual_annotation"])
                    .values
                )
                manual_annotation = manual_annotation * traj_params["show_pivot_points"]

                arc_points_screen = project_points(proj_matrix, arc_points)
                ground_points_screen = project_points(proj_matrix, ground_points)
                num_points = len(arc_points_screen)

                for i, (
                    arc_point,
                    is_pivot_point,
                    is_repaired_point,
                    is_manual_annotation,
                ) in enumerate(
                    zip(
                        arc_points_screen,
                        pivot_points,
                        repaired_points,
                        manual_annotation,
                    )
                ):
                    if is_pivot_point and is_manual_annotation:
                        point_color = eval(traj_params["arc_pivot_annotated_color"])
                    elif is_pivot_point:
                        point_color = eval(traj_params["arc_pivot_color"])
                    elif is_repaired_point:
                        point_color = eval(traj_params["arc_repaired_color"])
                    else:
                        point_color = eval(traj_params["arc_regular_color"])

                    # TODO: debug and remove warning filter
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ball_center = tuple(arc_point.astype(np.int32))
                    ball_size_ = (
                        int(ball_size * traj_params["pivot_size_multiplier"])
                        if is_pivot_point
                        else ball_size
                    )
                    line_thickness = (
                        traj_params["circle_thickness_repaired"]
                        if is_pivot_point or is_repaired_point
                        else traj_params["circle_thickness_regular"]
                    )
                    if fading_speed > 0:
                        bbox_x0, bbox_x1, bbox_y0, bbox_y1 = _get_visualization_bbox(
                            ball_center, ball_size, line_thickness, video_metadata
                        )
                        bbox_copy = image[bbox_x0:bbox_x1, bbox_y0:bbox_y1].copy()
                        alpha = (1 - (num_points - i - 1) / num_points) ** fading_speed

                    image = cv.circle(
                        image,
                        ball_center,
                        ball_size_,
                        color=point_color,
                        thickness=line_thickness,
                    )
                    if fading_speed > 0:
                        image[bbox_x0:bbox_x1, bbox_y0:bbox_y1] = (
                            alpha * image[bbox_x0:bbox_x1, bbox_y0:bbox_y1]
                            + (1 - alpha) * bbox_copy
                        )

                for i, (
                    ground_point,
                    is_pivot_point,
                    is_repaired_point,
                    is_manual_annotation,
                ) in enumerate(
                    zip(
                        ground_points_screen,
                        pivot_points,
                        repaired_points,
                        manual_annotation,
                    )
                ):
                    if is_pivot_point and is_manual_annotation:
                        point_color = eval(traj_params["ground_pivot_annotated_color"])
                    elif is_pivot_point:
                        point_color = eval(traj_params["ground_pivot_color"])
                    elif is_repaired_point:
                        point_color = eval(traj_params["ground_repaired_color"])
                    else:
                        point_color = eval(traj_params["ground_regular_color"])

                    # TODO: debug and remove warning filter
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ball_center = tuple(ground_point.astype(np.int32))
                    ball_size_ = (
                        int(ball_size * traj_params["pivot_size_multiplier"])
                        if is_pivot_point
                        else ball_size
                    )
                    line_thickness = (
                        traj_params["circle_thickness_repaired"]
                        if is_pivot_point or is_repaired_point
                        else traj_params["circle_thickness_regular"]
                    )
                    if fading_speed > 0:
                        bbox_x0, bbox_x1, bbox_y0, bbox_y1 = _get_visualization_bbox(
                            ball_center, ball_size, line_thickness, video_metadata
                        )
                        bbox_copy = image[bbox_x0:bbox_x1, bbox_y0:bbox_y1].copy()
                        alpha = (1 - (num_points - i - 1) / num_points) ** fading_speed

                    image = cv.circle(
                        image,
                        ball_center,
                        ball_size_,
                        color=point_color,
                        thickness=line_thickness,
                    )
                    if fading_speed > 0:
                        image[bbox_x0:bbox_x1, bbox_y0:bbox_y1] = (
                            alpha * image[bbox_x0:bbox_x1, bbox_y0:bbox_y1]
                            + (1 - alpha) * bbox_copy
                        )

            if ind in df_trajectory.index and not np.isnan(
                np.sum(df_trajectory.loc[ind, ["xk", "yk"]].values)
            ):
                if traj_params["show_ball_detection"]:
                    image = cv.circle(
                        image,
                        center=tuple(
                            df_trajectory.loc[ind, ["xk", "yk"]].values.astype("int")
                        ),
                        radius=2 * ball_size,
                        color=eval(traj_params["arc_regular_color"]),
                        thickness=2,
                    )
                if traj_params["show_ball_uncertainty"] and (
                    "xvar" in df_trajectory.columns
                ):
                    image = cv.circle(
                        image,
                        center=tuple(
                            df_trajectory.loc[ind, ["xk", "yk"]].values.astype("int")
                        ),
                        radius=np.round(np.sqrt(df_trajectory.loc[ind, "xvar"])).astype(
                            "int"
                        ),
                        color=eval(traj_params["ground_regular_color"]),
                        thickness=traj_params["circle_thickness_uncertainty"],
                    )

            yield image, ind

    return out_gen


def _get_visualization_bbox(ball_center, ball_size, line_thickness, video_metadata):
    return (
        np.clip(
            ball_center[1] - ball_size - line_thickness - 1, 0, video_metadata.height
        ),
        np.clip(
            ball_center[1] + ball_size + line_thickness + 2, 0, video_metadata.height
        ),
        np.clip(
            ball_center[0] - ball_size - line_thickness - 1, 0, video_metadata.width
        ),
        np.clip(
            ball_center[0] + ball_size + line_thickness + 2, 0, video_metadata.width
        ),
    )
