import numpy as np
import cv2 as cv
from tqdm import tqdm
import pandas as pd
from typing import Dict, Any, Tuple, NamedTuple, List, Optional
import warnings
from dataclasses import dataclass

from ball_estimation.functions import PIVOT_POINT_TYPES
from ball_estimation.io.video import Video, VideoMetadata, FrameGenerator
from ball_estimation.camera import Camera, scale_matrix, project_points


@dataclass
class TrajectoryParams:
    trajectory_trace_seconds: float
    fading_speed: float
    show_pivot_points: bool
    highlight_repaired_segments: bool
    pivot_size_multiplier: float
    circle_thickness_repaired: int
    circle_thickness_regular: int
    show_ball_detection: bool
    show_ball_uncertainty: bool
    circle_thickness_uncertainty: int
    arc_regular_color: str
    ground_regular_color: str
    arc_pivot_color: str
    ground_pivot_color: str
    arc_repaired_color: str
    ground_repaired_color: str
    arc_pivot_annotated_color: str
    ground_pivot_annotated_color: str
    arc_gt_color: str
    ground_gt_color: str
    show_ground_truth: bool


class PointAttributes(NamedTuple):
    is_pivot_point: bool
    is_repaired_point: bool
    is_manual_annotation: bool
    is_gt: bool


class TrajectoryPoint(NamedTuple):
    screen_point: np.ndarray
    attributes: PointAttributes


def visualize_trajectory(
    video: Video,
    df_trajectory: pd.DataFrame,
    camera_smooth_df: pd.DataFrame,
    video_metadata: VideoMetadata,
    traj_params: Dict[str, Any],
    step_frame: int = 1,
    df_trajectory_gt: Optional[pd.DataFrame] = None,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> FrameGenerator:
    """
    Visualize the ball trajectory on video frames.

    Args:
        video (Video): Input video object.
        df_trajectory (pd.DataFrame): DataFrame containing trajectory data.
        camera_smooth_df (pd.DataFrame): DataFrame with smoothed camera data.
        video_metadata (VideoMetadata): Metadata of the video.
        traj_params (Dict[str, Any]): Trajectory visualization parameters.
        step_frame (int, optional): Frame step size. Defaults to 1.
        df_trajectory_gt (Optional[pd.DataFrame], optional): Ground truth trajectory data. Defaults to None.

    Returns:
        FrameGenerator: Generator yielding visualized frames.
    """
    df_trajectory = preprocess_trajectory_data(df_trajectory)
    camera_smooth_df = camera_smooth_df.set_index("frame_index")

    # Preprocess ground truth trajectory data if provided
    if df_trajectory_gt is not None:
        df_trajectory_gt = df_trajectory_gt.set_index("file_name")

    params = TrajectoryParams(**traj_params)
    traj_trace_frames = int(params.trajectory_trace_seconds * video_metadata.fps) - 1

    return generate_visualized_frames(
        video,
        df_trajectory,
        camera_smooth_df,
        video_metadata,
        params,
        traj_trace_frames,
        step_frame,
        df_trajectory_gt,
        start_frame,
        end_frame,
    )


def preprocess_trajectory_data(df_trajectory: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the trajectory DataFrame."""
    if "is_repaired" not in df_trajectory.columns:
        df_trajectory["is_repaired"] = False

    pivot_mask = df_trajectory["type"].isin(PIVOT_POINT_TYPES + ["manual_annotation"])
    ball_cols = ["x_predicted", "y_predicted", "z_predicted", "z_predicted_raw"]
    df_trajectory.loc[pivot_mask, ball_cols] = (
        df_trajectory.loc[:, ball_cols].fillna(method="ffill").fillna(method="bfill")
    )

    return df_trajectory.set_index("file_name")


def generate_visualized_frames(
    video: Video,
    df_trajectory: pd.DataFrame,
    camera_smooth_df: pd.DataFrame,
    video_metadata: VideoMetadata,
    params: TrajectoryParams,
    traj_trace_frames: int,
    step_frame: int,
    df_trajectory_gt: Optional[pd.DataFrame] = None,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> FrameGenerator:
    """Generate frames with visualized trajectory."""
    for image, ind in tqdm(video(), desc="visualize_trajectory"):
        if ind < start_frame:
            continue
        if end_frame is not None and ind >= end_frame:
            break

        ball_size = image.shape[0] // 360
        cam_numpy = camera_smooth_df.loc[ind, Camera.columns_pandas()].values

        if not np.isnan(cam_numpy[0]):
            image = process_frame_with_trajectory(
                image,
                ind,
                cam_numpy,
                df_trajectory,
                video_metadata,
                params,
                traj_trace_frames,
                ball_size,
                df_trajectory_gt,
            )

        image = add_ball_detection_and_uncertainty(
            image, ind, df_trajectory, params, ball_size
        )
        yield image, ind


def process_frame_with_trajectory(
    image: np.ndarray,
    ind: int,
    cam_numpy: np.ndarray,
    df_trajectory: pd.DataFrame,
    video_metadata: VideoMetadata,
    params: TrajectoryParams,
    traj_trace_frames: int,
    ball_size: int,
    df_trajectory_gt: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Process a single frame with trajectory visualization."""
    cam = Camera.from_numpy(cam_numpy)
    proj_matrix = calculate_projection_matrix(cam, video_metadata)

    trajectory_points = extract_trajectory_points(
        df_trajectory, ind, traj_trace_frames, proj_matrix, params
    )

    image = draw_trajectory_points(
        image,
        trajectory_points,
        params,
        ball_size,
        video_metadata,
    )

    if df_trajectory_gt is not None:
        gt_trajectory_points = extract_trajectory_points(
            df_trajectory_gt, ind, traj_trace_frames, proj_matrix, params, is_gt=True
        )
        image = draw_trajectory_points(
            image,
            gt_trajectory_points,
            params,
            ball_size,
            video_metadata,
        )

        # Calculate and print the distance between predicted and ground truth ball positions
        if ind in df_trajectory_gt.index and ind in df_trajectory.index:
            pred_coords = df_trajectory.loc[
                ind, ["x_predicted", "y_predicted", "z_predicted_raw"]
            ].values
            gt_coords = df_trajectory_gt.loc[ind, ["x", "y", "z"]].values
            distance = np.linalg.norm(pred_coords - gt_coords)

            # Add distance text to the image
            cv.putText(
                image,
                f"Distance: {distance:.2f}dm",
                (image.shape[1] - 270, 30),  # Top right corner
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),  # Red color
                2,
                cv.LINE_AA,
            )

    return image


def calculate_projection_matrix(
    cam: Camera, video_metadata: VideoMetadata
) -> np.ndarray:
    """Calculate the projection matrix for the current frame."""
    proj_matrix = cam.to_projection_matrix()
    return (
        scale_matrix(video_metadata.width / 640, video_metadata.height / 360)
        @ proj_matrix
    )


def extract_trajectory_points(
    df_trajectory: pd.DataFrame,
    ind: int,
    traj_trace_frames: int,
    proj_matrix: np.ndarray,
    params: TrajectoryParams,
    is_gt: bool = False,
) -> Tuple[List[TrajectoryPoint], List[TrajectoryPoint]]:
    """Extract arc and ground points from the trajectory data."""
    if is_gt:
        coords = ["x", "y", "z"]
    else:
        coords = ["x_predicted", "y_predicted", "z_predicted_raw"]

    arc_points = df_trajectory.loc[ind - traj_trace_frames : ind, coords].values
    arc_points[:, 2] *= -1
    ground_points = arc_points.copy()
    ground_points[:, 2] = 0

    arc_points_screen = project_points(proj_matrix, arc_points)
    ground_points_screen = project_points(proj_matrix, ground_points)

    point_attributes = extract_point_attributes(
        df_trajectory, ind, traj_trace_frames, params, is_gt
    )

    arc_trajectory_points = [
        TrajectoryPoint(point, attr)
        for point, attr in zip(arc_points_screen, point_attributes)
    ]
    ground_trajectory_points = [
        TrajectoryPoint(point, attr)
        for point, attr in zip(ground_points_screen, point_attributes)
    ]

    return arc_trajectory_points, ground_trajectory_points


def extract_point_attributes(
    df_trajectory: pd.DataFrame,
    ind: int,
    traj_trace_frames: int,
    params: TrajectoryParams,
    is_gt: bool = False,
) -> List[PointAttributes]:
    """Extract attributes for pivot, repaired, and manually annotated points."""
    if is_gt:
        return [PointAttributes(False, False, False, True)] * (traj_trace_frames + 1)

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
    ) * params.show_pivot_points

    repaired_points = (
        df_trajectory.loc[ind - traj_trace_frames : ind, "is_repaired"].values
        * params.highlight_repaired_segments
    )

    manual_annotation = (
        df_trajectory.loc[ind - traj_trace_frames : ind, "type"]
        .isin(["manual_annotation"])
        .values
    ) * params.show_pivot_points

    return [
        PointAttributes(is_pivot, is_repaired, is_manual, False)
        for is_pivot, is_repaired, is_manual in zip(
            pivot_points, repaired_points, manual_annotation
        )
    ]


def draw_trajectory_points(
    image: np.ndarray,
    trajectory_points: Tuple[List[TrajectoryPoint], List[TrajectoryPoint]],
    params: TrajectoryParams,
    ball_size: int,
    video_metadata: VideoMetadata,
) -> np.ndarray:
    """Draw trajectory points on the image."""
    arc_points, ground_points = trajectory_points
    num_points = len(arc_points)

    for i, (arc_point, ground_point) in enumerate(zip(arc_points, ground_points)):
        image = draw_single_point(
            image,
            arc_point,
            "arc",
            params,
            ball_size,
            i,
            num_points,
            video_metadata,
        )
        image = draw_single_point(
            image,
            ground_point,
            "ground",
            params,
            ball_size,
            i,
            num_points,
            video_metadata,
        )

    return image


def draw_single_point(
    image: np.ndarray,
    trajectory_point: TrajectoryPoint,
    point_type: str,
    params: TrajectoryParams,
    ball_size: int,
    i: int,
    num_points: int,
    video_metadata: VideoMetadata,
) -> np.ndarray:
    """Draw a single trajectory point on the image."""
    point_color = get_point_color(point_type, trajectory_point.attributes, params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ball_center = tuple(trajectory_point.screen_point.astype(np.int32))

    ball_size_ = (
        int(ball_size * params.pivot_size_multiplier)
        if trajectory_point.attributes.is_pivot_point
        else ball_size
    )
    line_thickness = get_line_thickness(trajectory_point.attributes, params)

    if params.fading_speed > 0:
        bbox = _get_visualization_bbox(
            ball_center, ball_size_, line_thickness, video_metadata
        )
        bbox_copy = image[bbox[0] : bbox[1], bbox[2] : bbox[3]].copy()
        alpha = (1 - (num_points - i - 1) / num_points) ** params.fading_speed

    image = cv.circle(
        image, ball_center, ball_size_, color=point_color, thickness=line_thickness
    )

    if params.fading_speed > 0:
        image[bbox[0] : bbox[1], bbox[2] : bbox[3]] = cv.addWeighted(
            image[bbox[0] : bbox[1], bbox[2] : bbox[3]],
            alpha,
            bbox_copy,
            1 - alpha,
            0,
        )

    return image


def get_point_color(
    point_type: str,
    attributes: PointAttributes,
    params: TrajectoryParams,
) -> Tuple[int, int, int]:
    """Get the color for a trajectory point based on its attributes."""
    if attributes.is_gt:
        return eval(getattr(params, f"{point_type}_gt_color"))
    elif attributes.is_pivot_point and attributes.is_manual_annotation:
        return eval(getattr(params, f"{point_type}_pivot_annotated_color"))
    elif attributes.is_pivot_point:
        return eval(getattr(params, f"{point_type}_pivot_color"))
    elif attributes.is_repaired_point:
        return eval(getattr(params, f"{point_type}_repaired_color"))
    else:
        return eval(getattr(params, f"{point_type}_regular_color"))


def get_line_thickness(
    attributes: PointAttributes,
    params: TrajectoryParams,
) -> int:
    """Get the line thickness for drawing a trajectory point."""
    return (
        params.circle_thickness_repaired
        if attributes.is_pivot_point or attributes.is_repaired_point
        else params.circle_thickness_regular
    )


def add_ball_detection_and_uncertainty(
    image: np.ndarray,
    ind: int,
    df_trajectory: pd.DataFrame,
    params: TrajectoryParams,
    ball_size: int,
) -> np.ndarray:
    """Add ball detection and uncertainty visualization to the image."""
    if ind in df_trajectory.index and not np.isnan(
        np.sum(df_trajectory.loc[ind, ["xk", "yk"]].values)
    ):
        if params.show_ball_detection:
            image = cv.circle(
                image,
                center=tuple(df_trajectory.loc[ind, ["xk", "yk"]].values.astype("int")),
                radius=2 * ball_size,
                color=eval(params.arc_regular_color),
                thickness=2,
            )
        if params.show_ball_uncertainty and ("xvar" in df_trajectory.columns):
            image = cv.circle(
                image,
                center=tuple(df_trajectory.loc[ind, ["xk", "yk"]].values.astype("int")),
                radius=np.round(np.sqrt(df_trajectory.loc[ind, "xvar"])).astype("int"),
                color=eval(params.ground_regular_color),
                thickness=params.circle_thickness_uncertainty,
            )
    return image


def _get_visualization_bbox(
    ball_center: Tuple[int, int],
    ball_size: int,
    line_thickness: int,
    video_metadata: VideoMetadata,
) -> Tuple[int, int, int, int]:
    """Calculate the bounding box for visualization."""
    return (
        max(ball_center[1] - ball_size - line_thickness - 1, 0),
        min(ball_center[1] + ball_size + line_thickness + 2, video_metadata.height),
        max(ball_center[0] - ball_size - line_thickness - 1, 0),
        min(ball_center[0] + ball_size + line_thickness + 2, video_metadata.width),
    )
