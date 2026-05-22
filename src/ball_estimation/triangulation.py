import logging
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import DBSCAN

from ball_estimation.camera import Camera, project_points

_logger = logging.getLogger(__name__)


def get_ball_detections(
    camera_id: int,
    detection_path: str,
    kickoff: int,
    counts_threshold: int = 500,
    score_threshold: float = 0.2,
    detection_point: str = "center",
) -> pd.DataFrame:
    """Load and preprocess ball detections for a single camera.

    Reads a detection feather file, filters to ball category (if mixed),
    removes static detections appearing at the same pixel coordinates across
    many frames, applies a score threshold, and picks the best detection per frame.

    Parameters
    ----------
    camera_id : int
        Camera identifier (e.g. 1-6).
    detection_path : str
        Path to the detection .feather file.
    kickoff : int
        Kickoff frame id to subtract from file_name.
    counts_threshold : int
        Maximum allowed count for a detection at the same (xc, yc) coordinates.
        Detections at coordinates appearing >= this many times are filtered out.
    score_threshold : float
        Minimum detection score.
    detection_point : str
        Which point to extract from the bbox. "center" (default) uses bbox center,
        "bottom_center" uses bottom-middle (xc, y1) — closer to the ball's ground
        contact point.
    """
    detections = pd.read_feather(detection_path)
    detections["file_name"] -= kickoff

    if "category" in detections.columns:
        detections = detections[detections.category == "ball"].copy()

    detections["xc"] = 0.5 * detections["x0"] + 0.5 * detections["x1"]
    if detection_point == "bottom_center":
        # y1 is the bottom edge of the bbox (larger y = lower in image)
        detections["yc"] = detections["y1"]
    else:
        detections["yc"] = 0.5 * detections["y0"] + 0.5 * detections["y1"]

    ball_counts = detections[["xc", "yc"]].value_counts().reset_index()
    # pandas >= 1.1 names the count column "count"; rename for consistency
    if "count" in ball_counts.columns:
        ball_counts = ball_counts.rename(columns={"count": "counts"})
    detections = detections.merge(ball_counts, on=["xc", "yc"], how="left")

    detections = detections[detections["counts"] < counts_threshold]
    detections = detections[detections["score"] > score_threshold]

    detections = detections.sort_values(
        ["file_name", "counts", "score"], ascending=[True, True, False]
    )
    detections = detections.groupby("file_name").first().reset_index()

    detections.rename(
        columns={"xc": f"xc_{camera_id}", "yc": f"yc_{camera_id}"}, inplace=True
    )
    return detections[["file_name", f"xc_{camera_id}", f"yc_{camera_id}"]]


def _get_camera_object_line(
    camera: Camera, object_screen_coordinates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Get ray from camera position through an image point.

    Returns the camera position and the direction vector of the ray
    from the camera center through the image point projected onto z=0 plane.
    """
    line_first_point = camera.get_pos().T

    inverse_homography = camera.to_inverse_homography()
    object_screen_coordinates_scaled = np.array(
        [[object_screen_coordinates[0, 0], object_screen_coordinates[0, 1]]]
    )
    line_second_point = project_points(inverse_homography, object_screen_coordinates_scaled)
    line_second_point = np.concatenate(
        (line_second_point, np.zeros_like(line_second_point[:, :1])), axis=1
    )

    line_direction = line_second_point - line_first_point

    return line_first_point[0], line_direction[0]


def _get_camera_rays_batch(
    camera: Camera, detections_2d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Get rays from camera position through multiple image points.

    Parameters
    ----------
    camera : Camera
        Camera object.
    detections_2d : np.ndarray
        Array of shape (N, 2) with pixel coordinates.

    Returns
    -------
    ray_origin : np.ndarray
        Camera position, shape (3,).
    ray_dirs : np.ndarray
        Ray directions, shape (N, 3).
    """
    cam_pos = camera.get_pos().T[0]  # (3,)
    inv_h = camera.to_inverse_homography()
    projected = project_points(inv_h, detections_2d)  # (N, 2)
    ray_endpoints = np.column_stack([projected, np.zeros(len(projected))])  # (N, 3)
    ray_dirs = ray_endpoints - cam_pos  # (N, 3)
    return cam_pos, ray_dirs


def triangulate_pair(
    camera_1: Camera,
    image_point_1: np.ndarray,
    camera_2: Camera,
    image_point_2: np.ndarray,
    threshold_dm: float = 8,
    reject_behind_camera: bool = False,
    parallel_ray_threshold: float = 0,
) -> np.ndarray:
    """Triangulate a 3D point from two camera observations.

    Finds the closest point between two camera rays. If the minimum distance
    between rays exceeds threshold_dm, returns NaN array.
    """
    r1, e1 = _get_camera_object_line(camera=camera_1, object_screen_coordinates=image_point_1)
    r2, e2 = _get_camera_object_line(camera=camera_2, object_screen_coordinates=image_point_2)
    n = np.cross(e1, e2)
    n_norm = np.linalg.norm(n)

    # Guard against near-parallel rays
    if parallel_ray_threshold > 0 and n_norm < parallel_ray_threshold:
        return np.array([np.nan, np.nan, np.nan])

    if n_norm < 1e-15:
        return np.array([np.nan, np.nan, np.nan])

    d = np.abs(np.dot(n, r1 - r2) / n_norm)
    if d > threshold_dm:
        return np.array([np.nan, np.nan, np.nan])

    nn = np.dot(n, n)
    t1 = np.dot(np.cross(e2, n), (r2 - r1)) / nn
    t2 = np.dot(np.cross(e1, n), (r2 - r1)) / nn

    # Reject rays pointing behind cameras
    if reject_behind_camera and (t1 < 0 or t2 < 0):
        return np.array([np.nan, np.nan, np.nan])

    p1 = r1 + t1 * e1
    p2 = r2 + t2 * e2

    p = 0.5 * p1 + 0.5 * p2
    return p


def _triangulate_all_pairs_sequential(
    df: pd.DataFrame,
    cameras: dict[int, Camera],
    camera_ids: list[int],
    threshold_dm: float = 8,
    reject_behind_camera: bool = False,
    parallel_ray_threshold: float = 0,
) -> pd.DataFrame:
    """Sequential pairwise triangulation (reference implementation)."""
    for cam1 in camera_ids:
        for cam2 in camera_ids:
            if cam2 > cam1:
                _logger.info(f"Triangulating cameras pair {cam1}, {cam2}")
                df[
                    [f"px_{cam1}_{cam2}", f"py_{cam1}_{cam2}", f"pz_{cam1}_{cam2}"]
                ] = df.apply(
                    lambda row, c1=cam1, c2=cam2: triangulate_pair(
                        camera_1=cameras[c1],
                        image_point_1=np.array([[row[f"xc_{c1}"], row[f"yc_{c1}"]]]),
                        camera_2=cameras[c2],
                        image_point_2=np.array([[row[f"xc_{c2}"], row[f"yc_{c2}"]]]),
                        threshold_dm=threshold_dm,
                        reject_behind_camera=reject_behind_camera,
                        parallel_ray_threshold=parallel_ray_threshold,
                    ),
                    axis=1,
                    result_type="expand",
                )
    return df


def triangulate_all_pairs(
    df: pd.DataFrame,
    cameras: dict[int, Camera],
    camera_ids: list[int],
    threshold_dm: float = 8,
    reject_behind_camera: bool = False,
    parallel_ray_threshold: float = 0,
) -> pd.DataFrame:
    """Triangulate ball position from all camera pairs (vectorized).

    For each pair of cameras, triangulates the 3D position per frame using
    the 2D detections. Adds columns px_{i}_{j}, py_{i}_{j}, pz_{i}_{j}.
    """
    for cam1 in camera_ids:
        for cam2 in camera_ids:
            if cam2 <= cam1:
                continue

            _logger.info(f"Triangulating cameras pair {cam1}, {cam2}")

            xc1_col, yc1_col = f"xc_{cam1}", f"yc_{cam1}"
            xc2_col, yc2_col = f"xc_{cam2}", f"yc_{cam2}"
            px_col = f"px_{cam1}_{cam2}"
            py_col = f"py_{cam1}_{cam2}"
            pz_col = f"pz_{cam1}_{cam2}"

            # Initialize output columns with NaN
            n_rows = len(df)
            result = np.full((n_rows, 3), np.nan)

            # Find rows where both cameras have detections
            valid_mask = df[xc1_col].notna() & df[xc2_col].notna()
            valid_idx = np.where(valid_mask.values)[0]

            if len(valid_idx) == 0:
                df[[px_col, py_col, pz_col]] = result
                continue

            # Batch compute ray origins and directions
            dets1 = df.loc[valid_mask, [xc1_col, yc1_col]].values  # (M, 2)
            dets2 = df.loc[valid_mask, [xc2_col, yc2_col]].values  # (M, 2)

            r1, e1_batch = _get_camera_rays_batch(cameras[cam1], dets1)  # (3,), (M, 3)
            r2, e2_batch = _get_camera_rays_batch(cameras[cam2], dets2)  # (3,), (M, 3)

            # Cross product of ray directions: n = e1 x e2
            n_batch = np.cross(e1_batch, e2_batch)  # (M, 3)
            n_norms = np.linalg.norm(n_batch, axis=1)  # (M,)

            # Build a mask for valid triangulations
            ok = np.ones(len(valid_idx), dtype=bool)

            # Guard near-parallel rays
            if parallel_ray_threshold > 0:
                ok &= n_norms >= parallel_ray_threshold
            ok &= n_norms >= 1e-15

            # Distance between rays
            diff = r1 - r2  # broadcast (3,)
            d_batch = np.abs(np.einsum("ij,ij->i", n_batch, np.broadcast_to(diff, n_batch.shape)))
            # Avoid division by zero for rejected rays
            safe_norms = np.where(ok, n_norms, 1.0)
            d_batch = d_batch / safe_norms
            ok &= d_batch <= threshold_dm

            # Compute t1, t2 parameters
            nn = np.where(ok, n_norms ** 2, 1.0)  # avoid div-by-zero
            r2_minus_r1 = r2 - r1  # broadcast (3,)
            r2_minus_r1_batch = np.broadcast_to(r2_minus_r1, e1_batch.shape)  # (M, 3)

            cross_e2_n = np.cross(e2_batch, n_batch)  # (M, 3)
            cross_e1_n = np.cross(e1_batch, n_batch)  # (M, 3)

            t1 = np.einsum("ij,ij->i", cross_e2_n, r2_minus_r1_batch) / nn  # (M,)
            t2 = np.einsum("ij,ij->i", cross_e1_n, r2_minus_r1_batch) / nn  # (M,)

            # Reject rays pointing behind cameras
            if reject_behind_camera:
                ok &= (t1 >= 0) & (t2 >= 0)

            # Compute midpoints
            p1 = r1 + t1[:, None] * e1_batch  # (M, 3)
            p2 = r2 + t2[:, None] * e2_batch  # (M, 3)
            midpoints = 0.5 * p1 + 0.5 * p2  # (M, 3)

            # Apply mask
            midpoints[~ok] = np.nan

            result[valid_idx] = midpoints
            df[[px_col, py_col, pz_col]] = result

    return df


def _get_ball_position(
    row: pd.Series,
    camera_ids: list[int],
    epsilon: float = 20,
    min_samples: int = 3,
) -> list:
    """Estimate ball 3D position from triangulated camera pairs using DBSCAN.

    Collects all valid triangulation results for a single frame, clusters them
    with DBSCAN to find consensus, then computes the mean of the largest cluster.
    """
    points = []
    for cam1 in camera_ids:
        for cam2 in camera_ids:
            if cam2 > cam1:
                if not np.isnan(row[f"px_{cam1}_{cam2}"]):
                    points.append(
                        [
                            row[f"px_{cam1}_{cam2}"],
                            row[f"py_{cam1}_{cam2}"],
                            row[f"pz_{cam1}_{cam2}"],
                        ]
                    )
    triangulations_num = len(points)
    if triangulations_num < min_samples:
        return [triangulations_num, 0, np.nan, np.nan, np.nan]

    db = DBSCAN(eps=epsilon, min_samples=min_samples)
    db.fit(points)

    if np.all(db.labels_ == -1):
        return [triangulations_num, 0, np.nan, np.nan, np.nan]

    unique, counts = np.unique(db.labels_[db.labels_ != -1], return_counts=True)
    most_common_cluster = unique[np.argmax(counts)]
    cluster_size = counts[np.argmax(counts)]

    filtered_points = np.array(
        [points[i] for i in range(len(points)) if db.labels_[i] == most_common_cluster]
    )
    center = np.mean(filtered_points, axis=0)

    return [triangulations_num, cluster_size, center[0], center[1], center[2]]


# ---------------------------------------------------------------------------
# Multi-ray triangulation (V2)
# ---------------------------------------------------------------------------

def _triangulate_multi_ray_ls(
    ray_origins: np.ndarray,
    ray_dirs: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted least-squares multi-ray triangulation.

    Solves X* = argmin_X sum_i w_i ||(I - d_i d_i^T)(X - C_i)||^2
    which reduces to a 3x3 linear system A X = b.

    Parameters
    ----------
    ray_origins : np.ndarray
        Shape (N, 3), camera positions.
    ray_dirs : np.ndarray
        Shape (N, 3), ray direction vectors (need not be unit).
    weights : np.ndarray or None
        Shape (N,), per-ray weights. Default: uniform.

    Returns
    -------
    point_3d : np.ndarray
        Shape (3,). Returns NaN if the system is singular.
    """
    N = len(ray_origins)
    if N < 2:
        return np.full(3, np.nan)

    # Normalize directions
    norms = np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    d = ray_dirs / norms  # (N, 3)

    if weights is None:
        weights = np.ones(N)

    I = np.eye(3)
    A = np.zeros((3, 3))
    b = np.zeros(3)

    for i in range(N):
        di = d[i]
        P = I - np.outer(di, di)
        wP = weights[i] * P
        A += wP
        b += wP @ ray_origins[i]

    try:
        X = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.full(3, np.nan)

    return X


def _point_to_ray_distances_batch(
    point: np.ndarray,
    ray_origins: np.ndarray,
    ray_dirs: np.ndarray,
) -> np.ndarray:
    """Compute perpendicular distances from a 3D point to multiple rays.

    Parameters
    ----------
    point : np.ndarray
        Shape (3,).
    ray_origins : np.ndarray
        Shape (N, 3).
    ray_dirs : np.ndarray
        Shape (N, 3).

    Returns
    -------
    distances : np.ndarray
        Shape (N,).
    """
    diffs = point - ray_origins  # (N, 3)
    crosses = np.cross(diffs, ray_dirs)  # (N, 3)
    cross_norms = np.linalg.norm(crosses, axis=1)
    dir_norms = np.linalg.norm(ray_dirs, axis=1)
    dir_norms = np.where(dir_norms < 1e-15, 1.0, dir_norms)
    return cross_norms / dir_norms


def _triangulate_ransac(
    ray_origins: np.ndarray,
    ray_dirs: np.ndarray,
    n_iterations: int = 20,
    inlier_threshold_dm: float = 8,
    min_inliers: int = 2,
) -> tuple[np.ndarray, int, np.ndarray]:
    """RANSAC-based multi-ray triangulation.

    Samples pairs of rays, triangulates a hypothesis, evaluates inliers
    via point-to-ray distance, and refits on best inlier set using
    multi-ray least squares.

    Returns
    -------
    point_3d : np.ndarray
        Shape (3,). Best estimate or NaN.
    n_inliers : int
        Number of inlier cameras for the best hypothesis.
    inlier_mask : np.ndarray
        Boolean array of shape (N,) indicating which rays are inliers.
    """
    n_rays = len(ray_origins)
    nan_point = np.full(3, np.nan)
    empty_inlier_mask = np.zeros(n_rays, dtype=bool)
    if n_rays < min_inliers:
        return nan_point, 0, empty_inlier_mask

    # If only 2 rays, just do direct least-squares
    if n_rays == 2:
        pt = _triangulate_multi_ray_ls(ray_origins, ray_dirs)
        if np.any(np.isnan(pt)):
            return nan_point, 0, empty_inlier_mask
        dists = _point_to_ray_distances_batch(pt, ray_origins, ray_dirs)
        inlier_mask = dists < inlier_threshold_dm
        n_inliers = int(np.sum(inlier_mask))
        if n_inliers >= min_inliers:
            return pt, n_inliers, inlier_mask
        return nan_point, 0, empty_inlier_mask

    rng = np.random.RandomState(42)
    best_point = nan_point
    best_n_inliers = 0
    best_total_dist = np.inf
    best_inlier_mask = empty_inlier_mask

    for _ in range(n_iterations):
        # Sample 2 distinct rays
        idx = rng.choice(n_rays, size=2, replace=False)
        hypothesis = _triangulate_multi_ray_ls(ray_origins[idx], ray_dirs[idx])
        if np.any(np.isnan(hypothesis)):
            continue

        # Evaluate on all rays
        dists = _point_to_ray_distances_batch(hypothesis, ray_origins, ray_dirs)
        inlier_mask = dists < inlier_threshold_dm
        n_inliers = int(np.sum(inlier_mask))
        total_dist = float(np.sum(dists[inlier_mask])) if n_inliers > 0 else np.inf

        if (
            n_inliers > best_n_inliers
            or (n_inliers == best_n_inliers and total_dist < best_total_dist)
        ):
            best_n_inliers = n_inliers
            best_total_dist = total_dist
            best_point = hypothesis
            best_inlier_mask = inlier_mask

    if best_n_inliers < min_inliers:
        return nan_point, 0, empty_inlier_mask

    # Refit on inliers using multi-ray LS
    refit = _triangulate_multi_ray_ls(
        ray_origins[best_inlier_mask], ray_dirs[best_inlier_mask]
    )
    if np.any(np.isnan(refit)):
        return best_point, best_n_inliers, best_inlier_mask

    return refit, best_n_inliers, best_inlier_mask


def _batch_reprojection_errors(
    points_3d: np.ndarray,
    camera: Camera,
    detections_2d: np.ndarray,
) -> np.ndarray:
    """Compute reprojection errors for batch of 3D points against 2D detections.

    Parameters
    ----------
    points_3d : np.ndarray
        Shape (N, 3).
    camera : Camera
        Camera to project through.
    detections_2d : np.ndarray
        Shape (N, 2), observed pixel coordinates.

    Returns
    -------
    errors : np.ndarray
        Shape (N,), pixel distances.
    """
    P = camera.to_projection_matrix()  # (3, 4)
    projected = project_points(P, points_3d)  # (N, 2)
    return np.linalg.norm(projected - detections_2d, axis=1)


def compute_ball_positions_multi_ray(
    df: pd.DataFrame,
    cameras: dict[int, Camera],
    camera_ids: list[int],
    ransac_iterations: int = 20,
    ransac_inlier_threshold_dm: float = 8,
    min_cameras: int = 2,
    reject_behind_camera: bool = True,
    parallel_ray_threshold: float = 1e-8,
) -> pd.DataFrame:
    """Compute 3D ball positions using multi-ray RANSAC triangulation.

    For each frame, collects all camera rays with detections, runs RANSAC
    to find the best multi-ray solution, and records inlier count.

    Adds columns: triangulations_num, cluster_size, cx, cy, cz,
    and ``_ransac_inlier_{cam_id}`` per camera.
    """
    _logger.info("Computing ball positions via multi-ray RANSAC")

    # Pre-compute camera ray info
    cam_ray_info = {}
    for cam_id in camera_ids:
        cam_pos = cameras[cam_id].get_pos().T[0]  # (3,)
        inv_h = cameras[cam_id].to_inverse_homography()
        cam_ray_info[cam_id] = (cam_pos, inv_h)

    n_rows = len(df)
    tri_nums = np.zeros(n_rows, dtype=int)
    cluster_sizes = np.zeros(n_rows, dtype=int)
    positions = np.full((n_rows, 3), np.nan)
    inlier_flags = {cam_id: np.zeros(n_rows, dtype=bool) for cam_id in camera_ids}

    for i in range(n_rows):
        row = df.iloc[i]
        ray_origins = []
        ray_dirs = []
        frame_cam_ids = []

        for cam_id in camera_ids:
            xc = row.get(f"xc_{cam_id}")
            yc = row.get(f"yc_{cam_id}")
            if pd.isna(xc) or pd.isna(yc):
                continue

            cam_pos, inv_h = cam_ray_info[cam_id]
            det_2d = np.array([[xc, yc]])
            projected = project_points(inv_h, det_2d)  # (1, 2)
            endpoint = np.array([projected[0, 0], projected[0, 1], 0.0])
            ray_dir = endpoint - cam_pos

            # Optional: skip near-degenerate rays
            dir_norm = np.linalg.norm(ray_dir)
            if dir_norm < 1e-15:
                continue

            ray_origins.append(cam_pos)
            ray_dirs.append(ray_dir)
            frame_cam_ids.append(cam_id)

        n_cams = len(ray_origins)
        tri_nums[i] = n_cams

        if n_cams < min_cameras:
            continue

        origins = np.array(ray_origins)
        dirs = np.array(ray_dirs)

        pt, n_inliers, inlier_mask = _triangulate_ransac(
            origins, dirs,
            n_iterations=ransac_iterations,
            inlier_threshold_dm=ransac_inlier_threshold_dm,
            min_inliers=min_cameras,
        )

        if not np.any(np.isnan(pt)):
            cluster_sizes[i] = n_inliers
            positions[i] = pt
            for cam_id, is_inlier in zip(frame_cam_ids, inlier_mask):
                if is_inlier:
                    inlier_flags[cam_id][i] = True

    df["triangulations_num"] = tri_nums
    df["cluster_size"] = cluster_sizes
    df["cx"] = positions[:, 0]
    df["cy"] = positions[:, 1]
    df["cz"] = positions[:, 2]
    for cam_id in camera_ids:
        df[f"_ransac_inlier_{cam_id}"] = inlier_flags[cam_id]

    return df


def _apply_reprojection_filter(
    df: pd.DataFrame,
    cameras: dict[int, Camera],
    camera_ids: list[int],
    threshold_px: float = 15,
    min_cameras: int = 2,
) -> pd.DataFrame:
    """Filter 3D estimates by reprojection consistency.

    For each frame with a 3D estimate, projects back into each camera
    and checks if reprojection error is below threshold. Requires at least
    min_cameras passing the check.
    """
    _logger.info(f"Applying reprojection filter (threshold={threshold_px}px, min_cameras={min_cameras})")

    valid_mask = df["cx"].notna()
    valid_idx = df.index[valid_mask]

    if len(valid_idx) == 0:
        return df

    points_3d = df.loc[valid_mask, ["cx", "cy", "cz"]].values  # (M, 3)
    for cam_id in camera_ids:
        xc_col = f"xc_{cam_id}"
        yc_col = f"yc_{cam_id}"

        # Only process frames with detections for this camera
        has_det = df.loc[valid_mask, xc_col].notna().values

        if not np.any(has_det):
            continue

        # Compute reprojection errors for frames with detections
        det_points_3d = points_3d[has_det]
        det_2d = df.loc[valid_idx[has_det], [xc_col, yc_col]].values

        errors = _batch_reprojection_errors(det_points_3d, cameras[cam_id], det_2d)

        # Store per-camera pass/fail (kept for save_filtered_detections)
        col_name = f"_reproj_pass_{cam_id}"
        pass_arr = np.zeros(len(valid_idx), dtype=bool)
        pass_arr[has_det] = errors < threshold_px
        df.loc[valid_idx, col_name] = pass_arr

    # Count cameras passing for each frame
    pass_cols = [f"_reproj_pass_{cam_id}" for cam_id in camera_ids]
    existing_cols = [c for c in pass_cols if c in df.columns]
    if existing_cols:
        pass_count = df.loc[valid_idx, existing_cols].sum(axis=1)
        reject = pass_count < min_cameras
        reject_idx = valid_idx[reject.values]
        df.loc[reject_idx, ["cx", "cy", "cz"]] = np.nan

        n_rejected = int(reject.sum())
        _logger.info(f"Reprojection filter rejected {n_rejected}/{len(valid_idx)} frames")

    return df


def _fill_temporal_gaps(
    df: pd.DataFrame,
    max_gap_frames: int = 3,
) -> pd.DataFrame:
    """Fill short gaps in 3D trajectory via linear interpolation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain cx, cy, cz columns.
    max_gap_frames : int
        Maximum number of consecutive NaN frames to interpolate.
    """
    for col in ["cx", "cy", "cz"]:
        df[col] = df[col].interpolate(
            method="linear", limit=max_gap_frames, limit_direction="both"
        )
    return df


def compute_ball_3d_gt(
    detections_df: pd.DataFrame,
    cameras: dict[int, Camera],
    camera_ids: list[int],
    threshold_dm: float = 8,
    dbscan_epsilon: float = 20,
    dbscan_min_samples: int = 3,
    z_upper_bound: float | None = None,
    z_clamp_threshold: float | None = None,
    # V2 parameters
    reject_behind_camera: bool = False,
    parallel_ray_threshold: float = 0,
    method: str = "pairwise_dbscan",
    ransac_iterations: int = 20,
    ransac_inlier_threshold_dm: float = 8,
    min_cameras: int = 2,
    reprojection_filter: bool = False,
    reprojection_threshold_px: float = 15,
    reprojection_min_cameras: int = 2,
    temporal_gap_fill: bool = False,
    temporal_gap_max_frames: int = 3,
    z_compensate_radius: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute ground truth 3D ball positions from multi-camera detections.

    Full pipeline: triangulate -> cluster/RANSAC -> post-process -> format output.

    Parameters
    ----------
    method : str
        "pairwise_dbscan" (default, original) or "multi_ray_ransac" (V2).
    z_compensate_radius : bool
        If True (default), subtract 1 dm from z to convert from ball center to
        ball bottom. Set to False when using bottom_center detection point.

    Returns
    -------
    df_gt : pd.DataFrame
        Formatted output with columns file_name, track_id, x, y, z, velocities, accelerations.
    df_intermediate : pd.DataFrame
        Intermediate data with raw cx/cy/cz and per-camera detections (for error analysis).
    """
    # Auto-derive z thresholds from detection mode if not explicitly set.
    # center mode: ball center ~1dm above ground when on pitch → clamp to -1
    # bottom_center mode: ball bottom at ground level → clamp to 0
    radius_offset = 1 if z_compensate_radius else 0
    if z_clamp_threshold is None:
        z_clamp_threshold = -radius_offset
    if z_upper_bound is None:
        z_upper_bound = 3  # generous garbage filter, same for both modes

    df = detections_df.copy()

    if method == "pairwise_dbscan":
        df = triangulate_all_pairs(
            df, cameras, camera_ids, threshold_dm,
            reject_behind_camera=reject_behind_camera,
            parallel_ray_threshold=parallel_ray_threshold,
        )

        _logger.info("Computing ball positions via DBSCAN clustering")
        df[["triangulations_num", "cluster_size", "cx", "cy", "cz"]] = df.apply(
            lambda row: _get_ball_position(
                row, camera_ids, epsilon=dbscan_epsilon, min_samples=dbscan_min_samples
            ),
            axis=1,
            result_type="expand",
        )
        df["triangulations_num"] = df["triangulations_num"].astype(int)
        df["cluster_size"] = df["cluster_size"].astype(int)

    elif method == "multi_ray_ransac":
        df = compute_ball_positions_multi_ray(
            df, cameras, camera_ids,
            ransac_iterations=ransac_iterations,
            ransac_inlier_threshold_dm=ransac_inlier_threshold_dm,
            min_cameras=min_cameras,
            reject_behind_camera=reject_behind_camera,
            parallel_ray_threshold=parallel_ray_threshold,
        )
    else:
        raise ValueError(f"Unknown triangulation method: {method}")

    # Reprojection validation filter
    if reprojection_filter:
        df = _apply_reprojection_filter(
            df, cameras, camera_ids,
            threshold_px=reprojection_threshold_px,
            min_cameras=reprojection_min_cameras,
        )

    # Post-processing: filter invalid z, clamp near-ground z
    df.loc[df.cz > z_upper_bound, ["cx", "cy", "cz"]] = np.nan
    df.loc[df.cz > z_clamp_threshold, "cz"] = z_clamp_threshold

    # Temporal gap filling
    if temporal_gap_fill:
        _logger.info(f"Filling temporal gaps (max {temporal_gap_max_frames} frames)")
        df = _fill_temporal_gaps(df, max_gap_frames=temporal_gap_max_frames)

    # Keep intermediate data before formatting
    df_intermediate = df.copy()

    # Format output DataFrame
    df_gt = (
        pd.DataFrame(
            {"file_name": np.arange(df["file_name"].min(), df["file_name"].max() + 1)}
        )
        .merge(df[["file_name", "cx", "cy", "cz"]], on=["file_name"], how="left")
        .sort_values(["file_name"])
        .rename(columns={"cx": "x", "cy": "y", "cz": "z"})
        .assign(track_id=0)
        .reindex(columns=["file_name", "track_id", "x", "y", "z"])
        .assign(
            x_velocity=0,
            y_velocity=0,
            z_velocity=0,
            x_acceleration=0,
            y_acceleration=0,
            z_acceleration=0,
        )
        .reset_index(drop=True)
    )

    df_gt["x"] = np.round(df_gt["x"], 1)
    df_gt["y"] = np.round(df_gt["y"], 1)
    df_gt["z"] = -np.round(df_gt["z"], 1) - (1 if z_compensate_radius else 0)

    return df_gt, df_intermediate


def _point_to_ray_distance(point: np.ndarray, ray_origin: np.ndarray, ray_dir: np.ndarray) -> float:
    """Compute perpendicular distance from a 3D point to a ray."""
    diff = point - ray_origin
    cross = np.cross(diff, ray_dir)
    return np.linalg.norm(cross) / np.linalg.norm(ray_dir)


def compute_per_camera_errors(
    df: pd.DataFrame,
    cameras: dict[int, Camera],
    camera_ids: list[int],
    return_raw: bool = False,
) -> dict[int, dict] | tuple[dict[int, dict], dict[int, np.ndarray]]:
    """Compute per-camera ray-to-point error for frames with valid 3D estimates.

    For each camera that has a 2D detection in a frame, computes the distance
    from the camera ray (through that detection) to the estimated 3D point.

    Parameters
    ----------
    return_raw : bool
        If True, also returns raw distance arrays per camera.

    Returns
    -------
    errors : dict
        Mapping from camera_id to {"median_dm": float, "mean_dm": float, "n_frames": int,
        "p90_dm": float, "p95_dm": float}.
    raw_distances : dict (only if return_raw=True)
        Mapping from camera_id to np.ndarray of distances.
    """
    valid = df[df["cx"].notna()].copy()
    if valid.empty:
        empty = {cam_id: {"median_dm": None, "mean_dm": None, "p90_dm": None, "p95_dm": None, "n_frames": 0}
                 for cam_id in camera_ids}
        if return_raw:
            return empty, {cam_id: np.array([]) for cam_id in camera_ids}
        return empty

    errors = {}
    raw_distances = {}

    for cam_id in camera_ids:
        xc_col = f"xc_{cam_id}"
        yc_col = f"yc_{cam_id}"

        has_det = valid[xc_col].notna()
        cam_valid = valid[has_det]
        if cam_valid.empty:
            errors[cam_id] = {"median_dm": None, "mean_dm": None, "p90_dm": None, "p95_dm": None, "n_frames": 0}
            raw_distances[cam_id] = np.array([])
            continue

        cam_points_3d = cam_valid[["cx", "cy", "cz"]].values
        cam_detections = cam_valid[[xc_col, yc_col]].values

        camera = cameras[cam_id]
        inv_h = camera.to_inverse_homography()
        cam_pos = camera.get_pos().T[0]  # shape (3,)

        # Vectorized: compute ray directions for all detections at once
        projected = project_points(inv_h, cam_detections)  # shape (N, 2)
        ray_endpoints = np.column_stack([projected, np.zeros(len(projected))])  # z=0
        ray_dirs = ray_endpoints - cam_pos  # shape (N, 3)

        # Vectorized point-to-ray distance
        diffs = cam_points_3d - cam_pos  # shape (N, 3)
        crosses = np.cross(diffs, ray_dirs)  # shape (N, 3)
        cross_norms = np.linalg.norm(crosses, axis=1)
        dir_norms = np.linalg.norm(ray_dirs, axis=1)
        distances = cross_norms / dir_norms

        errors[cam_id] = {
            "median_dm": float(np.median(distances)),
            "mean_dm": float(np.mean(distances)),
            "p90_dm": float(np.percentile(distances, 90)),
            "p95_dm": float(np.percentile(distances, 95)),
            "n_frames": int(len(distances)),
        }
        raw_distances[cam_id] = distances

    if return_raw:
        return errors, raw_distances
    return errors


# ---------------------------------------------------------------------------
# Evaluation metrics (LOCO, frame-median, temporal jitter)
# ---------------------------------------------------------------------------

def compute_loco_errors(
    df: pd.DataFrame,
    cameras: dict[int, Camera],
    camera_ids: list[int],
    ransac_iterations: int = 20,
    ransac_inlier_threshold_dm: float = 8,
    min_cameras_loco: int = 2,
) -> dict:
    """Leave-One-Camera-Out cross-validation errors.

    For each frame with a valid 3D estimate and enough cameras, holds out
    each camera in turn, re-triangulates from the remaining cameras, and
    measures prediction error on the held-out camera in both world space
    (ray distance, dm) and image space (reprojection error, px).

    Parameters
    ----------
    min_cameras_loco : int
        Minimum cameras required for the leave-out re-triangulation (default 2).
        A frame needs >= min_cameras_loco + 1 detections to be LOCO-eligible.

    Returns
    -------
    dict with keys:
        "per_camera": {cam_id: {"ray_dist_dm": [...], "reproj_px": [...]}},
        "summary": {
            "ray_dist_median_dm", "ray_dist_p90_dm", "ray_dist_p95_dm",
            "reproj_median_px", "reproj_p90_px", "reproj_p95_px",
            "n_evaluations", "n_eligible_frames", "n_total_valid_frames"
        }
    """
    valid = df[df["cx"].notna()].copy()
    min_detections = min_cameras_loco + 1

    # Pre-compute per-camera ray info
    cam_info = {}
    for cam_id in camera_ids:
        cam_pos = cameras[cam_id].get_pos().T[0]
        inv_h = cameras[cam_id].to_inverse_homography()
        proj = cameras[cam_id].to_projection_matrix()
        cam_info[cam_id] = (cam_pos, inv_h, proj)

    per_camera_ray = {c: [] for c in camera_ids}
    per_camera_reproj = {c: [] for c in camera_ids}
    n_eligible = 0

    for i in range(len(valid)):
        row = valid.iloc[i]

        # Collect cameras with detections for this frame
        available_cams = []
        for cam_id in camera_ids:
            xc = row.get(f"xc_{cam_id}")
            if pd.notna(xc):
                available_cams.append(cam_id)

        if len(available_cams) < min_detections:
            continue

        n_eligible += 1

        # Pre-compute all rays for this frame
        frame_rays = {}
        for cam_id in available_cams:
            cam_pos, inv_h, _ = cam_info[cam_id]
            det_2d = np.array([[row[f"xc_{cam_id}"], row[f"yc_{cam_id}"]]])
            projected = project_points(inv_h, det_2d)
            endpoint = np.array([projected[0, 0], projected[0, 1], 0.0])
            ray_dir = endpoint - cam_pos
            frame_rays[cam_id] = (cam_pos, ray_dir)

        # Leave each camera out in turn
        for holdout_cam in available_cams:
            other_cams = [c for c in available_cams if c != holdout_cam]
            if len(other_cams) < min_cameras_loco:
                continue

            origins = np.array([frame_rays[c][0] for c in other_cams])
            dirs = np.array([frame_rays[c][1] for c in other_cams])

            pt, n_inl, _ = _triangulate_ransac(
                origins, dirs,
                n_iterations=ransac_iterations,
                inlier_threshold_dm=ransac_inlier_threshold_dm,
                min_inliers=min_cameras_loco,
            )

            if np.any(np.isnan(pt)):
                continue

            # Ray distance: held-out camera ray to re-triangulated point
            ho_pos, ho_dir = frame_rays[holdout_cam]
            ray_dist = float(_point_to_ray_distances_batch(
                pt, np.array([ho_pos]), np.array([ho_dir])
            )[0])
            per_camera_ray[holdout_cam].append(ray_dist)

            # Reprojection error: project re-triangulated point into held-out camera
            _, _, ho_proj = cam_info[holdout_cam]
            projected_2d = project_points(ho_proj, pt.reshape(1, 3))  # (1, 2)
            det_2d = np.array([row[f"xc_{holdout_cam}"], row[f"yc_{holdout_cam}"]])
            reproj_err = float(np.linalg.norm(projected_2d[0] - det_2d))
            per_camera_reproj[holdout_cam].append(reproj_err)

    # Aggregate
    all_ray = np.concatenate([np.array(v) for v in per_camera_ray.values() if len(v) > 0])
    all_reproj = np.concatenate([np.array(v) for v in per_camera_reproj.values() if len(v) > 0])

    summary = {
        "n_evaluations": int(len(all_ray)),
        "n_eligible_frames": n_eligible,
        "n_total_valid_frames": len(valid),
    }

    if len(all_ray) > 0:
        summary.update({
            "ray_dist_median_dm": float(np.median(all_ray)),
            "ray_dist_mean_dm": float(np.mean(all_ray)),
            "ray_dist_p90_dm": float(np.percentile(all_ray, 90)),
            "ray_dist_p95_dm": float(np.percentile(all_ray, 95)),
            "reproj_median_px": float(np.median(all_reproj)),
            "reproj_mean_px": float(np.mean(all_reproj)),
            "reproj_p90_px": float(np.percentile(all_reproj, 90)),
            "reproj_p95_px": float(np.percentile(all_reproj, 95)),
        })

    per_camera_summary = {}
    for cam_id in camera_ids:
        rd = np.array(per_camera_ray[cam_id])
        rp = np.array(per_camera_reproj[cam_id])
        if len(rd) > 0:
            per_camera_summary[cam_id] = {
                "ray_dist_median_dm": float(np.median(rd)),
                "reproj_median_px": float(np.median(rp)),
                "n": int(len(rd)),
            }
        else:
            per_camera_summary[cam_id] = {"ray_dist_median_dm": None, "reproj_median_px": None, "n": 0}

    return {"summary": summary, "per_camera": per_camera_summary}


def compute_frame_median_errors(
    df: pd.DataFrame,
    cameras: dict[int, Camera],
    camera_ids: list[int],
) -> dict:
    """Compute per-frame median ray-distance, then aggregate over frames.

    Instead of pooling all per-camera distances (dominated by outlier
    detections), computes the median distance across cameras for each frame
    first, producing a robust per-frame error. Then reports statistics
    over these frame-level medians.
    """
    valid = df[df["cx"].notna()].copy()
    if valid.empty:
        return {"median_dm": None, "mean_dm": None, "p90_dm": None, "p95_dm": None, "n_frames": 0}

    points_3d = valid[["cx", "cy", "cz"]].values

    # Pre-compute per-camera ray info
    cam_ray_info = {}
    for cam_id in camera_ids:
        cam_pos = cameras[cam_id].get_pos().T[0]
        inv_h = cameras[cam_id].to_inverse_homography()
        cam_ray_info[cam_id] = (cam_pos, inv_h)

    # For each frame, collect ray distances across cameras, take median
    frame_medians = []
    for i in range(len(valid)):
        row = valid.iloc[i]
        pt = points_3d[i]
        dists = []
        for cam_id in camera_ids:
            xc = row.get(f"xc_{cam_id}")
            if pd.isna(xc):
                continue
            cam_pos, inv_h = cam_ray_info[cam_id]
            det_2d = np.array([[row[f"xc_{cam_id}"], row[f"yc_{cam_id}"]]])
            projected = project_points(inv_h, det_2d)
            endpoint = np.array([projected[0, 0], projected[0, 1], 0.0])
            ray_dir = endpoint - cam_pos

            diff = pt - cam_pos
            cross = np.cross(diff, ray_dir)
            d = np.linalg.norm(cross) / np.linalg.norm(ray_dir)
            dists.append(d)

        if dists:
            frame_medians.append(float(np.median(dists)))

    frame_medians = np.array(frame_medians)
    return {
        "median_dm": float(np.median(frame_medians)),
        "mean_dm": float(np.mean(frame_medians)),
        "p90_dm": float(np.percentile(frame_medians, 90)),
        "p95_dm": float(np.percentile(frame_medians, 95)),
        "n_frames": int(len(frame_medians)),
    }


def compute_temporal_jitter(
    df_gt: pd.DataFrame,
    fps: int = 25,
    savgol_window: int = 11,
    savgol_polyorder: int = 2,
) -> dict:
    """Compute temporal jitter and physical plausibility metrics.

    Measures deviation of the raw trajectory from a Savitzky-Golay smoothed
    version, plus velocity/acceleration violation rates.

    Parameters
    ----------
    df_gt : pd.DataFrame
        Output GT dataframe with columns file_name, x, y, z.
    fps : int
        Frame rate.
    savgol_window : int
        Savitzky-Golay filter window (must be odd).
    savgol_polyorder : int
        Polynomial order for Savitzky-Golay.
    """
    from scipy.signal import savgol_filter

    df = df_gt[["file_name", "x", "y", "z"]].copy()

    # Find contiguous segments of valid positions
    valid_mask = df["x"].notna().values
    segments = []
    start = None
    for i in range(len(valid_mask)):
        if valid_mask[i] and start is None:
            start = i
        elif not valid_mask[i] and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(valid_mask)))

    # Jitter: deviation from smoothed trajectory
    all_jitter = []
    all_speeds = []
    all_accels = []

    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        if seg_len < max(savgol_window, 3):
            continue

        x = df["x"].values[seg_start:seg_end]
        y = df["y"].values[seg_start:seg_end]
        z = df["z"].values[seg_start:seg_end]

        # Savitzky-Golay smoothing
        win = min(savgol_window, seg_len if seg_len % 2 == 1 else seg_len - 1)
        if win < savgol_polyorder + 1:
            continue
        x_smooth = savgol_filter(x, win, savgol_polyorder)
        y_smooth = savgol_filter(y, win, savgol_polyorder)
        z_smooth = savgol_filter(z, win, savgol_polyorder)

        jitter = np.sqrt((x - x_smooth)**2 + (y - y_smooth)**2 + (z - z_smooth)**2)
        all_jitter.extend(jitter.tolist())

        # Velocity (dm/frame -> dm/s)
        vx = np.diff(x) * fps
        vy = np.diff(y) * fps
        vz = np.diff(z) * fps
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        all_speeds.extend(speed.tolist())

        # Acceleration (dm/s^2)
        if len(speed) >= 2:
            ax = np.diff(vx) * fps
            ay = np.diff(vy) * fps
            az = np.diff(vz) * fps
            accel = np.sqrt(ax**2 + ay**2 + az**2)
            all_accels.extend(accel.tolist())

    result = {"n_segments": len(segments)}

    if all_jitter:
        jitter_arr = np.array(all_jitter)
        result["jitter_median_dm"] = float(np.median(jitter_arr))
        result["jitter_mean_dm"] = float(np.mean(jitter_arr))
        result["jitter_p90_dm"] = float(np.percentile(jitter_arr, 90))

    if all_speeds:
        speed_arr = np.array(all_speeds)
        result["speed_median_dm_s"] = float(np.median(speed_arr))
        result["speed_p95_dm_s"] = float(np.percentile(speed_arr, 95))
        result["speed_violation_rate"] = float(np.mean(speed_arr > 500))

    if all_accels:
        accel_arr = np.array(all_accels)
        result["accel_median_dm_s2"] = float(np.median(accel_arr))
        result["accel_p95_dm_s2"] = float(np.percentile(accel_arr, 95))
        result["accel_violation_rate"] = float(np.mean(accel_arr > 2000))

    return result


def save_filtered_detections(
    df_intermediate: pd.DataFrame,
    camera_ids: list[int],
    match_root: str,
    half: int,
    detection_name: str,
    output_suffix: str = "dl_filtered",
    use_reprojection: bool = True,
) -> dict[int, dict]:
    """Save per-camera ball detection files filtered to triangulation inliers.

    For each camera, keeps only frames where the camera was a RANSAC inlier
    and the 3D estimate is valid (optionally also passing reprojection filter).

    Parameters
    ----------
    df_intermediate : pd.DataFrame
        Intermediate triangulation DataFrame with ``_ransac_inlier_{cam_id}``
        and optionally ``_reproj_pass_{cam_id}`` columns.
    camera_ids : list[int]
        Camera identifiers.
    match_root : str
        Root directory of the match (e.g. ``data/match_name``).
    half : int
        Half number (1 or 2).
    detection_name : str
        Base detection file name (e.g. ``ball_detection.dl``).
    output_suffix : str
        Version suffix for the output file (e.g. ``dl_filtered``).
    use_reprojection : bool
        If True and reprojection columns exist, also require reprojection pass.

    Returns
    -------
    dict[int, dict]
        Per-camera stats with original/filtered counts and retention rate.
    """
    per_camera_stats = {}
    valid_3d = df_intermediate["cx"].notna()

    for cam_id in camera_ids:
        ransac_col = f"_ransac_inlier_{cam_id}"
        if ransac_col not in df_intermediate.columns:
            _logger.warning(f"No RANSAC inlier column for camera {cam_id}, skipping")
            continue

        inlier_mask = df_intermediate[ransac_col] & valid_3d
        reproj_col = f"_reproj_pass_{cam_id}"
        if use_reprojection and reproj_col in df_intermediate.columns:
            inlier_mask &= df_intermediate[reproj_col]

        relative_inlier_frames = set(df_intermediate.loc[inlier_mask, "file_name"].values)

        # Convert relative (kickoff-subtracted) frame numbers to absolute
        cam_half_dir = os.path.join(match_root, f"camera0{cam_id}", f"half_{half}")
        kickoff_path = os.path.join(cam_half_dir, "kickoff.yaml")
        with open(kickoff_path, "r") as f:
            kickoff = yaml.safe_load(f)["frame_id"]
        absolute_inlier_frames = {int(fr + kickoff) for fr in relative_inlier_frames}

        detection_dir = os.path.join(cam_half_dir, "detection")
        original_path = os.path.join(detection_dir, f"{detection_name}.feather")
        if not os.path.isfile(original_path):
            _logger.warning(f"Original detection not found: {original_path}, skipping")
            continue

        original_df = pd.read_feather(original_path)
        filtered_df = original_df[original_df["file_name"].isin(absolute_inlier_frames)]

        output_path = os.path.join(
            detection_dir, f"ball_detection.{output_suffix}.feather"
        )
        filtered_df.reset_index(drop=True).to_feather(output_path)

        n_orig = len(original_df)
        n_filt = len(filtered_df)
        retention = round(n_filt / n_orig, 4) if n_orig > 0 else 0
        per_camera_stats[cam_id] = {
            "original": n_orig,
            "filtered": n_filt,
            "retention": retention,
        }
        _logger.info(
            f"Camera {cam_id}: {n_filt}/{n_orig} detections retained "
            f"({retention:.1%}) -> {output_path}"
        )

    return per_camera_stats
