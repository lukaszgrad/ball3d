"""Calculate arc/straight split from JSON evaluation results."""
import json
import logging
import math
import os

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

ARC_LONG_MIN_FRAMES = 25  # 1.0 second at 25fps


def _load_eval_json(json_path: str) -> list[dict]:
    """Load ball3d_eval_test_results JSON, return segments sorted by start_frame."""
    with open(json_path) as f:
        data = json.load(f)
    segments = sorted(data.values(), key=lambda s: s["start_frame"])
    return segments


def _build_frame_to_segment(segments: list[dict]) -> dict[int, dict]:
    """Map each frame to its JSON segment.

    For pivot-frame overlaps (adjacent segments share a boundary frame),
    the segment that starts at that frame takes priority.
    """
    frame_to_seg: dict[int, dict] = {}
    for seg in segments:
        for f in range(seg["start_frame"], seg["end_frame"]):
            if f not in frame_to_seg:
                frame_to_seg[f] = seg
        # Force: this segment owns its start frame
        frame_to_seg[seg["start_frame"]] = seg
    return frame_to_seg


def _load_gt_data(
    coords_path: str, pivot_path: str
) -> tuple[dict[int, float | None], set[int], list[int]]:
    """Load GT z-coordinates, frame set, and pivot frames.

    Returns
    -------
    gt_z : dict mapping frame_index -> z value (None if NaN)
    gt_frames : set of all GT frame indices
    pivot_frames : sorted list of frame indices where pivot_point == 1
    """
    df = pd.read_csv(coords_path)
    gt_z: dict[int, float | None] = {}
    for _, row in df.iterrows():
        z = row["z"]
        gt_z[int(row["file_name"])] = None if (isinstance(z, float) and math.isnan(z)) else float(z)
    gt_frames = set(gt_z.keys())

    df_pp = pd.read_csv(pivot_path)
    pivot_frames = sorted(
        df_pp.loc[df_pp["pivot_point"] == 1, "file_name"].astype(int).tolist()
    )
    return gt_z, gt_frames, pivot_frames


def _derive_z_threshold(
    segments: list[dict],
    gt_z: dict[int, float | None],
    default_threshold: float = 2.0,
) -> float:
    """Derive optimal z-height threshold separating arc from straight.

    For each JSON segment, computes mean GT z over its frames, then scans
    candidate thresholds to find the one that best classifies segments.
    """
    mean_zs: list[float] = []
    labels: list[bool] = []

    for seg in segments:
        z_vals = [
            gt_z[f] for f in range(seg["start_frame"], seg["end_frame"])
            if f in gt_z and gt_z[f] is not None
        ]
        if not z_vals:
            continue
        mean_zs.append(sum(z_vals) / len(z_vals))
        labels.append(seg["segment_type"] == "arc")

    if not mean_zs:
        _logger.warning("No valid GT z data for threshold derivation, using default %.1f dm", default_threshold)
        return default_threshold

    best_thr = default_threshold
    best_acc = -1.0
    for i in range(401):  # 0.0 to 40.0 dm in 0.1 steps
        thr = i / 10.0
        correct = sum((z > thr) == lbl for z, lbl in zip(mean_zs, labels))
        acc = correct / len(mean_zs)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    _logger.info(
        "Derived z threshold: %.1f dm (accuracy: %.1f%% on %d JSON segments)",
        best_thr, best_acc * 100, len(mean_zs),
    )
    return best_thr


def _classify_segment_by_z(
    frames: list[int],
    gt_z: dict[int, float | None],
    z_threshold: float,
) -> bool:
    """Classify a segment as arc (True) or straight (False) using mean GT z.

    Returns False (straight) if no valid z values exist.
    """
    z_vals = [gt_z[f] for f in frames if f in gt_z and gt_z[f] is not None]
    if not z_vals:
        return False
    return (sum(z_vals) / len(z_vals)) > z_threshold


def calc_split_from_json(ball_path: str) -> dict:
    """Calculate arc/straight split from JSON evaluation results.

    Always uses ``ball3d_eval_basic_angular_velocity_results.json`` produced
    by the external respo pipeline — the split is estimator-independent.

    Parameters
    ----------
    ball_path : str
        Path to clip root (e.g. "data/.../camera06/half_1").

    Returns
    -------
    dict with keys: frame_index, error_arc, error_straight, is_arc
    """
    # Build paths
    json_path = os.path.join(ball_path, "ball3d_eval_basic_angular_velocity_results.json")
    coords_path = os.path.join(ball_path, "track", "ball_3d-gt.csv")
    pivot_path = os.path.join(ball_path, "track", "ball_pivot_point-gt.csv")
    if not os.path.exists(pivot_path):
        _logger.warning("No GT pivot file found, falling back to ball_pivot_point.csv")
        pivot_path = os.path.join(ball_path, "track", "ball_pivot_point.csv")

    # Load data
    segments = _load_eval_json(json_path)
    gt_z, gt_frames, pivot_frames = _load_gt_data(coords_path, pivot_path)
    frame_to_seg = _build_frame_to_segment(segments)

    # Derive z-threshold from JSON data
    z_threshold = _derive_z_threshold(segments, gt_z)

    # Classify all GT frames between pivot points
    save_dict: dict[str, list] = {
        "frame_index": [], "error_arc": [], "error_straight": [],
        "is_arc": [], "is_arc_long": [],
    }
    num_json, num_fallback = 0, 0
    num_arc_seg, num_straight_seg = 0, 0
    num_arc_short_seg, num_arc_long_seg = 0, 0

    for i in range(len(pivot_frames) - 1):
        seg_start, seg_end = pivot_frames[i], pivot_frames[i + 1]
        # All GT frames in this inter-pivot segment
        active_frames = [f for f in range(seg_start, seg_end) if f in gt_frames]
        if not active_frames:
            continue

        # Pre-compute fallback label for uncovered frames
        fallback_is_arc = _classify_segment_by_z(active_frames, gt_z, z_threshold)

        # Track segment classification for logging
        seg_classified = None
        for f in active_frames:
            if f in frame_to_seg:
                seg = frame_to_seg[f]
                is_arc = seg["segment_type"] == "arc"
                seg_duration = seg["end_frame"] - seg["start_frame"]
                is_arc_long = is_arc and seg_duration > ARC_LONG_MIN_FRAMES
                num_json += 1
            else:
                is_arc = fallback_is_arc
                is_arc_long = False  # conservative — no segment duration info
                num_fallback += 1

            if seg_classified is None:
                seg_classified = is_arc
                if is_arc:
                    num_arc_seg += 1
                    if is_arc_long:
                        num_arc_long_seg += 1
                    else:
                        num_arc_short_seg += 1
                else:
                    num_straight_seg += 1

            save_dict["frame_index"].append(f)
            save_dict["error_arc"].append(0.0 if is_arc else 1.0)
            save_dict["error_straight"].append(1.0 if is_arc else 0.0)
            save_dict["is_arc"].append(is_arc)
            save_dict["is_arc_long"].append(is_arc_long)

    _logger.info(
        "Split: %d arc segments (%d short, %d long), %d straight segments | "
        "%d frames from JSON, %d frames from z-fallback",
        num_arc_seg, num_arc_short_seg, num_arc_long_seg,
        num_straight_seg, num_json, num_fallback,
    )
    return save_dict
