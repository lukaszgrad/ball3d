import json
import logging
import os
import pickle
from typing import Dict, Any

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig

from ball_estimation.io.versioned import load_versioned_df, versioned_path
from ball_estimation.io.video import FileVideo, VideoMetadata, GeneratorVideo
from ball_estimation.visualize import visualize_trajectory

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_pickle(file_path: str) -> Any:
    """Load a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg: DictConfig) -> None:
    """Main function to visualize ball trajectory."""
    root = cfg.root

    logger.info("Loading data...")
    df_trajectory = load_versioned_df(
        os.path.join(root, "track"), "ball_3d", version=cfg.version
    )

    if cfg.visualisation.show_ground_truth:
        df_trajectory_gt = pd.read_csv(os.path.join(root, "track", "ball_3d-gt.csv"))
    else:
        df_trajectory_gt = None

    camera_smooth_df = pd.read_csv(os.path.join(root, "camera_smooth.csv"))

    logger.info("Loading metadata and config...")
    video_metadata = load_json(os.path.join(root, "sequence_metadata.json"))

    video_meta = VideoMetadata(
        width=video_metadata["width"],
        height=video_metadata["height"],
        fps=float(video_metadata["fps"]),
    )

    video = FileVideo(video_meta, os.path.join(root, "input.mkv"))

    fps = video_meta.fps
    start_frame = int(fps * cfg.start_sec)
    end_frame = int(fps * cfg.end_sec) if cfg.end_sec >= 0 else None

    logger.info("Generating visualization...")
    vis_generator = visualize_trajectory(
        video=video,
        df_trajectory=df_trajectory,
        camera_smooth_df=camera_smooth_df,
        video_metadata=video_meta,
        traj_params=cfg.visualisation,
        step_frame=cfg.step_frame,
        df_trajectory_gt=df_trajectory_gt,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    vis_video = GeneratorVideo(video_meta, vis_generator)

    output_path = versioned_path(root, "ball_vis", version=cfg.version, extension=".mp4")
    logger.info(f"Saving visualization to {output_path}")
    vis_video.save(output_path)
    logger.info("Visualization saved successfully")


if __name__ == "__main__":
    main()
