import json
import logging
import os
import pickle
from typing import Dict, Any

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig

from ball_estimation.functions import estimate_trajectory
from ball_estimation.io.versioned import load_versioned_df, versioned_path

_logger = logging.getLogger(__name__)


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
    """Main function to estimate ball trajectory."""
    root = cfg.root

    # Load data
    ball_detection = load_versioned_df(
        os.path.join(root, "detection"), "ball_detection", version=cfg.version
    )
    df = load_versioned_df(
        os.path.join(root, "dev"), "df_merged_ball_player", version=cfg.version
    )

    if cfg.use_gt_pivots:
        try:
            ball_pivot_point = load_versioned_df(
                os.path.join(root, "track"), "ball_pivot_point-gt", version=cfg.version
            )
        except FileNotFoundError:
            _logger.warning(
                "GT pivot points not found, falling back to regular pivot points"
            )
            ball_pivot_point = load_versioned_df(
                os.path.join(root, "track"), "ball_pivot_point", version=cfg.version
            )
    else:
        ball_pivot_point = load_versioned_df(
            os.path.join(root, "track"), "ball_pivot_point", version=cfg.version
        )
    camera_smooth_df = pd.read_csv(os.path.join(root, "camera_smooth.csv"))

    camera = load_pickle(
        os.path.join(
            os.path.dirname(root), "pitch_geom", "calibrate_camera_dict.pickle"
        )
    )

    # Load metadata and config
    video_metadata = load_json(os.path.join(root, "sequence_metadata.json"))

    estimator_name = cfg.trajectory.estimator_name
    estimator_parameters = cfg.trajectory.estimator_params

    # Estimate trajectory
    result_df = estimate_trajectory(
        ball_detection=ball_detection,
        df=df,
        ball_pivot_point=ball_pivot_point,
        camera=camera,
        camera_smooth_df=camera_smooth_df,
        video_metadata=video_metadata,
        estimator_name=estimator_name,
        estimator_parameters=estimator_parameters,
        n_jobs=cfg.n_jobs,
        step_frame=cfg.step_frame,
        start_sec=cfg.start_sec,
        end_sec=cfg.end_sec,
    )

    # Save the resulting DataFrame
    output_version = cfg.get("output_version") or cfg.version
    output_path = versioned_path(
        os.path.join(root, "track"), "ball_3d", version=output_version
    )
    result_df.to_csv(output_path, index=False)
    print(f"Saved trajectory to {output_path}")


if __name__ == "__main__":
    main()
