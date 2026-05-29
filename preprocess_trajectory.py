import json
import logging
import os
import pickle
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig

from ball_estimation.io.versioned import load_versioned_df, versioned_path
from ball_estimation.preprocessing import preprocess_ball

_logger = logging.getLogger(__name__)


def load_pickle(file_path: str) -> Any:
    """Load a pickle file, trying ``_dict`` suffix fallback."""
    if not os.path.isfile(file_path):
        # Try without _dict suffix (e.g. stitch_pano_calib.pickle)
        alt = file_path.replace("_dict.pickle", ".pickle")
        if alt != file_path and os.path.isfile(alt):
            file_path = alt
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_json(file_path: str) -> dict[str, Any]:
    """Load a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg: DictConfig) -> None:
    """Preprocess raw ball detections into merged ball-player DF and pivot points."""
    root = cfg.root
    input_version = cfg.version
    output_version = cfg.output_version if cfg.output_version else cfg.version

    # Load inputs
    ball_detection = load_versioned_df(
        os.path.join(root, "detection"), "ball_detection", version=input_version
    )
    player_detection = load_versioned_df(
        os.path.join(root, "detection"), "detection", version=input_version
    )
    hom_smooth_df = pd.read_csv(os.path.join(root, "hom_smooth.csv"))
    video_metadata = load_json(os.path.join(root, "sequence_metadata.json"))


    # Run preprocessing pipeline
    df_merged = preprocess_ball(
        ball_detection_df=ball_detection,
        player_detection_df=player_detection,
        hom_smooth_df=hom_smooth_df,
        video_metadata=video_metadata,
        preprocessing_cfg=cfg.preprocessing,
        step_frame=cfg.step_frame,
    )

    # Save outputs
    dev_dir = os.path.join(root, "dev")
    os.makedirs(dev_dir, exist_ok=True)
    out_merged = versioned_path(dev_dir, "df_merged_ball_player", version=output_version)
    df_merged.to_csv(out_merged, index=False)
    _logger.info("Saved merged ball-player DF to %s (%d rows)", out_merged, len(df_merged))


if __name__ == "__main__":
    main()
