import json
import os
import pickle
from typing import Dict, Any

import cv2 as cv
import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig

from ball_estimation.evaluate import evaluate_ball_trajectory
from ball_estimation.io.versioned import load_versioned_df


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

    dev_ball3d = load_versioned_df(
        os.path.join(root, "track"), "ball_3d", version=cfg.version
    )

    if os.path.exists(os.path.join(root, "track", "ball_3d-gt.csv")):
        track_ball3d_gt = pd.read_csv(os.path.join(root, "track", "ball_3d-gt.csv"))
    else:
        raise FileNotFoundError(
            f"Ground truth ball trajectory not found in {root}/track"
        )

    if os.path.exists(os.path.join(root, "dev", "pauses.csv")):
        df_pauses = pd.read_csv(os.path.join(root, "dev", "pauses.csv"))
    else:
        df_pauses = None

    split_path = os.path.join(root, "track", "split.csv")
    if os.path.exists(split_path):
        split_df = pd.read_csv(split_path)
    else:
        split_df = None
        print(f"Split file not found at {split_path}. No split data will be used.")

    (
        error_plot,
        max_error_plot,
        metrics,
        gt_metrics,
        errors_df,
    ) = evaluate_ball_trajectory(
        ball_3d_gt=track_ball3d_gt,
        ball_3d_dev=dev_ball3d,
        dev__pauses=df_pauses,
        split_df=split_df,
    )

    # Save the results — use output_version suffix when provided
    out_ver = cfg.get("output_version") or cfg.version or None
    suffix = f"-{out_ver}" if out_ver else ""

    os.makedirs(os.path.join(root, "eval"), exist_ok=True)
    cv.imwrite(os.path.join(root, "eval", f"error_plot{suffix}.png"), error_plot)
    cv.imwrite(os.path.join(root, "eval", f"max_error_plot{suffix}.png"), max_error_plot)
    with open(os.path.join(root, "eval", f"metrics{suffix}.json"), "w") as f:
        json.dump(metrics.dict(), f, indent=4, sort_keys=True)
    with open(os.path.join(root, "eval", f"gt_metrics{suffix}.json"), "w") as f:
        json.dump(gt_metrics.dict(), f, indent=4, sort_keys=True)
    errors_df.to_csv(os.path.join(root, "eval", f"errors{suffix}.csv"), index=False)


if __name__ == "__main__":
    main()
