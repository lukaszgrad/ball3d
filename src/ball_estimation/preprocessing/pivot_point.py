"""Pivot point prediction using LightGBM ensemble model.

Extracted from respo.modules.ml.pivot_point.pivot_point (feature engineering
and inference) and respo.modules.module_pivot_points (postprocessing).
"""

import logging
import pickle

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


class PivotPointModel:
    """Inference-only pivot point model with feature engineering.

    Parameters
    ----------
    width : int
        Video frame width.
    height : int
        Video frame height.
    """

    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.model: list = []

    def make_feature_engineering(self, df_: pd.DataFrame) -> pd.DataFrame:
        """Engineer 49 features from ball bbox and smoothed coordinates.

        Parameters
        ----------
        df_ : pd.DataFrame
            Must contain: file_name, x0, y0, x1, y1, score, xk, yk.

        Returns
        -------
        pd.DataFrame
            DataFrame with 49 feature columns.
        """
        df = df_[
            ["file_name", "x0", "y0", "x1", "y1", "score", "xk", "yk"]
        ].copy()

        # normalise to 1920x1080 for FE
        for col in ["x0", "x1", "xk"]:
            df[col] /= self.width / 1920
        for col in ["y0", "y1", "yk"]:
            df[col] /= self.height / 1080

        for col in ["xk", "yk"]:
            df[f"{col}_prev"] = df[col].shift()
            df[f"{col}_next"] = df[col].shift(-1)

            df[f"d{col}_prev"] = df[col] - df[f"{col}_prev"]
            df[f"d{col}_next"] = df[f"d{col}_prev"].shift(-1)

            df[f"d{col}_diff"] = df[f"d{col}_next"] - df[f"d{col}_prev"]
            df[f"d{col}_frac"] = df[f"d{col}_next"] / df[f"d{col}_prev"]
            df[f"d{col}_prod"] = df[f"d{col}_next"] * df[f"d{col}_prev"]
            df[f"d{col}_diff_prev"] = df[f"d{col}_diff"].shift()
            df[f"d{col}_diff_next"] = df[f"d{col}_diff"].shift(-1)
            df[f"d{col}_diff_next_to_current"] = (
                df[f"d{col}_diff_next"] / df[f"d{col}_diff"]
            )
            df[f"d{col}_diff_current_to_prev"] = (
                df[f"d{col}_diff"] / df[f"d{col}_diff_prev"]
            )
            df[f"d{col}_diff_next_to_prev"] = (
                df[f"d{col}_diff_next"] / df[f"d{col}_diff_prev"]
            )

        df["vk_prev"] = np.sqrt(
            (df["xk"] - df["xk_prev"]) ** 2 + (df["yk"] - df["yk_prev"]) ** 2
        )
        df["vk_next"] = df["vk_prev"].shift(-1)
        df["vk_diff"] = df["vk_next"] - df["vk_prev"]
        df["vk_diff_prev"] = df["vk_diff"].shift()
        df["vk_diff_next"] = df["vk_diff"].shift(-1)
        df["vk_diff_next_to_current"] = df["vk_diff_next"] / df["vk_diff"]
        df["vk_diff_current_to_prev"] = df["vk_diff"] / df["vk_diff_prev"]
        df["vk_diff_next_to_prev"] = df["vk_diff_next"] / df["vk_diff_prev"]

        df["vk_frac"] = df["vk_next"] / df["vk_prev"]
        df["vk_frac_prev"] = df["vk_frac"].shift()
        df["vk_frac_next"] = df["vk_frac"].shift(-1)
        df["vk_frac_mean"] = (
            df["vk_frac_prev"] + df["vk_frac"] + df["vk_frac_next"]
        ) / 3
        df["vk_frac_max"] = df[["vk_frac_prev", "vk_frac", "vk_frac_next"]].max(axis=1)
        df["vk_prod"] = df["vk_next"] * df["vk_prev"]

        df["ball_dx_size"] = df["x1"] - df["x0"]
        df["ball_dy_size"] = df["y1"] - df["y0"]
        df["ball_dx_size"] = df["ball_dx_size"].ffill(limit=50)
        df["ball_dy_size"] = df["ball_dy_size"].ffill(limit=50)
        df["ball_v_size"] = np.sqrt(df["ball_dx_size"] ** 2 + df["ball_dy_size"] ** 2)

        for coord in ["dx", "dy", "v"]:
            for measure in ["prev", "next", "diff", "frac", "prod"]:
                df[f"{coord}k_{measure}_ball_rel"] = (
                    df[f"{coord}k_{measure}"] / df[f"ball_{coord}_size"]
                )

        df["angle"] = (
            df["dxk_prev"] * df["dxk_next"] + df["dyk_prev"] * df["dyk_next"]
        ) / (
            np.sqrt(df["dxk_prev"] ** 2 + df["dyk_prev"] ** 2)
            * np.sqrt(df["dxk_next"] ** 2 + df["dyk_next"] ** 2)
        )
        df["angle_prev"] = df["angle"].shift()
        df["angle_next"] = df["angle"].shift(-1)
        df["angle_mean"] = df[["angle_prev", "angle", "angle_next"]].mean(axis=1)
        df["angle_max"] = df[["angle_prev", "angle", "angle_next"]].max(axis=1)

        df["dxdy"] = df["ball_dx_size"] / df["ball_dy_size"]
        df["dxdy_prev"] = df["dxdy"].shift(1)
        df["dxdy_next"] = df["dxdy"].shift(-1)
        df["dxdy_next_to_current"] = df["dxdy_next"] / df["dxdy"]
        df["dxdy_current_to_prev"] = df["dxdy"] / df["dxdy_prev"]
        df["dxdy_next_to_prev"] = df["dxdy_next"] / df["dxdy_prev"]

        feats = [
            "x0", "y0", "x1", "y1", "score", "xk", "yk",
            "xk_prev", "xk_next", "yk_prev", "yk_next",
            "dxk_prev", "dxk_next", "dxk_diff", "dxk_frac", "dxk_prod",
            "dyk_prev", "dyk_next", "dyk_diff", "dyk_frac", "dyk_prod",
            "vk_prev", "vk_next", "vk_diff", "vk_frac", "vk_prod",
            "ball_dx_size", "ball_dy_size", "ball_v_size",
            "dxk_prev_ball_rel", "dxk_next_ball_rel", "dxk_diff_ball_rel",
            "dxk_frac_ball_rel", "dxk_prod_ball_rel",
            "dyk_prev_ball_rel", "dyk_next_ball_rel", "dyk_diff_ball_rel",
            "dyk_frac_ball_rel", "dyk_prod_ball_rel",
            "vk_prev_ball_rel", "vk_next_ball_rel", "vk_diff_ball_rel",
            "vk_frac_ball_rel", "vk_prod_ball_rel",
            "angle",
        ]
        return df[feats]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict pivot probability using ensemble averaging."""
        preds = np.zeros(df.shape[0])
        for fold_classifier in self.model:
            fold_preds = fold_classifier.predict(df)
            if fold_preds.ndim == 2:
                fold_preds = 1 - fold_preds[:, 0]
            preds += fold_preds / len(self.model)
        return preds


def _postprocess_pivots(
    pivot_probability: pd.Series,
    df: pd.DataFrame,
    threshold: float,
    window: int,
) -> pd.Series:
    """Apply threshold, temporal peak filtering, and physics constraints."""
    prob_above_threshold = pivot_probability > threshold
    best_prob_in_window = (
        pivot_probability
        == pivot_probability.rolling(window, min_periods=1, center=True).max()
    )
    not_unfounded_pivot = ~df.out | df.far_contact
    pivot_point = (
        prob_above_threshold & best_prob_in_window & not_unfounded_pivot
    ).astype(int)
    return pivot_point


def predict_pivot_points(
    df_ball: pd.DataFrame,
    model_checkpoint: str,
    video_metadata: dict,
    threshold: float = 0.11,
    window: int = 11,
) -> pd.DataFrame:
    """Predict pivot points using a pre-trained LightGBM ensemble.

    Parameters
    ----------
    df_ball : pd.DataFrame
        Output of ``detect_ball_player_contacts``. Must contain columns:
        file_name, track_id, x0, y0, x1, y1, score, xk, yk_original,
        out, far_contact.
    model_checkpoint : str
        Path to pickled LightGBM model.
    video_metadata : dict
        Must contain ``width``, ``height``.
    threshold : float
        Probability threshold for pivot detection.
    window : int
        Rolling window size for temporal peak filtering.

    Returns
    -------
    pd.DataFrame
        Columns: file_name, track_id, pivot_probability, pivot_point.
    """
    with open(model_checkpoint, "rb") as f:
        pp_model_weights = pickle.load(f)

    pp_model = PivotPointModel(
        width=video_metadata["width"], height=video_metadata["height"]
    )
    pp_model.model = pp_model_weights

    df = df_ball[
        [
            "file_name", "track_id", "x0", "y0", "x1", "y1", "score",
            "xk", "yk_original", "out", "far_contact",
        ]
    ].rename(columns={"yk_original": "yk"})

    out = []
    for _, df_track in df.groupby("track_id"):
        model_input = pp_model.make_feature_engineering(df_track)
        pivot_probability = pd.Series(
            pp_model.predict_proba(model_input), index=df_track.index
        )
        df_track = df_track.copy()
        df_track["pivot_probability"] = pivot_probability.values
        df_track["pivot_point"] = _postprocess_pivots(
            pivot_probability, df_track, threshold, window
        ).values
        out.append(df_track)

    result = pd.concat(out, ignore_index=True).sort_values(["file_name", "track_id"])
    return result[["file_name", "track_id", "pivot_probability", "pivot_point"]]
