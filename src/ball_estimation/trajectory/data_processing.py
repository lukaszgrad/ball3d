from typing import Tuple
import numpy as np
import pandas as pd
from ball_estimation.data.model.geometry import Point2D, Size2D, Rectangle
from ball_estimation.camera import project_points_multiple
from ball_estimation.processors import rle_to_bitmask

IMAGE_SIZE: Size2D = Size2D.construct(width=640, height=360)
TEMPLATE_PITCH_SHAPE = (780, 1150)

class DataProcessor:
    def __init__(self, video_metadata, ball_radius_correction=1.0, epsilon_frac=0.1,
                 max_distance_for_contact_approval=0.5, min_ball_height_to_detect_high_pivot=0.5,
                 v_frac_threshold=0.5, cos_threshold=0.5, out_margin=0.1, z_correction_coefficient=1.0):
        self.video_metadata = video_metadata
        self.ball_radius_correction = ball_radius_correction
        self.epsilon_frac = epsilon_frac
        self.max_distance_for_contact_approval = max_distance_for_contact_approval
        self.min_ball_height_to_detect_high_pivot = min_ball_height_to_detect_high_pivot
        self.v_frac_threshold = v_frac_threshold
        self.cos_threshold = cos_threshold
        self.out_margin = out_margin
        self.z_correction_coefficient = z_correction_coefficient

    # TODO: could pivot point manipulating functions be stored separately from trajectory optimization related stuff?
    def prepare_pivots(
        self, df_trajectory: pd.DataFrame, ball_pivot_point: pd.DataFrame
    ) -> Tuple[np.array, pd.DataFrame]:
        """Prepares a list of pivot points
        and adds it to a "type" column to the trajectory dataframe

        Parameters
        ----------
        df_trajectory: pd.DataFrame
            DataFrame containing per-frame ball detections together with
            smoothed ball coordinates, homography estimation
            and ball-player contacts detection
        ball_pivot_point: pd.DataFrame
            DataFrame containing per-frame pivot point probabilities
            and pivot point detection

        Returns
        -------
        pivot_points_frames: np array
            list of the pivot points frames indexes
        df_trajectory: pd.DataFrame
            trajectory DataFrame with added "type" column
        """

        df_trajectory = df_trajectory.merge(
            ball_pivot_point[["file_name", "pivot_probability", "pivot_point"]],
            on=["file_name"],
            how="left",
        )
        pp_mask = df_trajectory["pivot_point"] == 1
        df_trajectory["type"] = np.where(pp_mask, "pivot_point", None)

        pivot_points_frames = df_trajectory[pp_mask]["file_name"].values
        del df_trajectory["pivot_point"]

        return pivot_points_frames, df_trajectory

    def prepare_ball_player_contacts(self, df_main, df_trajectory):
        """Adds ball-players contacts information
        to the trajectory DataFrame

        Parameters
        ----------
        df_main: pd.DataFrame
            DataFrame containing per-frame ball and players detections
            together with homography estimation
        df_trajectory: pd.DataFrame
            DataFrame containing per-frame ball detections together with
            smoothed ball coordinates, homography estimation

        Returns
        -------
        df_trajectory: pd.DataFrame
            trajectory DataFrame with added ball-player contacts columns
        """
        if len(df_trajectory) > 1:
            print("\nAnalyzing ball-players contacts ...")

        add_pitch_pano_coords(df_trajectory, self.video_metadata)
        if len(df_main) == 0:
            df_main = pd.DataFrame(
                columns=[
                    "file_name",
                    "detection_id",
                    "segmentation",
                    "category",
                    "x0",
                    "x1",
                    "y0",
                    "y1",
                ]
            )

        df = df_trajectory[
            [
                "file_name",
                "x0",
                "y0",
                "x1",
                "y1",
                "xk",
                "yk",
                "rk",
                "detection_id",
            ]
        ].copy()

        # add ball segmentation
        df = df.merge(
            df_main[["file_name", "detection_id", "segmentation"]],
            on=["file_name", "detection_id"],
            how="left",
        )

        # shift ball bounding boxes in accordance with smoothed ball center position
        df["dx"] = np.round(df.xk - df.x0 / 2 - df.x1 / 2)
        df["dy"] = np.round(df.yk - df.y0 / 2 - df.y1 / 2)
        df["x0"] = (df.x0 + df.dx).fillna(
            np.round(df.xk - df.rk * self.ball_radius_correction)
        )
        df["x1"] = (df.x1 + df.dx).fillna(
            np.round(df.xk + df.rk * self.ball_radius_correction)
        )
        df["y0"] = (df.y0 + df.dy).fillna(
            np.round(df.yk - df.rk * self.ball_radius_correction)
        )
        df["y1"] = (df.y1 + df.dy).fillna(
            np.round(df.yk + df.rk * self.ball_radius_correction)
        )

        # select players bounding boxes
        df_main = df_main[
            (df_main.category == "goalkeeper") | (df_main.category == "player")
        ][["file_name", "detection_id", "x0", "y0", "x1", "y1", "segmentation"]].copy()

        # distinguish between the ball columns and the players columns
        for col in df_main.columns[1:]:
            df_main.rename(columns={col: "p_" + col}, inplace=True)
        df = df.merge(df_main, on=["file_name"], how="left")

        # leave only frames with ball-player bounding boxes contact
        df["epsilon"] = self.epsilon_frac * (df["p_y1"] - df["p_y0"])
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

        # we consider what coordinate the ball would have
        # at the level of the player's feet
        df["yk_corr"] = df.yk - df.y1 + df.p_y1
        df["mask_start_x"] = df[["x0", "p_x0"]].min(axis=1).astype(int) - 1
        df["mask_start_y"] = df[["y0", "p_y0"]].min(axis=1).astype(int) - 1
        df["width"] = (
            df[["x1", "p_x1"]].max(axis=1).astype(int) - df["mask_start_x"] + 2
        )
        df["height"] = (
            df[["y1", "p_y1"]].max(axis=1).astype(int) - df["mask_start_y"] + 2
        )

        df["common_points"] = 0
        df["distance"] = 1e3
        for i in tqdm(range(len(df))):
            segm_ball = df["segmentation"][i]
            segm_player = df["p_segmentation"][i]

            if segm_ball is np.nan:
                ball = self.create_circular_mask(
                    h=df.height[i],
                    w=df.width[i],
                    center=(
                        df.xk[i] - df.mask_start_x[i],
                        df.yk[i] - df.mask_start_y[i],
                    ),
                    radius=df.rk[i] * self.ball_radius_correction,
                )
            else:
                ball_area = rle_to_bitmask(
                    np.array(eval(segm_ball)),
                    int(df.y1[i] - df.y0[i]),
                    int(df.x1[i] - df.x0[i]),
                )
                ball = np.zeros((df.height[i], df.width[i])).astype(bool)
                ball[
                    int(df.y0[i])
                    - df.mask_start_y[i] : int(df.y1[i])
                    - df.mask_start_y[i],
                    int(df.x0[i])
                    - df.mask_start_x[i] : int(df.x1[i])
                    - df.mask_start_x[i],
                ] = ball_area
            player_area = rle_to_bitmask(
                np.array(eval(segm_player)),
                int(df.p_y1[i] - df.p_y0[i]),
                int(df.p_x1[i] - df.p_x0[i]),
            )
            player = np.zeros((df.height[i], df.width[i])).astype(bool)
            player[
                int(df.p_y0[i])
                - df.mask_start_y[i] : int(df.p_y1[i])
                - df.mask_start_y[i],
                int(df.p_x0[i])
                - df.mask_start_x[i] : int(df.p_x1[i])
                - df.mask_start_x[i],
            ] = player_area

            common_points = np.bitwise_and(ball, player).sum()
            if common_points > 0:
                dist = 0
            else:
                XA = np.column_stack(np.where(ball))
                XB = np.column_stack(np.where(player))
                try:
                    dist = distance.cdist(XA, XB).min()
                except Exception:
                    dist = 1e3
            df.loc[i, "common_points"] = common_points
            df.loc[i, "distance"] = dist
        del df["mask_start_x"], df["mask_start_y"], df["width"], df["height"]

        # sorting players by distance from the ball
        df.sort_values(
            ["file_name", "common_points", "distance"],
            ascending=(True, False, True),
            inplace=True,
        )

        # leave one nearest player
        df.drop_duplicates(subset=["file_name"], keep="first", inplace=True)
        df["ball_height_rel"] = (df.yk - df.p_y1) / (df.p_y0 - df.p_y1)

        df_trajectory = df_trajectory.merge(
            df[
                [
                    "file_name",
                    "p_detection_id",
                    "common_points",
                    "distance",
                    "ball_height_rel",
                    "yk_corr",
                    "epsilon",
                ]
            ],
            on=["file_name"],
            how="left",
        )

        dct = {"common_points": 0, "distance": 1e3, "ball_height_rel": -1}
        for col in dct:
            if col in df_trajectory.columns:
                df_trajectory[col].fillna(dct[col], inplace=True)
            else:
                df_trajectory[col] = dct[col]

        df_trajectory["out"] = outside_pitch(df_trajectory, self.out_margin)
        df_trajectory.loc[df_trajectory["x_pitch2D"].isna(), "out"] = False
        df_trajectory["far_contact"] = df_trajectory.distance < df_trajectory["epsilon"]
        df_trajectory["close_contact"] = (
            df_trajectory.distance
            < self.max_distance_for_contact_approval * df_trajectory["epsilon"]
        )
        df_trajectory["high_contact"] = (
            df_trajectory.ball_height_rel > self.min_ball_height_to_detect_high_pivot
        )
        del df_trajectory["epsilon"]
        if len(df_trajectory) > 1:
            print("Analyzing ball-players contacts ... done\n")

        return df_trajectory

    def prepare_high_pivots(self, df_trajectory, prev="", next=""):
        """Prepares a list of high pivots
        and updates a "type" column in the trajectory DataFrame

        Parameters
        ----------
        df_trajectory: pd.DataFrame
            DataFrame containing per-frame ball detections together with
            smoothed ball coordinates, pivot point detections
            and ball-players contacts information
        prev, next: str
            trajectory types in frames before and after the dataframe (if any)

        Returns
        -------
        pivot_points_frames: np array
            list of the pivot points frames indexes
        high_pivot_frames: np.array
            list of the high pivot points frames indexes
        df_trajectory: pd.DataFrame
            trajectory DataFrame with updated "type" column
        """
        # previous and next trajectories are not straight
        df_trajectory["not_on_line"] = (
            df_trajectory["type"].shift(1).fillna(prev) != "straight"
        ) & (df_trajectory["type"].shift(-1).fillna(next) != "straight")

        # keep old types
        df_trajectory.rename(columns={"type": "type_initial"}, inplace=True)
        df_trajectory["type"] = np.nan
        init_pivots = df_trajectory["type_initial"].isin(
            ["pivot_point", "additional_pivot_point"]
        )

        df_trajectory.loc[
            init_pivots,
            "type",
        ] = "pivot_point"
        df_trajectory.loc[
            init_pivots & df_trajectory["out"] & df_trajectory["far_contact"],
            "type",
        ] = "high_pivot_point"
        df_trajectory.loc[
            init_pivots
            & df_trajectory["close_contact"]
            & df_trajectory["high_contact"]
            & df_trajectory["not_on_line"],
            "type",
        ] = "high_pivot_point"

        # TODO: it seems that this function has multiple responbilities

        # correct ball position in high contact frames
        df_trajectory["yk"] = np.where(
            df_trajectory["type"] == "high_pivot_point",
            df_trajectory["yk_corr"],
            df_trajectory["yk"],
        )
        # re-calculate ball pitch position
        make_homography_to_pitch(df_trajectory, "xk", "yk", self.video_metadata)
        interpolate_cols = ["x_pitch2D", "y_pitch2D"]
        df_trajectory[interpolate_cols] = df_trajectory[interpolate_cols].interpolate(
            method="linear", limit=4, limit_direction="both"
        )

        pivot_points_frames = df_trajectory[
            df_trajectory.type.isin(["pivot_point", "high_pivot_point"])
        ]["file_name"].values
        high_pivot_frames = df_trajectory[df_trajectory.type == "high_pivot_point"][
            "file_name"
        ].values

        return pivot_points_frames, high_pivot_frames, df_trajectory

    def prepare_wrong_pivots(self, df_trajectory):
        """Prepares a list of wrong pivots
        and updates a "type" column in the trajectory DataFrame

        Parameters
        ----------
        df_trajectory: pd.DataFrame
            DataFrame containing per-frame ball detections together with
            smoothed ball coordinates, pivot point detections
            and ball-players contacts information

        Returns
        -------
        pivot_points_frames: np array
            list of the pivot points frames indexes
        high_pivot_frames: np.array
            list of the high pivot points frames indexes
        df_trajectory: pd.DataFrame
            trajectory DataFrame with updated "type" column
        """
        for i in ["x", "y"]:
            df_trajectory[f"d{i}0"] = (
                df_trajectory[f"{i}_predicted"].shift(1)
                - df_trajectory[f"{i}_predicted"].shift(2)
            ).astype("float64")
            df_trajectory[f"d{i}1"] = (
                df_trajectory[f"{i}_predicted"].shift(-2)
                - df_trajectory[f"{i}_predicted"].shift(-1)
            ).astype("float64")
        for j in [0, 1]:
            df_trajectory[f"d{j}"] = np.sqrt(
                df_trajectory[f"dx{j}"] * df_trajectory[f"dx{j}"]
                + df_trajectory[f"dy{j}"] * df_trajectory[f"dy{j}"]
            )
        df_trajectory["v_frac"] = df_trajectory["d1"] / df_trajectory["d0"]
        df_trajectory["cos"] = (
            df_trajectory["dx0"] * df_trajectory["dx1"]
            + df_trajectory["dy0"] * df_trajectory["dy1"]
        ) / (df_trajectory["d1"] * df_trajectory["d0"])

        init_pivots = df_trajectory["type"].isin(
            ["pivot_point", "additional_pivot_point", "high_pivot_point"]
        )
        wrong_pivots = (
            (~df_trajectory.far_contact)
            & (
                (df_trajectory.v_frac > self.v_frac_threshold)
                | (df_trajectory.cos < self.cos_threshold)
            )
            & (df_trajectory.x_predicted > 50)
            & (df_trajectory.x_predicted < TEMPLATE_PITCH_SHAPE[1] - 50)
        )
        df_trajectory.loc[
            init_pivots & wrong_pivots,
            "type",
        ] = "wrong_pivot_point"

        pivot_points_frames = df_trajectory[
            df_trajectory.type.isin(
                ["pivot_point", "additional_pivot_point", "high_pivot_point"]
            )
        ]["file_name"].values
        wrong_pivot_frames = df_trajectory[df_trajectory.type == "wrong_pivot_point"][
            "file_name"
        ].values

        return pivot_points_frames, wrong_pivot_frames, df_trajectory

    def add_z_corrected(self, df_trajectory):
        """Adds a column with corrected ball Z coordinate
           (obsolete)

        Parameters
        ----------
        df_trajectory: pd.DataFrame
            DataFrame containing per-frame ball detections together with
            predicted ball coordinates
        """
        if "z_predicted" not in df_trajectory.columns:
            df_trajectory["z_predicted"] = 0
        df_trajectory["z_predicted_raw"] = df_trajectory["z_predicted"]
        df_trajectory["z_predicted"] = (
            df_trajectory["z_predicted"] * self.z_correction_coefficient
        )


def get_hom_matrices(df):
    hom_keys = [f"h{i}" for i in range(9)]
    return df[hom_keys].values.reshape(-1, 3, 3)


def make_homography_to_pitch(df, x_col, y_col, video_metadata):
    # resampling broadcast image to IMAGE_SIZE
    df["xc"] = df[x_col] * IMAGE_SIZE.width / video_metadata["width"]
    df["yc"] = df[y_col] * IMAGE_SIZE.height / video_metadata["height"]

    # apply homography
    H = get_hom_matrices(df)
    df[["x_pitch2D", "y_pitch2D"]] = project_points_multiple(H, df[["xc", "yc"]].values)


def add_pitch_pano_coords(df, video_metadata):
    df.reset_index(drop=True, inplace=True)
    make_homography_to_pitch(df, "xk", "yk", video_metadata)

    interpolate_cols = ["x_pitch2D", "y_pitch2D"]
    df[interpolate_cols] = df[interpolate_cols].interpolate(
        method="linear", limit=4, limit_direction="both"
    )


def get_ellipse_mask(h, w, center, axes):
    Y, X = np.ogrid[:h, :w]

    x0, y0 = center
    a, b = axes

    d = (X - x0) ** 2 / a**2 + (Y - y0) ** 2 / b**2

    mask = d <= 1
    return mask


def outside_pitch(df, out_margin):
    """Detects if ball (x,y) coordinates are outside pitch2D

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing per-frame ball detections

    Returns
    -------
    list of boolean
        Determines if ball (x,y) coordinates are outside pitch2D
    """

    out_margin = float(out_margin)
    extended_pitch_area = Rectangle(
        p0=Point2D(x=-out_margin, y=-out_margin),
        p1=Point2D(x=TEMPLATE_PITCH_SHAPE[1] + out_margin, y=TEMPLATE_PITCH_SHAPE[0] + out_margin),
    )

    inside = extended_pitch_area.contains_points(df[["x_pitch2D", "y_pitch2D"]].values)
    out = ~inside

    return out

def create_circular_mask(h, w, center=None, radius=6):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + 0.75 * (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask
