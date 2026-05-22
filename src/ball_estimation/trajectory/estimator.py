from abc import ABC
from typing import List, Tuple
import warnings

import numpy as np
import pandas as pd

from ball_estimation.data.model.geometry import Size2D
from ball_estimation.camera import Camera
from ball_estimation.trajectory import Trajectory3D
from ball_estimation.trajectory.model import StraightMotionParameters
from ball_estimation.trajectory.data_processing import (
    DataProcessor,
)
from ball_estimation.trajectory.optimization.helpers import (
    fit_ball_straight_move_model,
    fit_ball_arc_move_model,
)

__author__ = "Aliaksandr Varashylau"


IMAGE_SIZE: Size2D = Size2D.construct(width=640, height=360)
TEMPLATE_PITCH_SHAPE = (780, 1150)

# TODO split functions and the mega class

class TrajectoryEstimator(ABC):
    """Base class for all trajectory estimation methods"""

    def __init__(self, *args, **kwargs):
        pass

    def estimate_trajectory(self, df, *args, **kwargs):
        """Estimate 3D ball trajectory

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing per-frame player and ball detections together with
            homography estimation
        args
        kwargs
        """
        raise NotImplementedError(
            "estimate_trajectory() should be implemented in a subclass"
        )


class KineticTrajectoryEstimator(TrajectoryEstimator):
    """Ball trajectory estimator based on kinetic ball model

    Parameters
    ----------
    camera: camera.Camera
        Instance of Camera containing intrinsic and extrinsic camera parameters for
        pitch panorama
    camera_smooth_df: pd.DataFrame
    video_metadata: VideoMetadata
        Video metadata.
    debug: int
        Limit the list of pivot points to speed up the calculation
        non-positive value means no limit
    verbose: bool
        Show debugging messages
    probability_threshold_to_detect_pivot_point: float
        The threshold value determining whether the frame contains a pivot point
    pivot_points_window: int
        The value of the frames window in which only one pivot point is allowed
    nans_allowed: float
        Permissible proportion of nan elements when fitting a trajectory
    ball_radius_correction: float
        Correction coefficient for ball radius in ball masks
    z_correction_coefficient: float
        Coefficient for Z-axis calibration
    max_distance_for_contact_approval: float
        The relative threshold value to determine if the ball is in contact with the player
    epsilon_frac: float
        Max allowed distance between bounding boxes (in pixels)
        to consider possibility of ball-player contact
        as a fraction of player height
    out_margin: int
        Min distance from ball to pitch2D bounders (in decimetres)
        to consider ball as "out of field" ball
    min_ball_height_to_detect_high_pivot: float
        The relative height of the ball in relation to the player,
        below which we do not fix high pivot
    weight_start_straight: float
        Weight of the starting point fit error in a straight trajectory error function
    weight_end_straight: float
        Weight of the final point fit error in a straight trajectory error function
    allowed_endpoints_shift_straight: float
        Max allowed endpoints shift in a straight trajectory fitting (in pixels = decimetres)
    good_fit_threshold: int
        The threshold value of the straight trajectory fit error
        to skip the arc trajectory fitting step
    allowed_startpoint_shift_arc: float
        Max allowed startpoint shift in an arc trajectory fitting (in pixels = decimetres)
    weight_end_arc: float
        Weight of the final point fit error in an arc trajectory error function
    weight_zpenalty_arc: float
        Weight of the z-penalty in an arc trajectory error function
    ballsize_relative_loss: bool
        Use ball size relative loss function in trajectory fitting
    use_relative_error: bool
        Use error values relative to trajectory length in trajectory fitting
    dynamic_improvement_factor: float
        The threshold factor value to accept new pivot foint
    gravitation_g: float
        The initial value of the gravitational constant g used in an arc trajectory fitting
    g_variation: float
        Allowable variation of 'g' when fitting the trajectory
    Magnus_effect: bool
        Take into account the Magnus effect
    kl_ks_variation: float
        Allowable variation of 'kl', 'ks' coefficients in Magnus formula
    correct_wrong_pivots: bool
        Detect and correct wrong pivot points
    v_frac_threshold: float
        The threshold value of the velocity fraction
        to detect wrong pivot point
    cos_threshold: float
        The threshold value of the cos(v1^v2) value
        to detect wrong pivot point
    max_ball_speed: float
        Limiting the maximum speed of the ball to avoid implausible trajectories
    too_short_track: int
        Lower bounady of frames number to skip trajectory fitting
    too_long_track: int
        Upper bounady of frames number to skip trajectory fitting
    allowed_detection_error:
        The threshold fiting error value to reject the trajectory

    Attributes
    ----------
    camera: camera.Camera
        Instance of Camera containing intrinsic and extrinsic camera parameters for
        pitch panorama
    camera_smooth_df: pd.DataFrame
        Instance of ImageStitcher used to estimate pitch panorama
    video_metadata: VideoMetadata
        Video metadata.
    debug: int
        Limit the list of pivot points to speed up the calculation
        non-positive value means no limit
    verbose: bool
        Show debugging messages
    probability_threshold_to_detect_pivot_point: float
        The threshold value determining whether the frame contains a pivot point
    pivot_points_window: int
        The value of the frames window in which only one pivot point is allowed
    ball_radius_correction: float
        Correction coefficient for ball radius in ball masks
    z_correction_coefficient: float
        Coefficient for Z-axis calibration
    max_distance_for_contact_approval: float
        The relative threshold value to determine if the ball is in contact with the player
    epsilon_frac: float
        Max allowed distance between bounding boxes (in pixels)
        to consider possibility of ball-player contact
        as a fraction of player height
    out_margin: int
        Min distance from ball to pitch2D bounders (in decimetres)
        to consider ball as "out of field" ball
    min_ball_height_to_detect_high_pivot: float
        The relative height of the ball in relation to the player,
        below which we do not fix high pivot
    weight_start_straight: float
        Weight of the starting point fit error in a straight trajectory error function
    weight_end_straight: float
        Weight of the final point fit error in a straight trajectory error function
    allowed_endpoints_shift_straight: float
        Max allowed endpoints shift in a straight trajectory fitting (in pixels = decimetres)
    good_fit_threshold: int
        The threshold value of the straight trajectory fit error
        to skip the arc trajectory fitting step
    allowed_startpoint_shift_arc: float
        Max allowed startpoint shift in an arc trajectory fitting (in pixels = decimetres)
    weight_end_arc: float
        Weight of the final point fit error in an arc trajectory error function
    weight_zpenalty_arc: float
        Weight of the z-penalty in an arc trajectory error function
    use_relative_error: bool
        Use error values relative to trajectory length in trajectory fitting
    use_dynamic_pivots: bool
        Use method of dynamical adding pivot points to improve existing trajectories
    top_K: int
        Number of additional pivot candidates at each fitting step
    max_depth: int
        Max depth of dynamic pivot points algorythm
    dynamic_improvement_factor: float
        The threshold factor value to accept new pivot foint
    gravitation_g: float
        The initial value of the gravitational constant g used in an arc trajectory fitting
    max_ball_speed: float
        Limiting the maximum speed of the ball to avoid implausible trajectories
    too_short_track: int
        Lower bounady of frames number to skip trajectory fitting
    too_long_track: int
        Upper bounady of frames number to skip trajectory fitting
    allowed_detection_error:
        The threshold fiting error value to reject the trajectory
    """

    def __init__(
        self,
        camera,
        camera_smooth_df,
        video_metadata,
        debug,
        verbose,
        use_high_pivots_correction,
        use_shots_fit,
        good_fit_threshold,
        use_relative_error,
        dynamic_improvement_factor,
        correct_wrong_pivots,
        too_short_track,
        too_long_track,
        allowed_detection_error,
        data_processor_config,
        arc_optimizer_config,
        straight_optimizer_config,
        estimator_interface_config,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.video_metadata = video_metadata
        self.debug = debug
        self.verbose = verbose
        self.use_high_pivots_correction = use_high_pivots_correction
        self.use_shots_fit = use_shots_fit
        self.good_fit_threshold = good_fit_threshold
        self.use_relative_error = use_relative_error
        self.dynamic_improvement_factor = dynamic_improvement_factor
        self.correct_wrong_pivots = correct_wrong_pivots
        self.too_short_track = too_short_track
        self.too_long_track = int(too_long_track * self.video_metadata["height"] / 1080) # 1080 is what?
        self.allowed_detection_error = allowed_detection_error

        self.fit_endpoint = True

        self.data_processor = DataProcessor(
            video_metadata, **data_processor_config
        )

        arc_optimizer_config.loss_function.update(camera, camera_smooth_df, video_metadata)
        straight_optimizer_config.loss_function.update(camera, camera_smooth_df, video_metadata)

        self.arc_optimizer = arc_optimizer_config
        self.straight_optimizer = straight_optimizer_config


    def relative_error(self, err, start_frame, end_frame):
        if self.use_relative_error and err < 1e8:
            return 25 * err / (end_frame - start_frame + 1) # 25 is fps or what?
        return err


    def fit_one_trajectory_segment(
        self,
        df_trajectory: pd.DataFrame,
        start_frame: int,
        end_frame: int,
        endpoint: bool,
        pfx: str = "",
    ) -> Tuple[
        List[float],
        List[float],
        List[float],
        float,
        float,
        str,
        float,
        float,
        float,
        float,
    ]:
        """
        Fits one trajectory segment.

        Parameters:
            df_trajectory (pd.DataFrame): DataFrame containing per-frame ball detections
                together with detected pivot points frames.
            start_frame (int): Index of the start frame of the trajectory to fit.
            end_frame (int): Index of the end frame of the trajectory to fit.
            endpoint (bool): Flag enabling trajectory endpoint fitting.
            pfx (str, optional): Prefix for prints.

        Returns:
            Tuple[List[float], List[float], List[float], float, float, str, float, float, float, float]:
                - xt, yt, zt: Lists of float representing the predicted (x, y, z) per-frame coordinates.
                - err: Float representing the trajectory fitting error.
                - g_best: Float representing the gravitation constant of the best arc trajectory.
                - trajectory_type: String describing the fitted trajectory.
                - k: Float representing the air resistance coefficient within the fitted trajectory.
                - vx, vy, vz: Floats representing ball velocities at the trajectory starting point along x, y, and z axes.
        """
        track_len = end_frame - start_frame + 1
        xt, yt, zt, g_best, trajectory_type, k, vx, vy, vz, best_motion_parameters = [None] * 10
        
        err = 1e9
        if self.verbose:
            self.print_segment_info(df_trajectory, pfx, start_frame, end_frame, track_len)

        if track_len <= self.too_short_track:
            # Insufficient frames to create a meaningful trajectory fit.
            trajectory_type = "too_short_track"
            if self.verbose:
                print("{}skipped: too short".format(pfx))
        elif track_len >= self.too_long_track:
            # Excessive frames in the fitted segment may indicate that
            # this is not a single trajectory segment but rather multiple segments
            # with missed pivot points due to camera changes or other reasons.
            trajectory_type = "too_long_track"
            if self.verbose:
                print("{}skipped: too long".format(pfx))
        else:
            # Attempting to fit a straight trajectory.
            (
                err_straight,
                x_start_best,
                y_start_best,
                x_end_best,
                y_end_best,
                k_best,
            ) = fit_ball_straight_move_model(
                df_trajectory, start_frame, end_frame,
                self.straight_optimizer,
                pfx
            )
            err_arc = 1e9

            # If the fitting error for the straight trajectory is too large
            # to confidently accept, we attempt to fit a parabolic trajectory.
            if (
                self.relative_error(err_straight, start_frame, end_frame)
                > self.good_fit_threshold
            ):
                print(df_trajectory)
                (
                    err_arc,
                    motion_parameters_estimate
                ) = fit_ball_arc_move_model(
                    df_trajectory, start_frame, end_frame, endpoint,
                    self.arc_optimizer, float(self.video_metadata["fps"])
                )
            
            # Choose the trajectory type that fits better:
            if err_arc is not None and err_arc < err_straight:
                fps = float(self.video_metadata["fps"])

                vx0_best = motion_parameters_estimate.vx0
                vy0_best = motion_parameters_estimate.vy0
                vz0_best = motion_parameters_estimate.vz0
                g_best = motion_parameters_estimate.g
                k3_best = motion_parameters_estimate.k3

                best_motion_parameters = motion_parameters_estimate
                T = track_len - 1
                t = (1 / fps) * np.linspace(0, T, track_len)
                initial_trajectory = Trajectory3D(time=t, position=t)
                out_trajectory = self.arc_optimizer.loss_function.model.simulate(initial_trajectory, best_motion_parameters)
                xt, yt, zt = out_trajectory.to_xyz()
                err = err_arc
                trajectory_type = "arc"
                k = k3_best
                vx = vx0_best
                vy = vy0_best
                vz = vz0_best
                if self.verbose:
                    print(
                        "\n{}decision: Arc trajectory is better\n".format(pfx)
                        + "{}x_end={:.1f}; y_end={:.1f}; z_end={:.1f}\n".format(
                            pfx, xt[-1], yt[-1], zt[-1]
                        )
                    )
            elif x_start_best is not None:
                best_motion_parameters = StraightMotionParameters(k_best, x_start_best, y_start_best, x_end_best, y_end_best)
                T = track_len - 1
                t = np.linspace(0, T, track_len)
                # TODO: clean this up
                initial_trajectory = Trajectory3D(time=t, position=t)
                #  TODO: use trajectory 3d
                out_trajectory = self.straight_optimizer.loss_function.model.simulate(initial_trajectory, best_motion_parameters)
                xt, yt, zt = out_trajectory.to_xyz()
                err = err_straight
                trajectory_type = "straight"
                g_best = None
                k = k_best
                vx = (
                    float(self.video_metadata["fps"])
                    * (x_end_best - x_start_best)
                    / (end_frame - start_frame)
                )
                vy = (
                    float(self.video_metadata["fps"])
                    * (y_end_best - y_start_best)
                    / (end_frame - start_frame)
                )
                vz = 0
                if self.verbose:
                    print(
                        "\n{}decision: Straight trajectory is better\n".format(pfx)
                    )

            # Check if the selected trajectory has a low enough fitting error to finally accept it
            if (
                self.relative_error(err, start_frame, end_frame)
                > self.allowed_detection_error
            ):
                trajectory_type = "not_fitted"
                xt, yt, zt = [None] * 3
                best_motion_parameters = None
                if self.verbose:
                    print(
                        "\n{}skipped: detection error too big;".format(pfx)
                        + " trajectory detection failed"
                    )
        err = self.relative_error(err, start_frame, end_frame)

        return xt, yt, zt, err, g_best, trajectory_type, k, vx, vy, vz, best_motion_parameters


    def predict_one_trajectory_segment(
        self,
        df_trajectory: pd.DataFrame,
        start_frame: int,
        end_frame: int,
        endpoint: bool,
    ) -> None:
        """
        Writes prediction of one trajectory segment to the trajectory file.

        Parameters:
            df_trajectory (pd.DataFrame): DataFrame containing per-frame ball detections
                together with detected pivot points frames.
            start_frame (int): Index of the start frame of the trajectory to predict.
            end_frame (int): Index of the end frame of the trajectory to predict.
            dynamic_pivots (bool): Flag enabling dynamic pivot points.
            endpoint (bool): Flag enabling trajectory endpoint fitting.
        """
        print(f"fitting frames {start_frame} -> {end_frame}")
        (xt, yt, zt, err, g_best, trajectory_type, k, vx, vy, vz, best_motion_parameters) = (
            self.fit_one_trajectory_segment(
                df_trajectory, start_frame, end_frame, endpoint
            )
        )
        # TODO: clean this up
        for key, value in {
            "g": g_best,
            "type": trajectory_type,
            "mean_error": err,
            "k": k,
            "vx": vx,
            "vy": vy,
            "vz": vz,
        }.items():
            df_trajectory.loc[
                (df_trajectory.file_name > start_frame)
                & (df_trajectory.file_name < end_frame),
                key,
            ] = value

        for key, value in {
            "x_predicted": xt,
            "y_predicted": yt,
            "z_predicted": zt,
        }.items():
            df_trajectory.loc[
                (df_trajectory.file_name >= start_frame)
                & (df_trajectory.file_name <= end_frame),
                key,
            ] = value

        # TODO: This is not super clean, but works like this:
        # checks that best_motion_parameters is a list with all elements of the same type
        # then for each parameter prepares a list of values and writes it to the trajectory df
        if isinstance(best_motion_parameters, list):
            if best_motion_parameters[0] is not None:
                parameter_dict_keys = best_motion_parameters[0].to_dict().keys()
                parameter_dict_type = type(best_motion_parameters[0])
                if all(isinstance(parameter_dict, parameter_dict_type) for parameter_dict in best_motion_parameters):
                    for key in parameter_dict_keys:
                        values = [parameter_dict.to_dict()[key] if parameter_dict is not None else np.nan for parameter_dict in best_motion_parameters]
                        df_trajectory.loc[
                            (df_trajectory.file_name > start_frame)
                            & (df_trajectory.file_name < end_frame),
                            key,
                        ] = values
        else:
            if best_motion_parameters is not None:
                for key, value in best_motion_parameters.to_dict().items():
                    df_trajectory.loc[
                        (df_trajectory.file_name > start_frame)
                        & (df_trajectory.file_name < end_frame),
                        key,
                    ] = value
            else:
                warnings.warn("best_motion_parameters has not been set")


    def estimate_trajectory(
        self, df: pd.DataFrame, ball_pivot_point: pd.DataFrame, *args, **kwargs
    ) -> pd.DataFrame:
        """Estimate 3D ball trajectory using homography transformation

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing per-frame ball detections together with
            smoothed ball coordinates, homography estimation and ball-player contacts
        ball_pivot_point: pd.DataFrame
            DataFrame containing per-frame pivot point probabilities
            and pivot point detection
        args
        kwargs

        Returns
        -------
        df_trajectory: pd.DataFrame
            DataFrame containing per-frame ball detections together with
            predicted ball X, Y, Z coordinates and model fitting errors
        """
        df["a"] = (df.x1 - df.x0) / 2
        df["b"] = (df.y1 - df.y0) / 2

        # Calculate the frame-wise diagonal of the ball bounding box,
        # where 'a' and 'b' represent the ball ellipse axes
        df["size"] = 2 * np.linalg.norm(df[["a", "b"]], axis=1)

        initial_pivot_points_frames, df_trajectory = self.data_processor.prepare_pivots(
            df, ball_pivot_point
        )

        df_trajectory["x_predicted"] = np.nan
        df_trajectory["y_predicted"] = np.nan
        print(f"initial_pivot_points_frames = {initial_pivot_points_frames}")
        for track_num in range(len(initial_pivot_points_frames) - 1):
            start_frame = initial_pivot_points_frames[track_num]
            end_frame = initial_pivot_points_frames[track_num + 1]
            self.predict_one_trajectory_segment(
                df_trajectory,
                start_frame,
                end_frame,
                self.fit_endpoint,
            )
        df_trajectory["track_ind"] = (
            df_trajectory["type"].isin(["pivot_point"]).cumsum()
        )

        # Post-processing step: If the pivot point occurs during ball contact with a player's head,
        # the ball's image projection onto the pitch may be far from its actual (x, y) coordinates.
        # To address this, we adjust the ball's coordinates to match the player's feet coordinates.
        if self.use_high_pivots_correction:
            recalc = self.correct_high_pivots(df_trajectory)
            

        print("\ncorrecting wrong pivots trajectories:")
        recalc_w = 0
        if self.correct_wrong_pivots:
            updated_pivot_points_frames = df_trajectory[
                df_trajectory.type.isin(
                    ["pivot_point", "additional_pivot_point", "high_pivot_point"]
                )
            ]["file_name"].values

            (
                pivot_points_frames,
                wrong_pivot_frames,
                df_trajectory,
            ) = self.data_processor.prepare_wrong_pivots(df_trajectory)

            print(f"wrong pivot frames: {wrong_pivot_frames}")
            if self.verbose:
                print(
                    "\n\n"
                    + "=" * 60
                    + "\nThird stage: correcting wrong pivots trajectories\n"
                    + "=" * 60
                    + "\n\n"
                )
            self.correct_wrong_pivots_trajectories(df_trajectory, wrong_pivot_frames,
                                                   updated_pivot_points_frames)
        if self.use_shots_fit:
            self.fit_shots(df_trajectory)
        self.data_processor.add_z_corrected(df_trajectory)
        print("")

        if self.verbose:
            if "type_initial" not in df_trajectory.columns:
                df_trajectory["type_initial"] = df_trajectory["type"]

            pp = len(df_trajectory[df_trajectory.type_initial == "pivot_point"])
            app = len(
                df_trajectory[df_trajectory.type_initial == "additional_pivot_point"]
            )
            tr = pp + app - 1
            pp_pp = len(
                df_trajectory[
                    (df_trajectory.type_initial == "pivot_point")
                    & (df_trajectory.type == "pivot_point")
                ]
            )
            pp_hpp = len(
                df_trajectory[
                    (df_trajectory.type_initial == "pivot_point")
                    & (df_trajectory.type == "high_pivot_point")
                ]
            )
            pp_del = len(
                df_trajectory[
                    (df_trajectory.type_initial == "pivot_point")
                    & (~df_trajectory.type.isin(["pivot_point", "high_pivot_point"]))
                ]
            )
            app_pp = len(
                df_trajectory[
                    (df_trajectory.type_initial == "additional_pivot_point")
                    & (df_trajectory.type == "pivot_point")
                ]
            )
            app_hpp = len(
                df_trajectory[
                    (df_trajectory.type_initial == "additional_pivot_point")
                    & (df_trajectory.type == "high_pivot_point")
                ]
            )
            app_del = len(
                df_trajectory[
                    (df_trajectory.type_initial == "additional_pivot_point")
                    & (~df_trajectory.type.isin(["pivot_point", "high_pivot_point"]))
                ]
            )

            print(
                "\n"
                + "*" * 60
                + f"\ninitial pivot points number: {pp}\n"
                + f"-> {pp_pp} stay pivot points\n"
                + f"-> {pp_hpp} become high pivot points\n"
                + f"-> {pp_del} removed from pivot points\n\n"
                + f"dynamic pivot points added: {app}\n"
                + f"-> {app_pp} stay pivot points\n"
                + f"-> {app_hpp} become high pivot points\n"
                + f"-> {app_del} removed from pivot points\n\n"
                + f"trajectories initial: {tr}\n"
                + f"trajectories re-calculated: {recalc}\n"
                + f"wrong pivots trajectories re-calculated: {recalc_w}\n"
                + "*" * 60
            )

        if "shot" not in df_trajectory.columns:
            df_trajectory["shot"] = None

        return df_trajectory

    def correct_high_pivots(self, df_trajectory):
        recalc = 0
        updated_pivot_points_frames = df_trajectory[
                    df_trajectory.type.isin(["pivot_point", "additional_pivot_point"])
                ]["file_name"].values

        (
            pivot_points_frames,
            high_pivot_frames,
            df_trajectory,
        ) = self.data_processor.prepare_high_pivots(df_trajectory)
        print("\ncorrecting high pivots trajectories:")
        print(f"high pivot frames: {high_pivot_frames}")
        if self.verbose:
            print(
                "\n\n"
                + "=" * 60
                + "\nSecond stage: correcting high pivots trajectories\n"
                + "=" * 60
                + "\n\n"
            )
        for track_num in range(len(pivot_points_frames) - 1):
            start_frame = pivot_points_frames[track_num]
            end_frame = pivot_points_frames[track_num + 1]
            if (
                (start_frame not in high_pivot_frames)
                and (end_frame not in high_pivot_frames)
                and (
                    list(updated_pivot_points_frames).index(end_frame)
                    - list(updated_pivot_points_frames).index(start_frame)
                )
                == 1
            ):
                df_trajectory.loc[
                    (df_trajectory.file_name > start_frame)
                    & (df_trajectory.file_name < end_frame),
                    "type",
                ] = df_trajectory.loc[
                    (df_trajectory.file_name > start_frame)
                    & (df_trajectory.file_name < end_frame),
                    "type_initial",
                ]
            else:
                self.predict_one_trajectory_segment(
                    df_trajectory,
                    start_frame,
                    end_frame,
                    endpoint=self.fit_endpoint,
                )
                recalc += 1
        df_trajectory["track_ind"] = (
            df_trajectory["type"].isin(["pivot_point", "high_pivot_point"]).cumsum()
        )
        return recalc

    def correct_wrong_pivots(self, df_trajectory, pivot_points_frames,
                             updated_pivot_points_frames):
        for track_num in range(len(pivot_points_frames) - 1):
            start_frame = pivot_points_frames[track_num]
            end_frame = pivot_points_frames[track_num + 1]
            if (
                list(updated_pivot_points_frames).index(end_frame)
                - list(updated_pivot_points_frames).index(start_frame)
            ) != 1:
                self.predict_one_trajectory_segment(
                    df_trajectory,
                    start_frame,
                    end_frame,
                    endpoint=self.fit_endpoint,
                )
                recalc_w += 1
        df_trajectory["track_ind"] = (
            df_trajectory["type"].isin(["pivot_point", "high_pivot_point"]).cumsum()
        )

    def fit_shots(self, df_trajectory):
        if self.verbose:
            print(
                "\n\n"
                + "=" * 60
                + "\nFourth stage: shots trajectories fit\n"
                + "=" * 60
                + "\n\n"
            )
        df_tracks = (
            df_trajectory.groupby("track_ind")
            .agg({"file_name": ["min", "max"]})
            .reset_index()
        )
        df_tracks.columns = ["track_ind", "track_start", "track_end"]
        df_nans = df_trajectory[
            pd.isna(df_trajectory["x_pitch2D"])
            & df_trajectory["type"].isin(["too_long_track", "NANs_inside_track"])
        ]
        df_nans = df_nans.groupby("track_ind").agg({"file_name": min}).reset_index()
        df_nans.columns = ["track_ind", "first_nan_ind"]
        df_nans = df_nans.merge(df_tracks, on=["track_ind"], how="left")
        df_nans = df_nans.loc[
            df_nans["first_nan_ind"]
            > df_nans["track_start"] + self.too_short_track + 1
        ].reset_index(drop=True)
        for start_frame, end_frame in zip(
            df_nans["track_start"].values, (df_nans["first_nan_ind"] - 1).values
        ):
            self.predict_one_trajectory_segment(
                df_trajectory,
                start_frame,
                end_frame,
                dynamic_pivots=False,
                endpoint=False,
            )
            df_trajectory.loc[
                (df_trajectory["file_name"] >= start_frame)
                & (df_trajectory["file_name"] <= end_frame),
                "shot",
            ] = True

    def print_segment_info(self, df_trajectory, pfx, start_frame, end_frame, track_len):
        print(
            pfx
            + "-" * 60
            + "\n{}handle track from frames {} to {} containing {} frames\n".format(
                pfx, start_frame, end_frame, track_len
            )
        + "{}xk_start = {:.1f}; xk_end = {:.1f};".format(
            pfx,
            df_trajectory[df_trajectory.file_name == start_frame]["xk"].values[
                0
            ],
            df_trajectory[df_trajectory.file_name == end_frame]["xk"].values[0],
        )
        + " x_pitch2D_start = {:.1f}; x_pitch2D_end = {:.1f}\n".format(
            df_trajectory[df_trajectory.file_name == start_frame][
                "x_pitch2D"
            ].values[0],
            df_trajectory[df_trajectory.file_name == end_frame][
                "x_pitch2D"
            ].values[0],
        )
        + "{}yk_start = {:.1f}; yk_end = {:.1f};".format(
            pfx,
            df_trajectory[df_trajectory.file_name == start_frame]["yk"].values[
                0
            ],
            df_trajectory[df_trajectory.file_name == end_frame]["yk"].values[0],
        )
        + " y_pitch2D_start = {:.1f}; y_pitch2D_end = {:.1f}".format(
            df_trajectory[df_trajectory.file_name == start_frame][
                "y_pitch2D"
            ].values[0],
            df_trajectory[df_trajectory.file_name == end_frame][
                "y_pitch2D"
            ].values[0],
        )
    )
