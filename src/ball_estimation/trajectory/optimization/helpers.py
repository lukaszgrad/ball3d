from copy import deepcopy
from typing import Tuple
import numpy as np
import pandas as pd
from ball_estimation.trajectory import Trajectory3D
from ball_estimation.trajectory.model import ArcMotionParameters, StraightMotionParameters
from ball_estimation.trajectory.optimization import ArcOptimizer, StraightOptimizer

TEMPLATE_PITCH_SHAPE = (780, 1150)



def relative_error(err, start_frame, end_frame, use_relative_error=True):
    if use_relative_error and err < 1e8:
        return 25 * err / (end_frame - start_frame + 1) # 25 is fps or what?
    return err


def fit_ball_straight_move_model(
    df_ball: pd.DataFrame, start_frame: int, end_frame: int, 
    straight_optimizer: StraightOptimizer,
    pfx: str = "",
    verbose=False,
    out_margin=100,
) -> Tuple[float, float, float, float, float, float]:
    """Fits the ball straight move model for a given set of frames

    Parameters
    ----------
    df_ball: pd.DataFrame
        DataFrame containing per-frame ball detections together with
        homography estimation and pivot point detections
    start_frame, end_frame: int
        indexes of starting and ending frames of sequence to fit the model
    pfx: str
        prefix for prints

    Returns
    -------
    err: float
        RMSE of fitted model (actual trajectory on panorama vs. predicted trajectory)
    x_start_best, y_start_best: float
        best starting ball X,Y coordinates found by model
    x_end_best, y_end_best: float
        best ending ball X,Y coordinates found by model
    k_best: float
        grass resistance coefficient of the best trajectory
    """

    # Initial coordinates of the ball's starting and ending points,
    # used as the initial values for optimization.
    xh_start, yh_start = df_ball[df_ball.file_name == start_frame][
        ["x_pitch2D", "y_pitch2D"]
    ].values[0]
    xh_end, yh_end = df_ball[df_ball.file_name == end_frame][
        ["x_pitch2D", "y_pitch2D"]
    ].values[0]
    num_points = end_frame - start_frame + 1

    # Ball coordinates on a screen, whether from a broadcast image or a panorama image,
    # depending on the estimator settings.
    # These coordinates are used for calculating trajectory fitting errors.
    real_coords = ["xk", "yk"]
    # The fitting loss could be relative to the ball size,
    # so we include a "size" column in the dataframe as well.
    data = df_ball[
        (df_ball.file_name >= start_frame) & (df_ball.file_name <= end_frame)
    ][real_coords + ["size"]].values
    straight_optimizer.optimization_bounds.update(xh_start, yh_start, xh_end, yh_end)

    k = 0
    motion_parameters = StraightMotionParameters(k, xh_start, yh_start, xh_end, yh_end)

    dif, err, k_best, x_start_best, y_start_best, x_end_best, y_end_best = straight_optimizer.optimize(motion_parameters, (num_points, data, start_frame))

    if verbose:
        print_straight(dif, err, k_best, x_start_best, y_start_best, x_end_best, y_end_best, pfx, xh_start, yh_start,
            xh_end, yh_end, start_frame, end_frame)
    # Calculate the optimal straight trajectory.
    best_motion_parameters = StraightMotionParameters(k_best, x_start_best,
                                                        y_start_best, x_end_best,
                                                        y_end_best)
    T = num_points - 1
    t = np.linspace(0, T, num_points)
    # TODO: clean this up
    initial_trajectory = Trajectory3D(time=t, position=t)
    if best_motion_parameters.x_start is not None:
        out_trajectory = straight_optimizer.loss_function.model.simulate(initial_trajectory, best_motion_parameters)
        xt, yt, _ = out_trajectory.to_xyz()
    else:
        xt = np.nan * t
        yt = np.nan * t

    out_of_pitch = (
        (min(xt) < -out_margin)
        or (max(xt) > TEMPLATE_PITCH_SHAPE[1] + out_margin)
        or (min(yt) < -out_margin)
        or (max(yt) > TEMPLATE_PITCH_SHAPE[0] + out_margin)
    )
    # Typically, the ball cannot roll far out of the pitch while moving on the turf
    # due to the barriers. Therefore, if we encounter something like this
    # in the fitted trajectory, we consider it incorrect.
    if out_of_pitch:
        err = 1e9
        if verbose:
            print(f"{pfx}skipped: trajectory contains points outside pitch")

    if err is None:
        err = 1e9
        if verbose:
            print(f"{pfx}skipped: optimization failed")
    return err, x_start_best, y_start_best, x_end_best, y_end_best, k_best

def fit_ball_arc_move_model(
    df_ball: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    endpoint: bool,
    arc_optimizer: ArcOptimizer,
    fps: float,
    verbose=False,
    pfx: str = "",
) -> Tuple[
    float, ArcMotionParameters
]:
    """
    Fits the ball parabolic move model for a given set of frames.

    Parameters
    ----------
    df_ball : pd.DataFrame
        DataFrame containing per-frame ball detections together with
        homography estimation and pivot point detections.
    start_frame : int
        Index of the starting frame of the sequence to fit the model.
    end_frame : int
        Index of the ending frame of the sequence to fit the model.
    endpoint : bool
        Flag enabling trajectory endpoint fitting.
    pfx : str, optional
        Prefix for prints.

    Returns
    -------
    err : float
        RMSE of the fitted model (actual trajectory on broadcast/panorama vs. predicted trajectory).
    x0_best, y0_best, z0_best : float
        Best starting ball X, Y, Z coordinates found by the optimizer.
    vx0_best, vy0_best, vz0_best : float
        Best starting ball X, Y, Z velocity found by the optimizer.
    k3_best : float
        Air resistance coefficient of the best trajectory.
    g_best : float
        Gravitation coefficient of the best trajectory.
    kl_best, ks_best : float
        Magnus effect's coefficients of the best trajectory.
    """

    # Initial coordinates of the ball's starting and ending points,
    # which are used to calculate an initial approximation of the ball's starting velocity.
    xh_start, yh_start = df_ball[df_ball.file_name == start_frame][
        ["x_pitch2D", "y_pitch2D"]
    ].values[0]
    xh_end, yh_end = df_ball[df_ball.file_name == end_frame][
        ["x_pitch2D", "y_pitch2D"]
    ].values[0]
    num_points = end_frame - start_frame + 1

    # Ball coordinates on a screen, whether from a broadcast image or a panorama image,
    # depending on the estimator settings.
    # These coordinates are used for calculating trajectory fitting errors.
    real_coords = ["xk", "yk"]

    # The fitting loss could be relative to the ball size,
    # so we include a "size" column in the dataframe as well.
    data = df_ball[
        (df_ball.file_name >= start_frame) & (df_ball.file_name <= end_frame)
    ][real_coords + ["size"]].values
    arc_optimizer.loss_function.ball_median_size = df_ball[
        (df_ball.file_name >= start_frame) & (df_ball.file_name <= end_frame)
    ]["size"].median()

    # find best parameters
    fly_time = (end_frame - start_frame) / fps
    arc_optimizer.optimization_bounds.build_bounds()
    arc_optimizer.optimization_bounds.update(xh_start, yh_start)
    motion_parameters_estimate = deepcopy(
        arc_optimizer.optimization_bounds.lower_bounds)
    motion_parameters_estimate.estimate_from_bounds(arc_optimizer.optimization_bounds, xh_end, yh_end, fly_time)

    (
    dif,
    err,
    motion_parameters_estimate
    ) = arc_optimizer.optimize(motion_parameters_estimate, (num_points, data, xh_end, yh_end, start_frame, 
                                                            endpoint))

    if verbose:
        print_arc(arc_optimizer, motion_parameters_estimate, dif, pfx, xh_start, yh_start,
                xh_end, yh_end, fly_time, err, data, start_frame, end_frame)

    return (
        err,
        motion_parameters_estimate
    )

def print_arc(arc_optimizer, motion_parameters_estimate, dif, pfx, xh_start, yh_start,
            xh_end, yh_end, fly_time, err, data, start_frame, end_frame):
    x0_best = motion_parameters_estimate.x0
    y0_best = motion_parameters_estimate.y0
    z0_best = motion_parameters_estimate.z0
    vx0_best = motion_parameters_estimate.vx0
    vy0_best = motion_parameters_estimate.vy0
    vz0_best = motion_parameters_estimate.vz0
    k3_best = motion_parameters_estimate.k3
    g_best = motion_parameters_estimate.g
    kl_best = motion_parameters_estimate.kl
    ks_best = motion_parameters_estimate.ks

    print(
        f"\n{pfx}arc move model:\n"
        + (
            "{}x0_init = {:.1f}; y0_init = {:.1f}; z0_init = {:.1f};"
            + " vx0_init = {:.1f}; vy0_init = {:.1f}; vz0_init = {:.1f}\n"
        ).format(
            pfx,
            xh_start,
            yh_start,
            0,
            (xh_end - xh_start) / fly_time,
            (yh_end - yh_start) / fly_time,
            10 * fly_time * arc_optimizer.optimization_bounds.gravitation_g / 2,
        )
        + (
            "{}x0_best = {:.1f}; y0_best = {:.1f}; z0_best = {:.1f};"
            + " vx0_best = {:.1f}; vy0_best = {:.1f}; vz0_best = {:.1f};"
            + " k3_best = {:.3f}; g_best = {:.1f}; kl_best = {:.3f}; ks_best = {:.3f}\n"
        ).format(
            pfx,
            x0_best,
            y0_best,
            z0_best,
            vx0_best,
            vy0_best,
            vz0_best,
            k3_best,
            g_best,
            kl_best,
            ks_best,
        )
        + "{}raw mean error = {:.2f}; relative error = {:.2f}".format(
            pfx, err, arc_optimizer.relative_error(err, start_frame, end_frame)
        )
    )
    add_end = int(arc_optimizer.loss_function.weight_end_arc * len(data) + 0.5)
    err_a = (dif[: len(data), 0] ** 2 + dif[: len(data), 1] ** 2).sum()
    err_b = (
        dif[len(data): len(data) + add_end, 0] ** 2
        + dif[len(data): len(data) + add_end, 1] ** 2
    ).sum()
    err_c = (
        dif[len(data) + add_end:, 0] ** 2 + dif[len(data) + add_end:, 1] ** 2
    ).sum()
    print(
        "{}error terms: trajectory_fit {}x({:.4f})^2 = {:0,.4f};".format(
            pfx, len(data), np.sqrt(err_a / len(data)), err_a
        )
        + " endpoint_fit {}x({:.4f})^2 = {:0,.4f};".format(
            add_end, np.sqrt(err_b / add_end), err_b
        )
        + " z_end penalty {}x({:.4f})^2 = {:0,.4f}".format(
            len(dif) - len(data) - add_end,
            np.sqrt(err_c / (len(dif) - len(data) - add_end)),
            err_c,
        )
    )

def print_straight(dif, err, k_best, x_start_best, y_start_best, x_end_best, y_end_best, pfx, xh_start, yh_start,
            xh_end, yh_end, start_frame, end_frame):
    print(
            f"\n{pfx}straight move model:\n"
            + (
                "{}x_start_init = {:.1f}; y_start_init = {:.1f};"
                + " x_end_init = {:.1f}; y_end_init = {:.1f}\n"
            ).format(pfx, xh_start, yh_start, xh_end, yh_end)
            + (
                "{}x_start_best = {:.1f}; y_start_best = {:.1f};"
                + " x_end_best = {:.1f}; y_end_best = {:.1f}; k_best = {:.4f}\n"
            ).format(
                pfx, x_start_best, y_start_best, x_end_best, y_end_best, k_best
            )
            + "{}raw mean error = {:.4f}; relative error = {:.4f}".format(
                pfx, err, relative_error(err, start_frame, end_frame)
            )
        )