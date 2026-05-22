from abc import ABC
from typing import List, Tuple
from pydantic import BaseModel, NonNegativeFloat

import numpy as np
from tqdm import tqdm
import pandas as pd
import json
from lmfit import Minimizer, Parameters
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

from hydra.utils import instantiate

from ball_estimation.trajectory import Trajectory3D


def load_gt(
    coords_path: str, pivot_path: str, metadata_path: str, pauses_path: str, splits_path: str
) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    """Load ground truth data."""
    df = pd.read_csv(coords_path)
    df2 = pd.read_csv(pivot_path)
    pivot_points = df2["pivot_point"].values  # 1 if pivot, 0 otherwise
    pivot_indices = df2["file_name"].values
    pivot_dict = {
        ind: pivot for ind, pivot in zip(pivot_indices, pivot_points)
    }
    coord_indices = df["file_name"].values
    # load coordinates
    r = np.zeros((len(coord_indices), 3), dtype=np.float64)
    if "x" in df.keys():
        r[:, 0] = df["x"].values# / 10  # dm to m
        r[:, 1] = df["y"].values# / 10  # dm to m
        r[:, 2] = df["z"].values# / 10  # dm to m
    elif "x_predicted" in df.keys():
        r[:, 0] = df["x_predicted"].values# / 10  # dm to m
        r[:, 1] = df["y_predicted"].values# / 10  # dm to m
        r[:, 2] = df["z_predicted"].values# / 10  # dm to m
    else:
        raise ValueError("Something is wrong with the groundtruth data")
    coord_dict = {ind: coord for ind, coord in zip(coord_indices, r)}

    # get fps from metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        fps = float(metadata["fps"])

    # load pauses
    #directly annotated pauses
    try:
        df3 = pd.read_csv(pauses_path)
        starts = df3["start_pause"].values
        ends = df3["end_pause"].values
        pauses = [(int(start), int(end)) for start, end in zip(starts, ends)]
    except FileNotFoundError:
        pauses = []
    # pauses inferred from the splits.csv file
    try:
        df4 = pd.read_csv(splits_path)
        frames = df4["frame_index"].values
        max_frame = frames[-1]
        new_pauses = []
        start, end = None, None
        for f in range(max_frame + 1):
            is_pause = False
            for pause in pauses:
                s, e = pause
                if s <= f <= e:
                    is_pause = True
            if f not in frames:
                is_pause = True
            if is_pause:
                if start is None:
                    start = f
                end = f
            else:
                if end is not None:
                    new_pauses.append((start, end))
                    start, end = None, None
        if start is not None and end is not None:  # if the last frame is a pause
            new_pauses.append((start, end))
    except FileNotFoundError:
        new_pauses = []
        print("No splits file found. Using only the normal pauses.csv, not the splits.csv for inferring the pauses!!!")
    # Update the pauses list with the new pauses
    pauses = new_pauses

    trajectories, trajectories_indices = [], []
    pivot_indices, pivot_points = pivot_indices[pivot_points == 1], pivot_points[pivot_points == 1]
    # iterate over all pivots
    for i in range(len(pivot_indices)-1):
        ind, next_ind = pivot_indices[i], pivot_indices[i+1]
        trajectory, indices = [], []
        # iterate over all frames between two pivot points
        for j in range(ind, next_ind):
            # iterate over all pauses and check if the current frame is a pause
            is_pause = False
            for pause in pauses:
                if pause[0] <= j <= pause[1]:
                    is_pause = True
            # if the current frame is not a pause, add it to the trajectory
            if not is_pause:
                if j in coord_dict.keys() and coord_dict[j] is not None:
                    trajectory.append(coord_dict[j])
                    indices.append(j)
        # if the trajectory is not empty, add it to the list of trajectories
        if len(trajectory) >= 5: # This number is arbitrary. We should not allow very short trajectories, since we fit a lot of parameters...
            trajectories.append(np.array(trajectory))
            trajectories_indices.append(np.array(indices))
    min_length = min([len(traj) for traj in trajectories])
    print("Minimum Trajectory length:", min_length)

    return trajectories, trajectories_indices, fps


def fit_trajectory(
    r_gt: np.ndarray, indices, fps: float,
    model: BaseModel, fit_params: dict, fixed_params: dict, variations: dict,
    arc: bool = True,):
    """Fit a trajectory to the given data.
    Parameters
    ----------
    r_gt : np.ndarray
        Ground truth position of the ball. Shape (n, 3). Units: m.
    t : np.ndarray
        Time points. Shape (n,). Units: s.
    model : BaseModel
        Model to fit the trajectory. It has the method get_arc/straight_trajectory.
    fit_params : dict
        Parameters to fit as keys and their initial values as values.
    fixed_params : dict
        Parameters that are fixed as keys and their values as values.
    variations : dict
        Allowed variations for the parameters.
    arc : bool
        If True, fit an arc trajectory. If False, fit a straight trajectory.
    """
    def fcn2min(params):
        function_params = {key: params[key].value for key in params.keys()}
        if arc is True:
            motion_params = model.motion_parameters_type(**function_params)
            initial_trajectory = Trajectory3D.from_xyz(r_gt[:, 0], r_gt[:, 1], r_gt[:, 2], num_frames=len(indices), fps=fps)
            res = model.simulate(initial_trajectory, motion_params)
            xt, yt, zt = res.position[:, 0], res.position[:, 1], res.position[:, 2]
            pass
        else:
            xt, yt, zt = model.get_straight_trajectory(num_points=len(indices), **function_params)
        r = np.array([xt, yt, zt]).T
        r_filtered, r_gt_filtered = (
            r[~np.isnan(r_gt).any(axis=1)],
            r_gt[~np.isnan(r_gt).any(axis=1)],
        )
        return np.abs(r_filtered - r_gt_filtered)

    # implement fittable parameters
    params = Parameters()
    for key, value in fit_params.items():
        if key in variations.keys() or key == 'k':
            if key == 'k':
                min_val = 0
                max_val = 0.15
            elif key == 'k3':
                min_val = max(0., value - variations[key])
                max_val = value + variations[key] + 1e-6
            else:
                min_val = value - variations[key] # if key not in ['k3', 'k', 'km'] else 0
                max_val = value + variations[key] + 1e-6
            params.add(key, value=value, min=min_val, max=max_val, vary=True)
        else:
            min_val = -np.inf # if key not in ['k3', 'k', 'km'] else 0
            params.add(key, value=value, min=min_val, vary=True)
    for key, value in fixed_params.items():
        params.add(key, value=value, vary=False)
    if arc:
        if 'allowed_endpoints_shift_straight' in variations.keys():
            shift = variants['allowed_endpoints_shift_straight']
        else:
            shift = 10000 # let's set some rational, but large value as a default
        params.add("x0", value=r_gt[0, 0], min=r_gt[0, 0] - shift, max=r_gt[0, 0] + shift, vary=True)  # I could maybe use the detection coordinates to estimate the allowed variations
        params.add("y0", value=r_gt[0, 1], min=r_gt[0, 1] - shift, max=r_gt[0, 1] + shift, vary=True)  # I could maybe use the detection coordinates to estimate the allowed variations
        params.add("z0", value=r_gt[0, 2], min=r_gt[0, 2] - shift, max=r_gt[0, 2] + shift, vary=True)
        v = (r_gt[1] - r_gt[0]) * fps
        max_ball_speed = variations['max_ball_speed']
        params.add("vx0", value=v[0], vary=True, min=-max_ball_speed, max=max_ball_speed)
        params.add("vy0", value=v[1], vary=True, min=-max_ball_speed, max=max_ball_speed)
        params.add("vz0", value=v[2], vary=True, min=-0.5*max_ball_speed, max=0.5*max_ball_speed)
    else:
        if 'allowed_endpoints_shift_straight' in variations.keys():
            shift = variations['allowed_endpoints_shift_straight']
            params.add("x_start", value=r_gt[0, 0], min=r_gt[0, 0] - shift, max=r_gt[0, 0] + shift, vary=True)
            params.add("y_start", value=r_gt[0, 1], min=r_gt[0, 1] - shift, max=r_gt[0, 1] + shift, vary=True)
            params.add("x_end", value=r_gt[-1, 0], min=r_gt[-1, 0] - shift, max=r_gt[-1, 0] + shift, vary=True)
            params.add("y_end", value=r_gt[-1, 1], min=r_gt[-1, 1] - shift, max=r_gt[-1, 1] + shift, vary=True)
        else:
            params.add("x_start", value=r_gt[0, 0], vary=True)
            params.add("y_start", value=r_gt[0, 1], vary=True)
            params.add("x_end", value=r_gt[-1, 0], vary=True)
            params.add("y_end", value=r_gt[-1, 1], vary=True)

    # fit the model
    try:
        minner = Minimizer(fcn2min, params)
        result = minner.minimize()
    except ValueError:
        print("Error during fitting. Returning high error.")
        return 10e8, None, None, None
    if arc is True:
        r0 = np.array([result.params["x0"], result.params["y0"], result.params["z0"]])
        v0 = np.array([result.params["vx0"], result.params["vy0"], result.params["vz0"]])
    else:
        r_start = np.array([result.params["x_start"], result.params["y_start"], 0])
        r_end = np.array([result.params["x_end"], result.params["y_end"], 0])
    fitted_params = {key: result.params[key] for key in fit_params.keys()}

    # calculate metric error
    function_params = {key: result.params[key].value for key in result.params.keys()}
    if arc is True:
        motion_params = model.motion_parameters_type(**function_params)
        initial_trajectory = Trajectory3D.from_xyz(r_gt[:, 0], r_gt[:, 1], r_gt[:, 2], num_frames=len(indices), fps=fps)
        res = model.simulate(initial_trajectory, motion_params)
        xt, yt, zt = res.position[:, 0], res.position[:, 1], res.position[:, 2]
    else:
        xt, yt, zt = model.get_straight_trajectory(num_points=len(indices), **function_params)
    r = np.array([xt, yt, zt]).T
    r_filtered, r_gt_filtered = (
        r[~np.isnan(r_gt).any(axis=1)],
        r_gt[~np.isnan(r_gt).any(axis=1)],
    )
    error = np.mean(np.linalg.norm(r_filtered - r_gt_filtered, axis=1, ord=2))
    # print("Error:", error)

    if arc is True:
        return error, r0, v0, fitted_params
    else:
        return error, r_start, r_end, fitted_params


def shift_center(ball_path: str) -> bool:
    """Check if a file is used where the ball location has to be shifted from center to bottom.
    Parameters
    ----------
    ball_path : str
        path to the data folder, e.g. "data/legia-stal-80sec/clip"
    """
    relevant_filenames = ["stalowa_wola"]
    for file in relevant_filenames:
        if file in ball_path:
            return True
    return False


def plot_bins(
    data: List[float],
    min_val: float,
    max_val: float,
    num_bins: int,
    title: str,
    plt_idx: tuple = None,
):
    """Plot a histogram of the data. Don't forget to do plt.show() after calling this function!
    Parameters
    ----------
    data : List[float]
        List of data points.
    min_val : float
        Minimum value of the histogram.
    max_val : float
        Maximum value of the histogram.
    num_bins : int
        Number of bins.
    title : str
        Title of the plot.
    plt_idx : tuple
        Indeces of the subplot. E.g. (2, 2, 1) for a 2x2 grid and the first plot.
    """
    if plt_idx is None:
        plt.hist(data, bins=np.linspace(min_val, max_val, num_bins))
    else:
        plt.subplot(*plt_idx)
        plt.hist(data, bins=np.linspace(min_val, max_val, num_bins))
        plt.title(title)


def statistics_gt_trajectory(ball_path: str, plot: bool, arc_model_name: str, straight_model_name: str,
                             arc_optimization_bounds: dict, straight_optimization_bounds: dict,
                             arc_optimization_options: dict, straight_optimization_options: dict,
                             further_information: dict) -> dict:
    """Test ground truth data.
    Parameters
    ----------
    ball_path : str
        path to the data folder, e.g. "data/legia-stal-80sec/clip"
    plot : bool
        If True, plot the results. -> 3D plot of the fitted trajectory and the groundtruth trajectory.
    arc_model_name : str
        Path to the arc model as defined in the config file.
    straight_model_name : str
        Path to the straight model as defined in the config file.
    arc_optimization_bounds : dict
        Dict from config containing the start values (and allowed variations) for the arc model.
    straight_optimization_bounds : dict
        Dict from config containing the start values (and allowed variations) for the straight model.
    arc_optimization_options : dict
        Dict containing which parameters are optimzed and which are fixed.
    straight_optimization_options : dict
        Dict containing which parameters are optimzed and which are fixed.
    further_information : dict
        Dict containing further information about the model, e.g. use_ellipsoid_fluid parameter
    """
    coords_path = os.path.join(ball_path, "track", "ball_3d-gt.csv")
    pivot_path = os.path.join(ball_path, "track", "ball_pivot_point-gt.csv")
    if not os.path.exists(pivot_path):
        print("No ground truth pivot point file found. Using the normal one!!!")
        pivot_path = os.path.join(ball_path, "track", "ball_pivot_point.csv")
    metadata_path = os.path.join(ball_path, "sequence_metadata.json")
    pauses_path = os.path.join(ball_path, "dev", "pauses.csv")  # optional
    split_path = os.path.join(ball_path, "track", "split.csv")
    if os.path.exists(split_path):
        split_df = pd.read_csv(split_path)
    else:
        split_df = None
        print('split.csv was not found. You should investigate this!!! Run calculate_splits.py before this script!!!')
    trajectories, trajectories_indices, fps = load_gt(
        coords_path, pivot_path, metadata_path, pauses_path, split_path
    )

    # instantiate models
    arc_model = instantiate({'_target_': arc_model_name})
    if 'use_ellipsoid_fluid' in further_information.keys():
        arc_model.use_ellipsoid_fluid = further_information['use_ellipsoid_fluid']
    straight_model = instantiate({'_target_': straight_model_name})
    # create dicts with the parameters that should be fitted and that are fixed
    fit_params_arc = {key: 0.0 for key, value in arc_optimization_options.items() if value is True}
    fixed_params_arc = {key: 0.0 for key, value in arc_optimization_options.items() if value is False}
    fit_params_straight = {key: 0.0 for key, value in straight_optimization_options.items() if value is True}
    fixed_params_straight = {key: 0.0 for key, value in straight_optimization_options.items() if value is False}
    # get th allowed variations for fitting
    ball_radius = 0.11
    DM = 1 / 10
    variations_arc = {}
    for key in fit_params_arc.keys():
        if f'{key}_variation' in arc_optimization_bounds.keys():
            variations_arc[key] = arc_optimization_bounds[f'{key}_variation']
        elif key in ['kl', 'ks'] and 'kl_ks_variation' in arc_optimization_bounds.keys():
            variations_arc[key] = arc_optimization_bounds['kl_ks_variation']
        elif key in ['wx0', 'wy0', 'wz0'] and 'angular_velocity_factor' in arc_optimization_bounds.keys():
            scale = (arc_optimization_bounds['angular_velocity_factor'] / ball_radius)
            variations_arc[key] = scale * arc_optimization_bounds['max_ball_speed'] * DM
    variations_arc['max_ball_speed'] = arc_optimization_bounds['max_ball_speed'] * DM
    variations_straight = {}
    for key in fit_params_straight.keys():
        if f'{key}_variation' in straight_optimization_bounds.keys():
            variations_straight[key] = straight_optimization_bounds[f'{key}_variation']
    variations_straight['allowed_endpoints_shift_straight'] = straight_optimization_bounds['allowed_endpoints_shift_straight']

    # Fill the dicts with the start values.
    for key in fit_params_arc.keys():
        if key in arc_optimization_bounds.keys():
            fit_params_arc[key] = arc_optimization_bounds[key]
    for key in fixed_params_arc.keys():
        if key in arc_optimization_bounds.keys():
            fixed_params_arc[key] = arc_optimization_bounds[key]
    for key in fit_params_straight.keys():
        if key in straight_optimization_bounds.keys():
            fit_params_straight[key] = straight_optimization_bounds[key]
    for key in fixed_params_straight.keys():
        if key in straight_optimization_bounds.keys():
            fixed_params_straight[key] = straight_optimization_bounds[key]
    # naming for g is weird in config: instead of g, it is gravitation_g
    if "gravitation_g" in arc_optimization_bounds.keys():
        if "g" in fixed_params_arc.keys():
            fixed_params_arc["g"] = arc_optimization_bounds["gravitation_g"]
        elif "g" in fit_params_arc.keys():
            fit_params_arc["g"] = arc_optimization_bounds["gravitation_g"]

    variations_arc["allowed_startpoint_shift_arc"] = arc_optimization_bounds["allowed_startpoint_shift_arc"]
    # create dict that saves the fitted results
    save_dict_arc = {key: [] for key in fit_params_arc.keys()}
    save_dict_arc.update({f'frames_{key}': [] for key in fit_params_arc.keys()})
    save_dict_arc['v0_abs'] = []
    save_dict_arc['frames_v0_abs'] = []
    # Since the kl and ks can be positive and negative, we are interested in the absolute statistics
    if 'kl' in fit_params_arc.keys():
        save_dict_arc['kl_abs'] = []
        save_dict_arc['frames_kl_abs'] = []
    if 'ks' in fit_params_arc.keys():
        save_dict_arc['ks_abs'] = []
        save_dict_arc['frames_ks_abs'] = []
    if 'wx0' in fit_params_arc.keys() and 'wy0' in fit_params_arc.keys() and 'wz0' in fit_params_arc.keys():
        save_dict_arc['w0_abs'] = []
        save_dict_arc['frames_w0_abs'] = []
    save_dict_straight = {key: [] for key in fit_params_straight.keys()}
    save_dict_straight.update({f'frames_{key}': [] for key in fit_params_straight.keys()})
    save_dict_straight['distance'] = []
    save_dict_straight['frames_distance'] = []
    save_dict_arc['frames_errors_arc'] = []
    save_dict_straight['frames_errors_straight'] = []

    errors_arc, errors_straight = [], []
    for indices, r_gt in tqdm(zip(trajectories_indices, trajectories)):
        error_arc, r0, v0, fitted_params_arc = fit_trajectory(r_gt, indices, fps, arc_model, fit_params_arc, fixed_params_arc, variations_arc, arc=True)
        error_straight, r_start, r_end, fitted_params_straight = fit_trajectory(r_gt, indices, fps, straight_model, fit_params_straight, fixed_params_straight, variations_straight, arc=False)

        if split_df is None:
            is_arc = error_arc < error_straight
        else:
            is_arc = split_df[split_df['frame_index'] == indices[0]]['is_arc'].values[0]

        # check if fitting failed
        if is_arc:
            if r0 is None or v0 is None or fitted_params_arc is None: continue
        else:
            if r_start is None or r_end is None or fitted_params_straight is None: continue

        if is_arc:  # arc is better
            for key in fitted_params_arc.keys():
                save_dict_arc[key].append(fitted_params_arc[key].value)
                save_dict_arc[f'frames_{key}'].append((int(indices[0]), int(indices[-1])))
            save_dict_arc['v0_abs'].append(np.linalg.norm(v0))
            save_dict_arc['frames_v0_abs'].append((int(indices[0]), int(indices[-1])))
            if 'kl' in fitted_params_arc.keys():
                save_dict_arc['kl_abs'].append(np.abs(fitted_params_arc['kl'].value))
                save_dict_arc['frames_kl_abs'].append((int(indices[0]), int(indices[-1])))
            if 'ks' in fitted_params_arc.keys():
                save_dict_arc['ks_abs'].append(np.abs(fitted_params_arc['ks'].value))
                save_dict_arc['frames_ks_abs'].append((int(indices[0]), int(indices[-1])))
            if 'wx0' in fitted_params_arc.keys() and 'wy0' in fitted_params_arc.keys() and 'wz0' in fitted_params_arc.keys():
                save_dict_arc['w0_abs'].append(np.linalg.norm([fitted_params_arc['wx0'], fitted_params_arc['wy0'], fitted_params_arc['wz0']]))
                save_dict_arc['frames_w0_abs'].append((int(indices[0]), int(indices[-1])))
            errors_arc.append(error_arc)
            save_dict_arc[f'frames_errors_arc'].append((int(indices[0]), int(indices[-1])))
        else:  # straight is better
            for key in fitted_params_straight.keys():
                save_dict_straight[key].append(fitted_params_straight[key].value)
                save_dict_straight[f'frames_{key}'].append((int(indices[0]), int(indices[-1])))
            save_dict_straight['distance'].append(np.linalg.norm(r_end - r_start))
            save_dict_straight['frames_distance'].append((int(indices[0]), int(indices[-1])))
            errors_straight.append(error_straight)
            save_dict_straight[f'frames_errors_straight'].append((int(indices[0]), int(indices[-1])))

    # evaluate the results
    print(f"Mean errors arc: {np.mean(errors_arc)}, Std errors arc: {np.std(errors_arc)}")
    print(f"Mean errors straight: {np.mean(errors_straight)}, Std errors straight: {np.std(errors_straight)}")

    for key in save_dict_arc.keys():
        print(f"Mean {key} arc: {np.mean(save_dict_arc[key])}, Std {key} arc: {np.std(save_dict_arc[key])}")
    for key in save_dict_straight.keys():
        print(f"Mean {key} straight: {np.mean(save_dict_straight[key])}, Std {key} straight: {np.std(save_dict_straight[key])}")

    print("Number of arcs:", len(errors_arc))
    print("Number of straights:", len(errors_straight))

    # if plot:
    #     length = len(save_dict_arc.keys()) + len(save_dict_straight.keys()) + 1
    #     for i, key in enumerate(save_dict_arc.keys()):
    #         plot_bins(save_dict_arc[key], np.min(save_dict_arc[key]), np.max(save_dict_arc[key]), 10, key, (1, length, i+1))
    #     for i, key in enumerate(save_dict_straight.keys()):
    #         plot_bins(save_dict_straight[key], np.min(save_dict_straight[key]), np.max(save_dict_straight[key]), 10, key, (1, length, len(save_dict_arc.keys())+i+1))
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(ball_path, "statistics_gt.png"))

    # save the results
    save_dict = {
        "num_arcs": len(errors_arc),
        "num_straights": len(errors_straight),
        "mean_error_arc": np.mean(errors_arc),
        "std_error_arc": np.std(errors_arc),
        "errors_arc": errors_arc,
        "mean_error_straight": np.mean(errors_straight),
        "std_error_straight": np.std(errors_straight),
        "errors_straight": errors_straight,
    }
    for key in save_dict_arc.keys():
        if 'frames' not in key:
            save_dict[f"mean_{key}_arc"] = np.mean(save_dict_arc[key])
            save_dict[f"std_{key}_arc"] = np.std(save_dict_arc[key])
        if 'arc' in key:
            save_dict[f"{key}"] = save_dict_arc[key]
        else:
            save_dict[f"{key}_arc"] = save_dict_arc[key]
    for key in save_dict_straight.keys():
        if 'frames' not in key:
            save_dict[f"mean_{key}_straight"] = np.mean(save_dict_straight[key])
            save_dict[f"std_{key}_straight"] = np.std(save_dict_straight[key])
        if 'straight' in key:
            save_dict[f"{key}"] = save_dict_straight[key]
        else:
            save_dict[f"{key}_straight"] = save_dict_straight[key]

    return save_dict


if __name__ == "__main__":
    statistics_gt_trajectory(ball_path="data/legia-stal-80sec/clip", plot=True)
