from abc import abstractmethod, ABC
import numpy as np
from ball_estimation.trajectory import Trajectory3D
from ball_estimation.trajectory.model import ArcTrajectoryModel, TrajectoryModel, \
    MotionParameters, StraightTrajectoryModel, StraightMotionParameters, \
    ArcMotionParameters


class LossFunction(ABC):
    def __init__(self, initial_trajectory: Trajectory3D, trajectory_model: TrajectoryModel):
        self.initial_trajectory = initial_trajectory
        self.trajectory_model = trajectory_model

    @abstractmethod
    def compute(self, motion_parameters: MotionParameters) -> float:
        trajectory_estimate = self.trajectory_model.simulate(self.initial_trajectory, motion_parameters)
        return 0.

class Loss2D(LossFunction):
    def __init__(self, initial_trajectory: Trajectory3D, trajectory_model: TrajectoryModel,
                 gt_image, projection_matrices, image_size):
        self.projection_matrices = projection_matrices
        self.image_size = image_size
        self.gt_image = gt_image
        super().__init__(initial_trajectory, trajectory_model)
        
    def compute(self, motion_parameters: MotionParameters):
        simulated_trajectory = self.trajectory_model.simulate(self.initial_trajectory, motion_parameters)
        projected_trajectory = simulated_trajectory.to_2d_from_projection_matrices(self.projection_matrices, self.image_size)
        loss = np.nanmean((projected_trajectory.position - self.gt_image.position)**2)
        return loss

class ArcLossFunction:
    def __init__(self, camera, camera_smooth_df, video_metadata,
                 ballsize_relative_loss,
                 weight_end_arc, weight_zpenalty_arc,
                 ball_median_size, nans_allowed, model):
        self.camera = camera
        self.camera_smooth_df = camera_smooth_df
        self.video_metadata = video_metadata
        self.ballsize_relative_loss = ballsize_relative_loss
        self.weight_end_arc = weight_end_arc
        self.weight_zpenalty_arc = weight_zpenalty_arc
        self.ball_median_size = ball_median_size
        self.nans_allowed = nans_allowed
        self.model = model

    def fcn2min(self, params, num_points, data, xh_end, yh_end, start_frame, endpoint):
        T = num_points - 1
        fps = float(self.video_metadata["fps"])
        t = (1 / fps) * np.linspace(0, T, num_points)
        # TODO: clean this up
        inital_trajectory = Trajectory3D(time=t, position=t)
        motion_params = self.model.motion_parameters_type(**{key: params[key].value for key in params.keys()})
        out_trajectory = self.model.simulate(inital_trajectory, motion_params)

        xt, yt, zt = out_trajectory.to_xyz()

        # Map the ball trajectory to a screen, either as a broadcast image
        # or a panorama image, based on the estimator settings.
        # This allows us to calculate the deviation of a model from the actual observed trajectory.
        model = out_trajectory.transfer_3Dtrajectory_to_frames(start_frame,
                                                                self.camera_smooth_df,
                                                                self.video_metadata)

        diff = np.abs(data[:, :2] - model)
        if self.ballsize_relative_loss:
            diff[:, 0] /= data[:, 2]
            diff[:, 1] /= data[:, 2]

        # We aim to penalize the optimizer for accumulating a large discrepancy
        # as it approaches the final point of the trajectory.
        # The `add_end` parameter controls how much we penalize the optimizer.
        add_end = int(self.weight_end_arc * len(data) + 0.5)

        # We aim to penalize our model if it unreasonably ends the trajectory
        # at a height z >> 0 as well as z < 0 which should be impossible.
        # The `add_zpenalty` parameter controls how much we penalize the optimizer.
        add_zpenalty = int(self.weight_zpenalty_arc * len(data) + 0.5)

        # Calculate the discrepancy at the final point of the trajectory.
        # In cases like shots fitting, where we lack accurate ball (X, Y) pitch coordinates
        # due to projecting a high ball onto the pitch, we use screen discrepancy instead.
        endpoint_shift = (
            np.array([xh_end - xt[-1], yh_end - yt[-1]])
            if endpoint
            else np.array([data[-1][0] - model[-1][0], data[-1][1] - model[-1][1]])
        ) / (self.ball_median_size if self.ballsize_relative_loss else 1)

        # When optimizing an endpoint (e.g., if it's not a shot), the first formula calculates
        # a loss of 0.5*z if 0 < z < 2.4m. Otherwise, the loss grows rapidly (e.g., 100*z).
        # For shots, we relax the restriction, and the second formula gives a loss of 0.5*z
        # if 0 < z < 10m, with a similarly fast-growing loss otherwise.
        z_penalty = (
            np.array([0, max(0.5 * zt[-1], 100 * (np.abs(zt[-1] - 12) - 12))]) # a lot of constants
            if endpoint
            else np.array([0, max(0.5 * zt[-1], 100 * (np.abs(zt[-1] - 50) - 50))]) # even more constants
        ) / (self.ball_median_size if self.ballsize_relative_loss else 1)

        # Allow a certain percentage of NaNs in observed data
        # (frames without detected and smoothed ball).
        if np.isnan(diff).any(axis=1).mean() < self.nans_allowed:
            diff = diff[~np.isnan(diff).any(axis=1)]

        # Finally, we concatenate additional penalties to the loss vector:
        # 'add_end' times 'endpoint_shift` and 'add_zpenalty' times `z_penalty`.
        return np.concatenate(
            [diff, add_end * [endpoint_shift], add_zpenalty * [z_penalty]]
        )

    def update(self, camera, camera_smooth_df, video_metadata):
        self.camera = camera
        self.camera_smooth_df = camera_smooth_df
        self.video_metadata = video_metadata

class StraightLossFunction:
    def __init__(self, camera, camera_smooth_df, video_metadata,
                 ballsize_relative_loss, weight_start_straight,
                 weight_end_straight, nans_allowed, model):
        self.camera = camera
        self.camera_smooth_df = camera_smooth_df
        self.video_metadata = video_metadata
        self.ballsize_relative_loss = ballsize_relative_loss
        self.weight_start_straight = weight_start_straight
        self.weight_end_straight = weight_end_straight
        self.nans_allowed = nans_allowed
        self.model = model

    def fcn2min(self, params, num_points, data, start_frame):
        T = num_points - 1
        t = np.linspace(0, T, num_points)
        # TODO: clean this up
        inital_trajectory = Trajectory3D(time=t, position=t)
        motion_params = StraightMotionParameters(**{key: params[key].value for key in params.keys()})
        out_trajectory = self.model.simulate(inital_trajectory, motion_params)

        # Map the ball trajectory to a screen, either as a broadcast image
        # or a panorama image, based on the estimator settings.
        # This allows us to calculate the deviation of a model from the actual observed trajectory.
        model = out_trajectory.transfer_3Dtrajectory_to_frames(start_frame,
                                                                self.camera_smooth_df,
                                                                self.video_metadata)
        diff = np.abs(data[:, :2] - model)
        if self.ballsize_relative_loss:
            diff[:, 0] /= data[:, 2]
            diff[:, 1] /= data[:, 2]

        # Initial estimates for start and end points may not be perfect.
        # Therefore we allow slight shifts within 'allowed_endpoints_shift_straight'.
        # To penalize excessive movement, we add terms to the RMSE loss.
        # These terms reflect discrepancies at start and end points.

        # Those vector lengths (add_start, add_end) depend on
        # 'weight_start_straight' and 'weight_end_straight'.
        # These reflect how much we want to penalize start and end discrepancies.
        add_start = int(self.weight_start_straight * len(data) + 0.5) # np.ceil?
        add_end = int(self.weight_end_straight * len(data) + 0.5)

        # Allow a certain percentage of NaNs in observed data
        # (frames without detected and smoothed ball).
        if np.isnan(diff).any(axis=1).mean() < self.nans_allowed:
            diff = diff[~np.isnan(diff).any(axis=1)]

        # The optimized function returns a [2,N]-shaped vector to minimize its norm.
        # We concatenate vectors for start and end discrepancies.
        return np.concatenate([diff, add_start * [diff[0]], add_end * [diff[-1]]])

    def update(self, camera, camera_smooth_df, video_metadata):
        self.camera = camera
        self.camera_smooth_df = camera_smooth_df
        self.video_metadata = video_metadata
