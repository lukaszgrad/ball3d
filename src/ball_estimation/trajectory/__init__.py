from dataclasses import dataclass, field
import numpy as np
from typing import Optional
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional
import cv2 as cv
from ball_estimation.camera import Camera, project_points_multiple

@dataclass
class Trajectory3D:
    time: np.array
    position: np.array
    velocity: Optional[np.ndarray] = field(default=None, init=False)
    acceleration: Optional[np.ndarray] = field(default=None, init=False)
    angle: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self):
        if len(self.time) != len(self.position):
            raise ValueError("The length of times must be equal to the length of positions")
        
        # TODO: check that the last dimension is equal to 3
        # TODO: Should there be a class that has arbitrary dimensionality and then subclasses?
    
    @classmethod
    def from_xyz(cls, x,y,z,time=None,num_frames=1, fps=1):
        if time is None:
            time = np.arange(num_frames) / fps
        position = np.vstack((x,y,z)).T
        return cls(time, position)
    
    def to_xyz(self):
        x = self.position[:,0]
        y = self.position[:,1]
        z = self.position[:,2]
        return x, y, z
    
    def to_2d_from_camera(self, camera):
        xt, yt, zt = self.to_xyz()
        object_points = np.concatenate(
            (
                np.array(xt).reshape(-1, 1),
                np.array(yt).reshape(-1, 1),
                -np.array(zt).reshape(-1, 1),
            ),
            axis=1,
        )
        model, _ = cv.projectPoints(
            object_points,
            rvec=camera.rotation,
            tvec=camera.translation,
            cameraMatrix=camera.intrinsics,
            distCoeffs=None,
        ).reshape(-1, 2)
        return Trajectory2D(self.time, model)
    
    def to_2d_from_projection_matrices(self, projection_matrices, image_size):
        xt, yt, zt = self.to_xyz()
        points = np.concatenate(
            (
                np.array(xt).reshape(-1, 1),
                np.array(yt).reshape(-1, 1),
                -np.array(zt).reshape(-1, 1),
            ),
            axis=1,
        )
        
        model = project_points_multiple(np.stack(projection_matrices), points)

        # TODO: please, no constants
        model[:, 0] *= image_size.width / 640
        model[:, 1] *= image_size.height / 360
        return Trajectory2D(self.time, model)
        
    def calculate_velocity(self, initial_velocity=None):
        # Calculate differences
        delta_position = np.diff(self.position)
        delta_time = np.diff(self.time)
        
        # Calculate velocity
        velocity = delta_position / delta_time

        # Initialize full velocity array
        full_velocity = np.empty_like(self.position, dtype=float)

        if initial_velocity is not None:
            full_velocity[0] = initial_velocity
        else:
            full_velocity[0] = np.nan

        # Fill the rest of the velocity array
        full_velocity[1:] = velocity
        
        self.velocity = full_velocity

        return full_velocity
    
    def calculate_acceleration(self, initial_velocity, initial_acceleration):
        if self.velocity is None:
            self.calculate_velocity()
        
        # Calculate differences
        delta_velocity = np.diff(self.velocity)
        delta_time = np.diff(self.time)
        
        # Calculate velocity
        acceleration = delta_velocity / delta_time

        # Initialize full velocity array
        full_acceleration = np.empty_like(self.velocity, dtype=float)

        if initial_velocity is not None:
            full_acceleration[0] = initial_acceleration
        else:
            full_acceleration[0] = np.nan

        # Fill the rest of the velocity array
        full_acceleration[1:] = acceleration
        
        self.acceleration = full_acceleration

        return full_acceleration

    def transfer_3Dtrajectory_to_pano(self, camera):
        """converts 3D coordinates of a trajectory to 2D coordinates in a panorama

        Parameters
        ----------
        xt, yt, zt: list of float
            lists of trajectory points coordinates in 3D

        Returns
        -------
        model: (,2) list of float
            lists of trajectory points coordinates in 2D panorama
        """

        xt, yt, zt = self.to_xyz()

        object_points = np.concatenate(
            (
                np.array(xt).reshape(-1, 1),
                np.array(yt).reshape(-1, 1),
                -np.array(zt).reshape(-1, 1),
            ),
            axis=1,
        )
        model, _ = cv.projectPoints(
            object_points,
            rvec=camera.rotation,
            tvec=camera.translation,
            cameraMatrix=camera.intrinsics,
            distCoeffs=None,
        )
        return model.reshape(-1, 2)

    def transfer_3Dtrajectory_to_frames(self, start_frame, camera_smooth_df, video_metadata):
        """converts 3D coordinates of a trajectory to 2D coordinates on frames

        Parameters
        ----------
        xt, yt, zt: list of float
            lists of trajectory points coordinates in 3D
        start_frame: int
            start frame index

        Returns
        -------
        model: (,2) list of float
            lists of trajectory points coordinates on frames
        """
        xt, yt, zt = self.to_xyz()

        cam_numpy = camera_smooth_df.loc[
            (camera_smooth_df.frame_index >= start_frame)
            & (camera_smooth_df.frame_index < start_frame + len(xt)),
            Camera.columns_pandas(),
        ].values
        points = np.concatenate(
            (
                np.array(xt).reshape(-1, 1),
                np.array(yt).reshape(-1, 1),
                -np.array(zt).reshape(-1, 1),
            ),
            axis=1,
        )
        cams = [Camera.from_numpy(c) for c in cam_numpy]
        proj_matrices = [cam.to_projection_matrix() for cam in cams]
        model = project_points_multiple(np.stack(proj_matrices), points)

        model[:, 0] *= video_metadata["width"] / 640
        model[:, 1] *= video_metadata["height"] / 360
        return model

@dataclass
class Trajectory2D:
    time: np.array
    position: np.array
    
    @classmethod
    def from_pandas(cls, df: pd.DataFrame, suffix: Optional[str] = "",
                    start_frame=0, end_frame=np.inf, fps=1):
        df_limited = df[df["file_name"].between(start_frame, end_frame)]
        position = df_limited[[f"x{suffix}",f"y{suffix}"]].values
        # TODO: decide whether time should be scaled and shifted like this
        time = (df_limited["file_name"].values - start_frame) / fps
        return cls(time, position)

