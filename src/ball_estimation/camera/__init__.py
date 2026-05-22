# Author: Lukasz Grad

from __future__ import annotations

import cv2 as cv
import numpy as np
import pandas as pd
from numba import jit

from ball_estimation.data.model.geometry import Point2D, Rectangle, Size2D


class Camera:
    """Pin-hole camera model

    TODO: Move this implementation to `respo.core.data.model.pitch_geom`
    TODO: and rewrite it to inherit from `pydantic.BaseModel`.

    """

    def __init__(self, intrinsics, rotation, translation):
        self.intrinsics = intrinsics
        self.rotation = rotation
        self.translation = translation
        self._normalize_parameters()

    def _normalize_parameters(self):
        """Normalize camera parameters"""
        self._normalize_intrinsics()
        # TODO - rotation should be a property but there is
        #        a problem with old pickled camera objects
        self.rotation = self._normalize_rotation(self.rotation)

    def _normalize_intrinsics(self):
        """Normalize intrinsics matrix so that the focal lengths are positive"""
        if self.intrinsics[0, 0] < 0:
            # If the focal length is negative, we negate the focal lengths
            # and propagate the negation to the extrinsics
            scale_matrix = np.diag([-1, -1, 1])
            self.intrinsics = self.intrinsics @ scale_matrix
            self.rotation = cv.Rodrigues(scale_matrix @ cv.Rodrigues(self.rotation)[0])[
                0
            ]
            self.translation = scale_matrix @ self.translation

    @staticmethod
    def _normalize_rotation(rotation):
        """Normalize Rodriguez rotation vector so that the angle is in [0, pi]"""
        rot_theta = np.linalg.norm(rotation)
        if rot_theta > np.pi:
            # If the angle is greater than pi, we negate the unit rotation
            # vector and subtract the angle from 2pi
            unit_rot = rotation / rot_theta
            rot_theta = rot_theta % (2 * np.pi)
            rot_theta = (2 * np.pi) - rot_theta
            rotation = -unit_rot * rot_theta
        return rotation

    def get_pos(self):
        return -cv.Rodrigues(self.rotation)[0].T @ self.translation

    def get_pixel_aspect_ratio(self):
        return self.intrinsics[0, 0] / self.intrinsics[1, 1]

    def get_height_angle(self, reference_point: np.ndarray) -> float:
        """Compute height angle in degrees between camera position and reference point.

        Based on the ratio between the height of the camera and the distance from the camera
        to the reference point on 2D pitch plane.
        """
        pos = self.get_pos().squeeze()
        angle = np.arctan(-pos[2] / np.linalg.norm(pos[:2] - reference_point[:2]))
        return np.rad2deg(angle)

    def get_center_angle(self, reference_point: np.ndarray) -> float:
        """Compute center angle in degrees between camera position and reference point.

        Based on the ratio between the change in x coordinate to the change in y coordinate.
        """
        pos = self.get_pos().squeeze()
        angle = np.arctan(
            np.abs(pos[0] - reference_point[0]) / np.abs(pos[1] - reference_point[1])
        )
        return np.rad2deg(angle)

    def get_relative_scale(self, points):
        H_inv = self.to_inverse_homography()
        return get_relative_scale(H_inv, points)

    def get_relative_world_scale(
        self, world_points: np.ndarray, image_size: Size2D | None = None
    ) -> np.ndarray:
        """Compute relative scale of the world points in the camera coordinate system.

        Parameters
        ----------
        world_point : np.ndarray
            A set of `n` points with shape (`n`, 3).
        image_size : Size2D, optional
            Size of the image on which the camera estimation was performed on, if
            provided, estimation is performed only for world points that backproject
            inside the image.

        Returns
        -------
        relative_scales: np.ndarray
            An array of shape (`n`, 2) of scaling factors that convert pixels to world units.
            First column contains the width factors and the second column contains the height
            factors.
        """
        pos = self.get_pos().squeeze()

        # Compute orthogonal unit vectors to the world_points - camera position vectors
        points_pos = world_points - pos
        orthogonal_points_pos = np.stack((-points_pos[:, 1], points_pos[:, 0]), axis=1)
        orthogonal_points_pos = orthogonal_points_pos / np.linalg.norm(
            orthogonal_points_pos, axis=1, keepdims=True
        )
        orthogonal_points_pos = np.append(
            orthogonal_points_pos, np.zeros_like(orthogonal_points_pos[:, :1]), axis=1
        )

        # Compute world points used for relative width scale estimation
        width_points_left = world_points - orthogonal_points_pos * 0.5
        width_points_right = world_points + orthogonal_points_pos * 0.5
        width_scale = _get_relative_world_scale(
            self.to_projection_matrix(),
            width_points_left,
            width_points_right,
            image_size=image_size,
        )

        # Compute world points used for relative height scale estimation
        height_points_top = world_points - np.array([0, 0, 1])
        height_scale = _get_relative_world_scale(
            self.to_projection_matrix(),
            height_points_top,
            world_points,
            image_size=image_size,
        )

        return np.stack((width_scale, height_scale), axis=1)

    def to_projection_matrix(self):
        proj_matrix = np.zeros((3, 4))
        proj_matrix[:3, :3] = cv.Rodrigues(self.rotation)[0]
        proj_matrix[:3, 3:] = self.translation
        proj_matrix = self.intrinsics @ proj_matrix
        return proj_matrix

    def to_homography(self):
        """Compute homography corresponding to intrinsic and extrinsic parameters"""
        R, _ = cv.Rodrigues(self.rotation)
        R[:, 2:] = self.translation
        return combine_matrix(self.intrinsics, R)

    def to_inverse_homography(self):
        """Inverse homography corresponding to intrinsic and extrinsic parameters"""
        try:
            return inverse_matrix(self.to_homography())
        except np.linalg.LinAlgError:
            return np.full((3, 3), np.nan)

    def to_numpy(self):
        intrinsics = self.intrinsics[[0, 1, 0, 1], [0, 1, 2, 2]]
        return np.concatenate(
            (
                np.squeeze(self.rotation),
                np.squeeze(self.translation),
                intrinsics,
            )
        )

    @staticmethod
    def from_numpy(x):
        intrinsics = np.zeros((3, 3), dtype=np.float64)
        intrinsics[2, 2] = 1.0
        intrinsics[[0, 1, 0, 1], [0, 1, 2, 2]] = x[-4:]
        return Camera(
            intrinsics=intrinsics,
            rotation=x[:3].reshape((3, 1)),
            translation=x[3:6].reshape((3, 1)),
        )

    @staticmethod
    def columns_pandas():
        return [
            "rot_x",
            "rot_y",
            "rot_z",
            "tx",
            "ty",
            "tz",
            "fx",
            "fy",
            "princ_x",
            "princ_y",
        ]

    def to_pandas(self):
        values = self.to_numpy()
        return pd.Series(values, index=Camera.columns_pandas())

    @staticmethod
    def from_pandas(x):
        return Camera.from_numpy(x[Camera.columns_pandas()].values)

    @staticmethod
    def empty_pandas():
        return pd.Series(np.full((10,), np.nan), index=Camera.columns_pandas())


def project_points(projection_matrix, points):
    """Project a set of points using given projection matrix

    Parameters
    ----------
    projection_matrix : np.ndarray
        A projection matrix with shape (3, 4)
    points : np.ndarray
        A set of `n` points with shape (`n`, 3)

    Returns
    -------
    points_projected : np.ndarray
        A set of projected points with shape (`n`, 2)
    """
    points = np.concatenate((points, np.ones_like(points[:, :1])), axis=1)
    points_projected = (projection_matrix @ points.T).T
    points_projected /= points_projected[:, 2:]
    return points_projected[:, :2]


def project_points_multiple(projection_matrices, points):
    """Project a set of points using a corresponding set of matrices

    Parameters
    ----------
    projection_matrices : np.ndarray
        A set of `n` projection matrices with shape (`n`, 3, 4) or (`n`, 3, 3)
    points : np.ndarray
        A set of `n` points with shape (`n`, 3) or (`n`, 2) respectively

    Returns
    -------
    points_projected : np.ndarray
        A set of projected points with shape (`n`, 2). Each point is projected with
        a corresponding projection matrix in a row-wise fashion
    """
    points = np.concatenate((points, np.ones_like(points[:, :1])), axis=1)
    points_projected = np.matmul(projection_matrices, points[..., np.newaxis])
    points_projected = points_projected[..., 0]
    points_projected /= points_projected[:, 2:]
    return points_projected[:, :2]


def combine_matrix(h1, h2):
    """Combine two homography matrices into a single one.

    Parameters
    ----------
    h1: np.ndarray
        A homography matrix with shape (3, 3).
    h2: np.ndarray
        A second homography matrix with shape (3, 3).

    Returns
    -------
    h_comb: np.ndarray
        A combined homography matrix.
    """
    h = h1 @ h2
    h /= h[2, 2]
    return h


@jit(nopython=True)
def get_relative_scale(H_inv, points):
    """Compute a scaling factor that converts pixels to world units
       given two image points (in pixels) assuming that the points
       lie on the pitch plane (z=0)

    Parameters
    ----------
    H_inv: np.ndarray
        An inverse homography matrix with shape (3, 3).
    points: np.ndarray
        Location of image points (units: pixels, shape (2,3))

    Returns
    -------
    relative_scale: float
        A scaling factor that converts pixels to world units.
    """
    points_world = warp_points_H(points, H_inv)
    distance_in_world_units = np.linalg.norm(points_world[0] - points_world[1])
    distance_in_pixels = np.linalg.norm(points[0] - points[1])
    relative_scale = distance_in_world_units / distance_in_pixels
    return relative_scale


@jit(nopython=True)
def inverse_matrix(H):
    """Compute inverse of a given homography.

    Parameters
    ----------
    H: np.ndarray
        A homography matrix with shape (3, 3).

    Returns
    -------
    H_inv: np.ndarray
        An inverse homography matrix.
    """
    H_inv = np.linalg.inv(H)
    H_inv = H_inv / H_inv[2, 2]
    return H_inv


def translate_matrix(x, y):
    """Create a homography matrix from a given translation vector.

    Parameters
    ----------
    x: float
        Translation in x direction.
    y: float
        Translation in y direction.

    Returns
    -------
    H: np.ndarray
        A homography matrix that translates coordinates by a given vector.
    """
    translate_h = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=np.float32)
    return translate_h


def combine_matrix_multiple(h1, h2):
    """Combine homography matrix and a vector of homography matrices
       into a vector of homography matrices.

    Parameters
    ----------
    h1: np.ndarray
        A homography matrix with shape (3, 3).
    h2: np.ndarray
        A vector of homography matrices with shape (N, 3, 3).

    Returns
    -------
    h_comb: np.ndarray
        A vector of combined homography matrices.
    """
    h = h1 @ h2
    h /= h[:, 2:, 2:]
    return h


@jit(nopython=True)
def warp_points_H(X, H):
    """Warp an array of points using a given homography matrix.

    Parameters
    ----------
    X: np.ndarray
        Array on `n` point coordinates with shape (`n`, 2) in (x, y) format.
    H: np.ndarray
        A homography matrix with shape (3, 3).

    Returns
    -------
    X_warped: np.ndarray
        Array of `n` warped point coordinates with shape (`n`, 2) in (x, y) format.
    """
    X = X.astype(H.dtype)
    X_aug = np.concatenate((X, np.ones_like(X[:, 0:1])), axis=1)  # (N, 3)
    T = (H @ X_aug.T).T  # (N, 3)
    T = T / T[:, 2:]
    return T[:, :2]


def scale_matrix(sx, sy):
    """ "Create a homography matrix from a given scaling vector.

    Parameters
    ----------
    x: float
        Scaling factor in x direction.
    y: float
        Scaling factor in y direction.

    Returns
    -------
    H: np.ndarray
        A homography matrix that scales coordinates by given scale factors.
    """
    translate_h = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32)
    return translate_h
