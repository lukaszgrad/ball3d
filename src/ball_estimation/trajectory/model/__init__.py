from abc import abstractmethod, ABC
from ball_estimation.trajectory import Trajectory3D
from typing import Tuple, List
import numpy as np
from scipy.integrate import odeint

class SimulationOptions(ABC):
    pass

class MotionParameters(ABC):
    def to_dict(self):
        return self.__dict__

class TrajectoryModel(ABC):
    def __init__(self, options: SimulationOptions):
        self.options = options

    @abstractmethod
    def simulate(self, initial_trajectory: Trajectory3D, motion_parameters: MotionParameters) -> Trajectory3D:
        pass

class UniformAccelerationTrajectoryModel(TrajectoryModel):
    def simulate(self, initial_trajectory: Trajectory3D, motion_parameters: MotionParameters) -> Trajectory3D:
        x, y, z = initial_trajectory.to_xyz()
        z_out = z[0] + motion_parameters.v0z * initial_trajectory.time + 0.5 * motion_parameters.g * initial_trajectory.time**2 
        x_out = x[0] + motion_parameters.v0x * initial_trajectory.time
        y_out = y[0] + motion_parameters.v0x * initial_trajectory.time
        return initial_trajectory.from_xyz(x_out, y_out, z_out, initial_trajectory.time)

class UniformAccelerationMotionParameters(MotionParameters):
    def __init__(self, g=-98, v0x=10, v0z=100, v0y=10):
        self.g = g # dm / s**2
        self.v0x = v0x # dm / s
        self.v0y = v0y # dm / s
        self.v0z = v0z # dm / s
    
    def unpack_x(self):
        return [self.g, self.v0x, self.v0y, self.v0z]
    
    def pack_x(self, x):
        g, v0x, v0y, v0z = x
        self.g = g # dm / s**2
        self.v0x = v0x # dm / s
        self.v0y = v0y # dm / s
        self.v0z = v0z # dm / s
        return self

class ArcMotionParameters(MotionParameters):
    def __init__(self, x0=0., y0=0.,
                 z0=0., vx0=0., vy0=0., vz0=0.,
                 g=0., k3=0., kl=0., ks=0.):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.vx0 = vx0
        self.vy0 = vy0
        self.vz0 = vz0
        self.k3 = k3
        self.g = g
        self.kl = kl
        self.ks = ks

    def estimate_from_bounds(self, optimization_bounds, xh_end, yh_end, fly_time: float):
        for key, value in optimization_bounds.__dict__.items():
            if key in self.__dict__.keys():
                self.__setattr__(key, value)

        # TODO: unify more names with bounds
        self.g = optimization_bounds.gravitation_g
        self.x0 = optimization_bounds.xh_start
        self.y0 = optimization_bounds.yh_start
        self.z0 = 0.0
        self.vx0 = (xh_end - optimization_bounds.xh_start) / fly_time
        self.vy0 = (yh_end - optimization_bounds.yh_start) / fly_time
        self.vz0 = 10 * fly_time * optimization_bounds.gravitation_g / 2

class StraightMotionParameters(MotionParameters):
    def __init__(self, k, x_start, y_start, x_end, y_end):
        self.k = k
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end


class StraightTrajectoryModel:
    def simulate(self, initial_trajectory: Trajectory3D, motion_params):
        x, y, z = self.get_straight_trajectory(**motion_params.__dict__,
                                               num_points=len(initial_trajectory.time))
        out_trajectory = Trajectory3D.from_xyz(x, y, z, initial_trajectory.time)
        return out_trajectory

    @staticmethod
    def get_straight_trajectory(
        x_start: float,
        y_start: float,
        x_end: float,
        y_end: float,
        k: float,
        num_points: int,
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Calculates the trajectory of a ball moving in a straight line between two points.

        Parameters
        ----------
        x_start, y_start: float
            X,Y coordinates of the ball's starting point
        x_end, y_end: float
            X,Y coordinates of the ball's ending point
        k: float
            Resistance coefficient
        num_points: int
            Number of points required in the trajectory

        Returns
        -------
        xt, yt, zt: Tuple[List[float], List[float], List[float]]
            Lists of trajectory points coordinates (zt is a list of zeroes)
        """

        dx = x_end - x_start
        dy = y_end - y_start
        D = np.sqrt(dx ** 2 + dy ** 2)
        cos, sin = dx / D, dy / D

        T = num_points - 1
        t = np.linspace(0, T, num_points)

        v0 = (D + k * (T ** 2)) / T

        # equation of motion: x = x0 + v0*t + a*t^2/2; a = -F/m = const
        xt = [x_start + v0 * t_ * cos - k * t_ * t_ * cos for t_ in t]
        yt = [y_start + v0 * t_ * sin - k * t_ * t_ * sin for t_ in t]
        zt = [0 for t_ in t]  # option: set Z = ball center height (approx 10 cm)

        return xt, yt, zt

class ArcTrajectoryModel:
    motion_parameters_type = ArcMotionParameters
    def simulate(self, initial_trajectory: Trajectory3D, motion_params):
        dt = initial_trajectory.time[1] - initial_trajectory.time[0]
        x, y, z = self.get_arc_trajectory(**motion_params.__dict__,
                                               num_points=len(initial_trajectory.time),
                                            fps=1/dt,
                                         )
        out_trajectory = Trajectory3D.from_xyz(x, y, z, initial_trajectory.time)
        return out_trajectory
    @staticmethod
    def get_arc_trajectory(
        x0, y0, z0, vx0, vy0, vz0, k3, num_points, fps, g, kl=0, ks=0
    ):
        """Gives the ball parabolic trajectory when moving in the air

        Parameters
        ----------
        x0, y0, z0: float
            X,Y,Z coordinates of ball starting point
        vx0, vy0, vz0: float
            starting velocity X,Y,Z coordinates
        k3: float
            air resistance coefficient
        num_points: int
            number of points required in trajectory
        fps: int
            video frames per second
        g: float
            gravitational constant (normally 9.8)
        kl: float
            top-spin coefficient
        ks: float
            side-spin coefficient

        Returns
        -------
        xt, yt, zt: list of float
            lists of trajectory points coordinates
        """

        def dU_dt(U, t):
            # Here U is a vector such that X=U[0], Y=U[1], Z=U[2], Vx=U[3], Vy=U[4], Vz=U[5].
            # This function should return [X', Y', Z', V'x, V'y, V'z]
            V = np.sqrt(U[3] ** 2 + U[4] ** 2 + U[5] ** 2)
            Vp = np.sqrt(U[3] ** 2 + U[4] ** 2)
            return [
                U[3],
                U[4],
                U[5],
                -k3 * V * U[3] + kl * V * U[3] * U[5] / Vp + ks * V * V * U[4] / Vp,
                -k3 * V * U[4] + kl * V * U[4] * U[5] / Vp - ks * V * V * U[3] / Vp,
                -g - k3 * V * U[5] - kl * V * Vp,
            ]

        # solving the differential equation of ball motion (if air resistance ~ V^2):
        # x'' = -k3|v|x' + kl|v|x'z'/|vp| + ks|v|^2y'/|vp|
        # y'' = -k3|v|y' + kl|v|y'z'/|vp| - ks|v|^2x'/|vp|
        # z'' = -g - k3|v|z' - kl|v||vp|
        #
        # Tim G Myers, Sarah L Mitchell
        # A mathematical analysis of the motion of an in-ﬂight soccer ball
        # Article in Sports Engineering · March 2012

        U0 = [
            x0 / 10,
            y0 / 10,
            z0 / 10,
            vx0 / 10,
            vy0 / 10,
            vz0 / 10,
        ]  # decimeters to meters
        t = (1 / fps) * np.linspace(0, num_points - 1, num_points)
        Us = odeint(dU_dt, U0, t)

        xt = list(10 * Us[:, 0])  # meters to decimeters
        yt = list(10 * Us[:, 1])
        zt = list(10 * Us[:, 2])

        return xt, yt, zt
